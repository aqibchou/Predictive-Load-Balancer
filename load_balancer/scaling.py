import asyncio
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import subprocess
import json

QTABLE_PATH      = "qtable_checkpoint.json"
K8S_SCALING      = os.getenv("K8S_SCALING_ENABLED", "false").lower() == "true"
K8S_NAMESPACE    = os.getenv("POD_NAMESPACE", "default")
K8S_BACKEND_DEPL = os.getenv("K8S_BACKEND_DEPLOYMENT", "backend")

LEARNING_RATE   = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05
EPSILON_DECAY   = 0.995

MIN_SERVERS            = 1
MAX_SERVERS            = 5
SCALE_COOLDOWN_SECONDS = 60

LATENCY_CRITICAL_PENALTY  = -10
LATENCY_WARNING_PENALTY   = -5
SERVER_COST_PENALTY       = -0.5
SLA_BONUS                 = 5
ACTION_PENALTY            = -1
PREDICTION_ACCURATE_BONUS = 2.0


class Action(Enum):
    SCALE_UP   = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD       = "hold"

class LoadLevel(Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

class PredictionTrend(Enum):
    DECREASING = "decreasing"
    STABLE     = "stable"
    INCREASING = "increasing"

class LatencyStatus(Enum):
    OK       = "ok"
    WARNING  = "warning"
    CRITICAL = "critical"

class UncertaintyLevel(Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


@dataclass
class SystemState:
    current_servers:   int
    load_level:        LoadLevel
    prediction_trend:  PredictionTrend
    latency_status:    LatencyStatus
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.MEDIUM

    def to_tuple(self) -> Tuple:
        return (
            self.current_servers,
            self.load_level.value,
            self.prediction_trend.value,
            self.latency_status.value,
            self.uncertainty_level.value,
        )


@dataclass
class RawMetrics:
    current_servers:      int
    avg_load_percent:     float
    predicted_load_5min:  float
    current_load:         float
    p95_latency_ms:       float
    prediction_uncertainty: float


async def execute_scaling_action(action: Action, current_count: int) -> int:
    if action == Action.SCALE_UP:
        new_count = min(current_count + 1, MAX_SERVERS)
    elif action == Action.SCALE_DOWN:
        new_count = max(current_count - 1, MIN_SERVERS)
    else:
        return current_count

    if new_count == current_count:
        return current_count

    if K8S_SCALING:
        return await _k8s_scale(new_count, current_count)
    return await _docker_scale(new_count, current_count)


async def _k8s_scale(new_count: int, current_count: int) -> int:
    """Scale via the Kubernetes Python API (no kubectl binary required)."""
    def _do_scale():
        from kubernetes import client, config as k8s_config
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        client.AppsV1Api().patch_namespaced_deployment_scale(
            name=K8S_BACKEND_DEPL,
            namespace=K8S_NAMESPACE,
            body={"spec": {"replicas": new_count}},
        )

    for attempt in range(3):
        try:
            await asyncio.to_thread(_do_scale)
            print(f"Scaled to {new_count} servers via Kubernetes API")
            return new_count
        except Exception as e:
            print(f"K8s scale attempt {attempt + 1} error: {e}")
        if attempt < 2:
            await asyncio.sleep(1)

    print(f"All K8s scale attempts failed, staying at {current_count}")
    return current_count


async def _docker_scale(new_count: int, current_count: int) -> int:
    """Scale via docker-compose subprocess (local / Docker Compose deployments)."""
    cmd = [
        "docker-compose", "-f", "/mnt/project/docker-compose.yml",
        "-p", "predictive-load-balancer",
        "up", "-d", "--scale", f"backend={new_count}",
        "--no-recreate", "--no-deps", "backend",
    ]

    for attempt in range(3):
        try:
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"Scaled to {new_count} servers")
                return new_count
            print(f"Scale attempt {attempt + 1} failed: {result.stderr}")
        except Exception as e:
            print(f"Scale attempt {attempt + 1} error: {e}")
        if attempt < 2:
            await asyncio.sleep(1)

    print(f"All scale attempts failed, staying at {current_count}")
    return current_count


class QLearningScaler:
    def __init__(self):
        self.q_table: Dict[Tuple, Dict[Action, float]] = defaultdict(
            lambda: {action: 0.0 for action in Action}
        )
        self.epsilon           = EPSILON_START
        self.current_servers   = 3
        self.last_scale_time:  Optional[datetime] = None
        self.is_running        = False
        self._task:            Optional[asyncio.Task] = None
        self.decision_history: List[Dict] = []
        self.last_state:       Optional[SystemState] = None
        self.last_action:      Optional[Action] = None
        self._prev_predicted_load: Optional[float] = None

    async def save_qtable(self):
        from database import save_qtable_to_db
        data = {
            "qtable": {
                str(k): {a.value: round(v, 6) for a, v in actions.items()}
                for k, actions in self.q_table.items()
            },
            "epsilon": self.epsilon,
        }
        with open(QTABLE_PATH, "w") as f:
            json.dump(data, f)
        rows = [
            (str(state_key), action.value, q_value)
            for state_key, actions in self.q_table.items()
            for action, q_value in actions.items()
        ]
        await save_qtable_to_db(rows)
        print(f"Q-table saved. States: {len(self.q_table)}, epsilon: {self.epsilon:.3f}")

    async def load_qtable(self):
        from database import load_qtable_from_db
        db_rows = await load_qtable_from_db()
        if db_rows:
            self.q_table = defaultdict(lambda: {action: 0.0 for action in Action})
            for row in db_rows:
                self.q_table[eval(row['state_key'])][Action(row['action'])] = row['q_value']
            print(f"Q-table loaded from DB. States: {len(self.q_table)}, epsilon: {self.epsilon:.3f}")
            return
        if os.path.exists(QTABLE_PATH):
            with open(QTABLE_PATH) as f:
                data = json.load(f)
            self.q_table = defaultdict(lambda: {action: 0.0 for action in Action})
            for k_str, actions_dict in data["qtable"].items():
                state_key = eval(k_str)
                for a_str, v in actions_dict.items():
                    try:
                        self.q_table[state_key][Action(a_str)] = v
                    except ValueError:
                        pass
            self.epsilon = min(data.get("epsilon", EPSILON_START), EPSILON_MIN)
            print(f"Q-table loaded from JSON. States: {len(self.q_table)}, epsilon: {self.epsilon:.3f}")
        else:
            print("No checkpoint found, starting fresh.")

    def reset_qtable(self):
        self.q_table = defaultdict(lambda: {action: 0.0 for action in Action})
        self.epsilon = EPSILON_START
        print("Q-table reset.")

    def discretize_load(self, load_percent: float) -> LoadLevel:
        if load_percent < 30:   return LoadLevel.LOW
        if load_percent < 70:   return LoadLevel.MEDIUM
        return LoadLevel.HIGH

    def discretize_trend(self, predicted: float, current: float) -> PredictionTrend:
        if current == 0:
            return PredictionTrend.STABLE
        change = (predicted - current) / current * 100
        if change < -10:  return PredictionTrend.DECREASING
        if change >  10:  return PredictionTrend.INCREASING
        return PredictionTrend.STABLE

    def discretize_latency(self, p95_latency_ms: float) -> LatencyStatus:
        if p95_latency_ms < 100:  return LatencyStatus.OK
        if p95_latency_ms < 200:  return LatencyStatus.WARNING
        return LatencyStatus.CRITICAL

    def discretize_uncertainty(self, uncertainty: float) -> UncertaintyLevel:
        if uncertainty < 20:   return UncertaintyLevel.LOW
        if uncertainty <= 50:  return UncertaintyLevel.MEDIUM
        return UncertaintyLevel.HIGH

    def get_state(self, metrics: RawMetrics) -> SystemState:
        return SystemState(
            current_servers   = metrics.current_servers,
            load_level        = self.discretize_load(metrics.avg_load_percent),
            prediction_trend  = self.discretize_trend(metrics.predicted_load_5min, metrics.current_load),
            latency_status    = self.discretize_latency(metrics.p95_latency_ms),
            uncertainty_level = self.discretize_uncertainty(metrics.prediction_uncertainty),
        )

    def calculate_reward(self, state_before: SystemState, action: Action,
                         state_after: SystemState, metrics_after: RawMetrics) -> float:
        reward = 0.0
        if state_after.latency_status == LatencyStatus.CRITICAL:
            reward += LATENCY_CRITICAL_PENALTY
        elif state_after.latency_status == LatencyStatus.WARNING:
            reward += LATENCY_WARNING_PENALTY
        reward += state_after.current_servers * SERVER_COST_PENALTY
        if state_after.latency_status == LatencyStatus.OK and state_after.current_servers <= 3:
            reward += SLA_BONUS
        if action != Action.HOLD:
            reward += ACTION_PENALTY
        if self._prev_predicted_load and self._prev_predicted_load > 0:
            error_pct = abs(metrics_after.current_load - self._prev_predicted_load) / max(self._prev_predicted_load, 1)
            if error_pct < 0.25:
                reward += PREDICTION_ACCURATE_BONUS
        return reward

    def select_action(self, state: SystemState) -> Action:
        if self.last_scale_time:
            if (datetime.now() - self.last_scale_time).total_seconds() < SCALE_COOLDOWN_SECONDS:
                return Action.HOLD
        valid = list(Action)
        if self.current_servers >= MAX_SERVERS:  valid.remove(Action.SCALE_UP)
        if self.current_servers <= MIN_SERVERS:  valid.remove(Action.SCALE_DOWN)
        if random.random() < self.epsilon:
            return random.choice(valid)
        q_values = self.q_table[state.to_tuple()]
        return max({a: q_values[a] for a in valid}, key=lambda a: q_values[a])

    def update_q_table(self, state: SystemState, action: Action,
                       reward: float, next_state: SystemState):
        sk, nsk = state.to_tuple(), next_state.to_tuple()
        current_q  = self.q_table[sk][action]
        max_next_q = max(self.q_table[nsk].values())
        self.q_table[sk][action] = current_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * max_next_q - current_q
        )
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    async def make_decision(self, metrics: RawMetrics, router=None) -> Dict:
        current_state = self.get_state(metrics)
        if self.last_state and self.last_action:
            reward = self.calculate_reward(self.last_state, self.last_action, current_state, metrics)
            self.update_q_table(self.last_state, self.last_action, reward, current_state)

        action       = self.select_action(current_state)
        action_taken = False
        if action != Action.HOLD:
            new_count = await execute_scaling_action(action, self.current_servers)
            if new_count != self.current_servers:
                self.current_servers  = new_count
                self.last_scale_time  = datetime.now()
                action_taken          = True

        self.last_state            = current_state
        self.last_action           = action
        self._prev_predicted_load  = metrics.predicted_load_5min

        decision = {
            "timestamp":    datetime.now().isoformat(),
            "state":        {
                "servers":     current_state.current_servers,
                "load":        current_state.load_level.value,
                "trend":       current_state.prediction_trend.value,
                "latency":     current_state.latency_status.value,
                "uncertainty": current_state.uncertainty_level.value,
            },
            "action":       action.value,
            "action_taken": action_taken,
            "epsilon":      round(self.epsilon, 3),
        }
        self.decision_history.append(decision)
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        return decision

    def get_q_table_summary(self) -> Dict:
        non_zero = sum(1 for a in self.q_table.values() if any(v != 0.0 for v in a.values()))
        return {
            'total_states':    len(self.q_table),
            'non_zero_states': non_zero,
            'epsilon':         round(self.epsilon, 3),
            'current_servers': self.current_servers,
            'states': {
                str(k): {
                    'best_action': max(actions, key=actions.get).value,
                    'q_values':    {a.value: round(v, 3) for a, v in actions.items()},
                }
                for k, actions in self.q_table.items()
            },
        }


scaler = QLearningScaler()
