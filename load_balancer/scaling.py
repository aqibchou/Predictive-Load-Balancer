import asyncio
import pickle
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

import subprocess
import json

# Learned Q table with epsilon 0.1
QTABLE_PATH = "qtable_checkpoint.json"

# Q-Learning hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995

# Scaling constraints
MIN_SERVERS = 1
MAX_SERVERS = 5
SCALE_COOLDOWN_SECONDS = 60

# Reward weights
LATENCY_CRITICAL_PENALTY = -10
LATENCY_WARNING_PENALTY = -5
SERVER_COST_PENALTY = -0.5
SLA_BONUS = 5
ACTION_PENALTY = -1


class Action(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"


class LoadLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionTrend(Enum):
    DECREASING = "decreasing"
    STABLE = "stable"
    INCREASING = "increasing"


class LatencyStatus(Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SystemState:
    current_servers: int
    load_level: LoadLevel
    prediction_trend: PredictionTrend
    latency_status: LatencyStatus

    def to_tuple(self) -> Tuple:
        return (
            self.current_servers,
            self.load_level.value,
            self.prediction_trend.value,
            self.latency_status.value,
        )


@dataclass
class RawMetrics:
    current_servers: int
    avg_load_percent: float
    predicted_load_5min: float
    current_load: float
    p95_latency_ms: float
    prediction_uncertainty: float


def execute_scaling_action(action: Action, current_count: int) -> int:
    if action == Action.SCALE_UP:
        new_count = min(current_count + 1, MAX_SERVERS)
    elif action == Action.SCALE_DOWN:
        new_count = max(current_count - 1, MIN_SERVERS)
    else:
        return current_count

    if new_count == current_count:
        return current_count

    try:
        result = subprocess.run(
            [
                "docker-compose",
                "-f",
                "/mnt/project/docker-compose.yml",
                "-p",
                "project-group-101",
                "up",
                "-d",
                "--scale",
                f"backend={new_count}",
                "--no-recreate",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print(f"Successfully scaled to {new_count} servers")
            return new_count
        else:
            print(f"Scale failed: {result.stderr}")
            return current_count

    except Exception as e:
        print(f"Scale failed: {e}")
        return current_count


class QLearningScaler:
    def __init__(self):
        self.q_table: Dict[Tuple, Dict[Action, float]] = defaultdict(
            lambda: {action: 0.0 for action in Action}
        )

        self.epsilon = EPSILON_START
        self.current_servers = 3
        self.last_scale_time: Optional[datetime] = None
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

        self.decision_history: List[Dict] = []
        self.last_state: Optional[SystemState] = None
        self.last_action: Optional[Action] = None

    def save_qtable(self):
        data = {
            "qtable": {str(k): v for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
        }
        with open(QTABLE_PATH, "w") as f:
            json.dump(data, f)
        print(f"Q-table saved. States: {len(self.q_table)}, epsilon: {self.epsilon:.3f}")

    def load_qtable(self):
        if os.path.exists(QTABLE_PATH):
            with open(QTABLE_PATH) as f:
                data = json.load(f)
            self.q_table = defaultdict(
                lambda: {action: 0.0 for action in Action},
                {eval(k): v for k, v in data["qtable"].items()},
            )
            self.epsilon = data["epsilon"]
            print(f"Q-table loaded. States: {len(self.q_table)}, epsilon: {self.epsilon:.3f}")
        else:
            print("No checkpoint found, starting fresh.")

    def reset_qtable(self):
        self.q_table = defaultdict(
            lambda: {action: 0.0 for action in Action}
        )
        self.epsilon = EPSILON_START
        print("Q-table reset. Starting from scratch.")

    def discretize_load(self, load_percent: float) -> LoadLevel:
        if load_percent < 30:
            return LoadLevel.LOW
        elif load_percent < 70:
            return LoadLevel.MEDIUM
        return LoadLevel.HIGH

    def discretize_trend(self, predicted: float, current: float) -> PredictionTrend:
        if current == 0:
            return PredictionTrend.STABLE

        change_percent = (predicted - current) / current * 100

        if change_percent < -10:
            return PredictionTrend.DECREASING
        elif change_percent > 10:
            return PredictionTrend.INCREASING
        return PredictionTrend.STABLE

    def discretize_latency(self, p95_latency_ms: float) -> LatencyStatus:
        if p95_latency_ms < 100:
            return LatencyStatus.OK
        elif p95_latency_ms < 200:
            return LatencyStatus.WARNING
        return LatencyStatus.CRITICAL

    def get_state(self, metrics: RawMetrics) -> SystemState:
        return SystemState(
            current_servers=metrics.current_servers,
            load_level=self.discretize_load(metrics.avg_load_percent),
            prediction_trend=self.discretize_trend(
                metrics.predicted_load_5min,
                metrics.current_load,
            ),
            latency_status=self.discretize_latency(metrics.p95_latency_ms),
        )

    def calculate_reward(
        self,
        state_before: SystemState,
        action: Action,
        state_after: SystemState,
        metrics_after: RawMetrics,
    ) -> float:
        reward = 0.0

        if state_after.latency_status == LatencyStatus.CRITICAL:
            reward += LATENCY_CRITICAL_PENALTY
        elif state_after.latency_status == LatencyStatus.WARNING:
            reward += LATENCY_WARNING_PENALTY

        reward += state_after.current_servers * SERVER_COST_PENALTY

        if (
            state_after.latency_status == LatencyStatus.OK
            and state_after.current_servers <= 3
        ):
            reward += SLA_BONUS

        if action != Action.HOLD:
            reward += ACTION_PENALTY

        return reward

    def select_action(self, state: SystemState) -> Action:
        if self.last_scale_time:
            time_since_scale = (
                datetime.now() - self.last_scale_time
            ).total_seconds()
            if time_since_scale < SCALE_COOLDOWN_SECONDS:
                return Action.HOLD

        valid_actions = list(Action)

        if self.current_servers >= MAX_SERVERS:
            valid_actions.remove(Action.SCALE_UP)
        if self.current_servers <= MIN_SERVERS:
            valid_actions.remove(Action.SCALE_DOWN)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_key = state.to_tuple()
        q_values = self.q_table[state_key]
        valid_q = {a: q_values[a] for a in valid_actions}
        return max(valid_q, key=valid_q.get)

    def update_q_table(
        self,
        state: SystemState,
        action: Action,
        reward: float,
        next_state: SystemState,
    ):
        state_key = state.to_tuple()
        next_state_key = next_state.to_tuple()

        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values())

        new_q = current_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * max_next_q - current_q
        )

        self.q_table[state_key][action] = new_q
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def make_decision(self, metrics: RawMetrics, router=None) -> Dict:
        current_state = self.get_state(metrics)

        if self.last_state and self.last_action:
            reward = self.calculate_reward(
                self.last_state,
                self.last_action,
                current_state,
                metrics,
            )
            self.update_q_table(
                self.last_state,
                self.last_action,
                reward,
                current_state,
            )

        action = self.select_action(current_state)
        action_taken = False

        if action != Action.HOLD:
            new_count = execute_scaling_action(action, self.current_servers)
            if new_count != self.current_servers:
                self.current_servers = new_count
                self.last_scale_time = datetime.now()
                action_taken = True

        self.last_state = current_state
        self.last_action = action

        decision = {
            "timestamp": datetime.now().isoformat(),
            "state": {
                "servers": current_state.current_servers,
                "load": current_state.load_level.value,
                "trend": current_state.prediction_trend.value,
                "latency": current_state.latency_status.value,
            },
            "action": action.value,
            "action_taken": action_taken,
            "epsilon": round(self.epsilon, 3),
        }

        self.decision_history.append(decision)
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]

        return decision


scaler = QLearningScaler()