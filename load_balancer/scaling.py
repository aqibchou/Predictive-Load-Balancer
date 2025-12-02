import asyncio
import pickle
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# Q-Learning hyperparameters
LEARNING_RATE = 0.1          # α - how fast we learn from new experiences
DISCOUNT_FACTOR = 0.95       # γ - how much we value future rewards
EPSILON_START = 1.0          # Initial exploration rate (100% random)
EPSILON_MIN = 0.1            # Minimum exploration rate (10% random)
EPSILON_DECAY = 0.995        # How fast exploration decreases

# Scaling constraints
MIN_SERVERS = 1
MAX_SERVERS = 5
SCALE_COOLDOWN_SECONDS = 60  # Prevent thrashing

# Reward weights
LATENCY_CRITICAL_PENALTY = -10
LATENCY_WARNING_PENALTY = -5
SERVER_COST_PENALTY = -0.5
SLA_BONUS = 5
ACTION_PENALTY = -1          # Penalty for not holding (prevents thrashing)


class Action(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"


class LoadLevel(Enum):
    LOW = "low"           # < 30%
    MEDIUM = "medium"     # 30-70%
    HIGH = "high"         # > 70%


class PredictionTrend(Enum):
    DECREASING = "decreasing"  # Predicted < current
    STABLE = "stable"          # Predicted ≈ current (±10%)
    INCREASING = "increasing"  # Predicted > current


class LatencyStatus(Enum):
    OK = "ok"             # < 100ms
    WARNING = "warning"   # 100-200ms
    CRITICAL = "critical" # > 200ms


@dataclass
class SystemState:
    """Discretized system state for Q-table lookup"""
    current_servers: int
    load_level: LoadLevel
    prediction_trend: PredictionTrend
    latency_status: LatencyStatus
    
    def to_tuple(self) -> Tuple:
        """Convert to hashable tuple for Q-table key"""
        return (
            self.current_servers,
            self.load_level.value,
            self.prediction_trend.value,
            self.latency_status.value
        )


@dataclass
class RawMetrics:
    """Raw metrics before discretization"""
    current_servers: int
    avg_load_percent: float
    predicted_load_5min: float
    current_load: float
    p95_latency_ms: float
    prediction_uncertainty: float


class QLearningScaler:
    def __init__(self):
        # Q-table: maps (state, action) -> expected reward
        self.q_table: Dict[Tuple, Dict[Action, float]] = defaultdict(
            lambda: {action: 0.0 for action in Action}
        )
        
        self.epsilon = EPSILON_START
        self.current_servers = 3  # Starting server count
        self.last_scale_time: Optional[datetime] = None
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        
        # Decision history for analysis
        self.decision_history: List[Dict] = []
        self.last_state: Optional[SystemState] = None
        self.last_action: Optional[Action] = None
    
    def discretize_load(self, load_percent: float) -> LoadLevel:
        """Convert continuous load to discrete level"""
        if load_percent < 30:
            return LoadLevel.LOW
        elif load_percent < 70:
            return LoadLevel.MEDIUM
        else:
            return LoadLevel.HIGH
    
    def discretize_trend(self, predicted: float, current: float) -> PredictionTrend:
        """Compare prediction to current traffic"""
        if current == 0:
            return PredictionTrend.STABLE
        
        change_percent = (predicted - current) / current * 100
        
        if change_percent < -10:
            return PredictionTrend.DECREASING
        elif change_percent > 10:
            return PredictionTrend.INCREASING
        else:
            return PredictionTrend.STABLE
    
    def discretize_latency(self, p95_latency_ms: float) -> LatencyStatus:
        """Convert latency to status level"""
        if p95_latency_ms < 100:
            return LatencyStatus.OK
        elif p95_latency_ms < 200:
            return LatencyStatus.WARNING
        else:
            return LatencyStatus.CRITICAL
    
    def get_state(self, metrics: RawMetrics) -> SystemState:
        """Convert raw metrics to discretized state"""
        return SystemState(
            current_servers=metrics.current_servers,
            load_level=self.discretize_load(metrics.avg_load_percent),
            prediction_trend=self.discretize_trend(
                metrics.predicted_load_5min, 
                metrics.current_load
            ),
            latency_status=self.discretize_latency(metrics.p95_latency_ms)
        )
    
    def calculate_reward(
        self, 
        state_before: SystemState, 
        action: Action, 
        state_after: SystemState,
        metrics_after: RawMetrics
    ) -> float:
        """Calculate reward based on outcomes"""
        reward = 0.0
        
        # Latency penalties
        if state_after.latency_status == LatencyStatus.CRITICAL:
            reward += LATENCY_CRITICAL_PENALTY
        elif state_after.latency_status == LatencyStatus.WARNING:
            reward += LATENCY_WARNING_PENALTY
        
        # Server cost penalty (more servers = more cost)
        reward += state_after.current_servers * SERVER_COST_PENALTY
        
        # SLA bonus: low latency with minimal servers
        if (state_after.latency_status == LatencyStatus.OK and 
            state_after.current_servers <= 3):
            reward += SLA_BONUS
        
        # Action penalty to prevent thrashing
        if action != Action.HOLD:
            reward += ACTION_PENALTY
        
        return reward
    
    def select_action(self, state: SystemState) -> Action:
        """Epsilon-greedy action selection"""
        # Check cooldown
        if self.last_scale_time:
            time_since_scale = (datetime.now() - self.last_scale_time).total_seconds()
            if time_since_scale < SCALE_COOLDOWN_SECONDS:
                return Action.HOLD
        
        # Check constraints
        valid_actions = list(Action)
        if self.current_servers >= MAX_SERVERS:
            valid_actions.remove(Action.SCALE_UP)
        if self.current_servers <= MIN_SERVERS:
            valid_actions.remove(Action.SCALE_DOWN)
        
        # Epsilon-greedy: explore vs exploit
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(valid_actions)
        else:
            # Exploit: best known action
            state_key = state.to_tuple()
            q_values = self.q_table[state_key]
            
            # Filter to valid actions and find best
            valid_q = {}
            for a in valid_actions:
                valid_q[a] = q_values[a]

            return max(valid_q, key=valid_q.get)
    
    def update_q_table(
        self,
        state: SystemState,
        action: Action,
        reward: float,
        next_state: SystemState
    ):
        """Q-Learning update rule"""
        state_key = state.to_tuple()
        next_state_key = next_state.to_tuple()
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Best Q-value for next state
        max_next_q = max(self.q_table[next_state_key].values())
        
        # Q-Learning formula
        new_q = current_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
        # Decay exploration rate
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def execute_action(self, action: Action) -> bool:
        """Execute scaling action (returns True if action taken)"""
        if action == Action.HOLD:
            return False
        
        if action == Action.SCALE_UP and self.current_servers < MAX_SERVERS:
            self.current_servers += 1
            self.last_scale_time = datetime.now()
            print(f"SCALE UP: Now running {self.current_servers} servers")
            return True
        
        if action == Action.SCALE_DOWN and self.current_servers > MIN_SERVERS:
            self.current_servers -= 1
            self.last_scale_time = datetime.now()
            print(f"SCALE DOWN: Now running {self.current_servers} servers")
            return True
        
        return False
    
    def make_decision(self, metrics: RawMetrics) -> Dict:
        """Main entry point: observe state, decide action, learn from past"""
        current_state = self.get_state(metrics)
        
        # Learn from previous decision
        if self.last_state and self.last_action:
            reward = self.calculate_reward(
                self.last_state, 
                self.last_action, 
                current_state,
                metrics
            )
            self.update_q_table(
                self.last_state, 
                self.last_action, 
                reward, 
                current_state
            )
        
        # Select and execute action
        action = self.select_action(current_state)
        action_taken = self.execute_action(action)
        
        # Store for next iteration
        self.last_state = current_state
        self.last_action = action
        
        # Log decision
        decision = {
            "timestamp": datetime.now().isoformat(),
            "state": {
                "servers": current_state.current_servers,
                "load": current_state.load_level.value,
                "trend": current_state.prediction_trend.value,
                "latency": current_state.latency_status.value
            },
            "action": action.value,
            "action_taken": action_taken,
            "epsilon": round(self.epsilon, 3),
            "metrics": {
                "avg_load": round(metrics.avg_load_percent, 1),
                "predicted": round(metrics.predicted_load_5min, 1),
                "p95_latency": round(metrics.p95_latency_ms, 1)
            }
        }
        
        self.decision_history.append(decision)
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        
        return decision
    
    def get_q_table_summary(self) -> Dict:
        """Return Q-table for debugging"""
        summary = {}
        for state_key, actions in self.q_table.items():
            state_str = f"servers={state_key[0]},load={state_key[1]},trend={state_key[2]},latency={state_key[3]}"
            summary[state_str] = {a.value: round(v, 2) for a, v in actions.items()}
        return summary
    
    def save_model(self, path: str):
        """Save Q-table to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'current_servers': self.current_servers
            }, f)
        print(f"Q-table saved to {path}")
    
    def load_model(self, path: str) -> bool:
        """Load Q-table from disk"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(
                    lambda: {action: 0.0 for action in Action},
                    data['q_table']
                )
                self.epsilon = data['epsilon']
                self.current_servers = data['current_servers']
            print(f"Q-table loaded from {path}")
            return True
        except FileNotFoundError:
            print(f"No existing Q-table at {path}")
            return False


# Global scaler instance
scaler = QLearningScaler()