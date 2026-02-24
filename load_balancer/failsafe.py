from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class FallbackPolicy(str, Enum):
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS    = "least_connections"
    EQUAL_SPLIT          = "equal_split"


class FailsafeReason(str, Enum):
    ML_UNAVAILABLE    = "ml_unavailable"
    LATENCY_EXCEEDED  = "latency_exceeded"
    ERROR_RATE_HIGH   = "error_rate_high"
    ALL_BACKENDS_DOWN = "all_backends_down"
    MANUAL_OVERRIDE   = "manual_override"


@dataclass(frozen=True)
class BackendSnapshot:
    """Immutable view of a backend's current state."""
    backend_id:         str
    is_healthy:         bool
    active_connections: int
    capacity:           float       # normalised req/s
    recent_error_rate:  float = 0.0 # [0, 1] over last 60s
    avg_latency_ms:     float = 0.0


@dataclass
class FailsafeDecision:
    """Output of ThresholdFailsafe.decide()."""
    weights:     dict[str, float]  # backend_id → routing weight (sum=1)
    policy_used: FallbackPolicy
    reason:      FailsafeReason
    triggered:   bool              # False = ML routing is fine
    timestamp:   float = field(default_factory=time.time)


class ThresholdFailsafe:
    """Heuristic fallback when ML routing is unavailable or too slow."""

    def __init__(
        self,
        latency_threshold_ms: float          = 200.0,
        error_rate_threshold: float          = 0.3,
        min_healthy_backends: int            = 1,
        default_policy:       FallbackPolicy = FallbackPolicy.LEAST_CONNECTIONS,
    ):
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self.min_healthy_backends = min_healthy_backends
        self.default_policy       = default_policy
        self._override_count: int = 0
        self._last_reason: Optional[FailsafeReason] = None

    def decide(
        self,
        backends:      list[BackendSnapshot],
        ml_weights:    Optional[dict[str, float]] = None,
        ml_latency_ms: float = 0.0,
        ml_available:  bool  = True,
    ) -> FailsafeDecision:
        """Return routing weights; falls back to heuristic if ML is unhealthy."""
        healthy = [b for b in backends if b.is_healthy]

        if len(healthy) < self.min_healthy_backends:
            reason  = FailsafeReason.ALL_BACKENDS_DOWN
            weights = self._equal_split(backends)
            self._log_override(reason, backends, weights)
            return FailsafeDecision(weights, FallbackPolicy.EQUAL_SPLIT, reason, triggered=True)

        if not ml_available or ml_weights is None:
            reason  = FailsafeReason.ML_UNAVAILABLE
            weights = self._apply_policy(self.default_policy, healthy)
            self._log_override(reason, backends, weights)
            return FailsafeDecision(weights, self.default_policy, reason, triggered=True)

        if ml_latency_ms > self.latency_threshold_ms:
            reason  = FailsafeReason.LATENCY_EXCEEDED
            weights = self._apply_policy(FallbackPolicy.LEAST_CONNECTIONS, healthy)
            self._log_override(reason, backends, weights)
            return FailsafeDecision(weights, FallbackPolicy.LEAST_CONNECTIONS, reason, triggered=True)

        critical = [b for b in healthy if b.recent_error_rate > self.error_rate_threshold]
        if critical:
            reason  = FailsafeReason.ERROR_RATE_HIGH
            safe    = [b for b in healthy if b not in critical]
            pool    = safe if safe else healthy
            weights = self._apply_policy(FallbackPolicy.WEIGHTED_ROUND_ROBIN, pool)
            for b in backends:
                if b not in pool:
                    weights[b.backend_id] = 0.0
            weights = _normalise(weights)
            self._log_override(reason, backends, weights)
            return FailsafeDecision(weights, FallbackPolicy.WEIGHTED_ROUND_ROBIN, reason, triggered=True)

        sanitised = _sanitise_ml_weights(ml_weights, healthy)
        return FailsafeDecision(
            weights=sanitised,
            policy_used=FallbackPolicy.WEIGHTED_ROUND_ROBIN,
            reason=FailsafeReason.ML_UNAVAILABLE,
            triggered=False,
        )

    def manual_override(
        self,
        backends: list[BackendSnapshot],
        policy:   FallbackPolicy = FallbackPolicy.EQUAL_SPLIT,
    ) -> FailsafeDecision:
        """Force a heuristic policy (e.g. during maintenance)."""
        healthy = [b for b in backends if b.is_healthy] or backends
        weights = self._apply_policy(policy, healthy)
        return FailsafeDecision(weights, policy, FailsafeReason.MANUAL_OVERRIDE, triggered=True)

    def _apply_policy(self, policy: FallbackPolicy, backends: list[BackendSnapshot]) -> dict[str, float]:
        if policy == FallbackPolicy.LEAST_CONNECTIONS:
            return self._least_connections(backends)
        elif policy == FallbackPolicy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(backends)
        else:
            return self._equal_split(backends)

    @staticmethod
    def _least_connections(backends: list[BackendSnapshot]) -> dict[str, float]:
        if not backends:
            return {}
        inv   = [1.0 / (b.active_connections + 1) for b in backends]
        total = sum(inv)
        return {b.backend_id: v / total for b, v in zip(backends, inv)}

    @staticmethod
    def _weighted_round_robin(backends: list[BackendSnapshot]) -> dict[str, float]:
        if not backends:
            return {}
        caps  = [b.capacity for b in backends]
        total = sum(caps) or 1.0
        return {b.backend_id: c / total for b, c in zip(backends, caps)}

    @staticmethod
    def _equal_split(backends: list[BackendSnapshot]) -> dict[str, float]:
        if not backends:
            return {}
        w = 1.0 / len(backends)
        return {b.backend_id: w for b in backends}

    def _log_override(self, reason: FailsafeReason, backends: list[BackendSnapshot], weights: dict[str, float]):
        self._override_count += 1
        self._last_reason     = reason
        healthy_ids = [b.backend_id for b in backends if b.is_healthy]
        logger.warning("FAILSAFE TRIGGERED | reason=%s | healthy=%s | weights=%s",
                       reason.value, healthy_ids, {k: f"{v:.3f}" for k, v in weights.items()})

    @property
    def override_count(self) -> int:
        return self._override_count

    @property
    def last_reason(self) -> Optional[FailsafeReason]:
        return self._last_reason


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def _sanitise_ml_weights(ml_weights: dict[str, float], healthy: list[BackendSnapshot]) -> dict[str, float]:
    """Zero out ML weights for unhealthy backends and re-normalise."""
    healthy_ids = {b.backend_id for b in healthy}
    sanitised   = {k: (v if k in healthy_ids else 0.0) for k, v in ml_weights.items()}
    return _normalise(sanitised)
