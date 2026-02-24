import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../load_balancer"))

from failsafe import (
    BackendSnapshot, FailsafeReason, FallbackPolicy, ThresholdFailsafe,
)


def backend(bid, healthy=True, conns=0, cap=1.0, error_rate=0.0):
    return BackendSnapshot(bid, is_healthy=healthy, active_connections=conns,
                           capacity=cap, recent_error_rate=error_rate)


def healthy_pool(n=3):
    return [backend(f"b{i}", conns=i) for i in range(1, n + 1)]


# ── Pass-through (ML healthy) ─────────────────────────────────────────────────

def test_passthrough_when_ml_healthy():
    fs = ThresholdFailsafe()
    bs = healthy_pool(3)
    ml = {"b1": 0.5, "b2": 0.3, "b3": 0.2}
    d  = fs.decide(bs, ml_weights=ml, ml_latency_ms=50)
    assert not d.triggered
    assert abs(sum(d.weights.values()) - 1.0) < 1e-6


def test_passthrough_zeroes_unhealthy_backends():
    fs = ThresholdFailsafe()
    bs = [backend("b1"), backend("b2", healthy=False)]
    ml = {"b1": 0.5, "b2": 0.5}
    d  = fs.decide(bs, ml_weights=ml, ml_latency_ms=50)
    assert not d.triggered
    assert d.weights["b2"] == pytest.approx(0.0, abs=1e-9)
    assert d.weights["b1"] == pytest.approx(1.0, abs=1e-6)


# ── ALL_BACKENDS_DOWN ─────────────────────────────────────────────────────────

def test_all_backends_down_triggers_equal_split():
    fs = ThresholdFailsafe(min_healthy_backends=1)
    bs = [backend("b1", healthy=False), backend("b2", healthy=False)]
    d  = fs.decide(bs, ml_weights={"b1": 1.0})
    assert d.triggered
    assert d.reason      == FailsafeReason.ALL_BACKENDS_DOWN
    assert d.policy_used == FallbackPolicy.EQUAL_SPLIT
    assert abs(sum(d.weights.values()) - 1.0) < 1e-6


# ── ML_UNAVAILABLE ────────────────────────────────────────────────────────────

def test_ml_none_triggers_fallback():
    fs = ThresholdFailsafe()
    d  = fs.decide(healthy_pool(2), ml_weights=None)
    assert d.triggered
    assert d.reason == FailsafeReason.ML_UNAVAILABLE


def test_ml_unavailable_flag_triggers_fallback():
    fs = ThresholdFailsafe()
    d  = fs.decide(healthy_pool(2), ml_weights={"b1": 1.0}, ml_available=False)
    assert d.triggered
    assert d.reason == FailsafeReason.ML_UNAVAILABLE


# ── LATENCY_EXCEEDED ─────────────────────────────────────────────────────────

def test_high_latency_triggers_least_connections():
    fs = ThresholdFailsafe(latency_threshold_ms=200)
    d  = fs.decide(healthy_pool(3), ml_weights={"b1": 1.0}, ml_latency_ms=201)
    assert d.triggered
    assert d.reason      == FailsafeReason.LATENCY_EXCEEDED
    assert d.policy_used == FallbackPolicy.LEAST_CONNECTIONS


def test_latency_below_threshold_does_not_trigger():
    fs = ThresholdFailsafe(latency_threshold_ms=200)
    d  = fs.decide(healthy_pool(3), ml_weights={"b1": 0.4, "b2": 0.3, "b3": 0.3},
                   ml_latency_ms=199)
    assert not d.triggered


# ── ERROR_RATE_HIGH ───────────────────────────────────────────────────────────

def test_high_error_rate_excludes_bad_backend():
    fs = ThresholdFailsafe(error_rate_threshold=0.3)
    bs = [backend("b1", error_rate=0.9), backend("b2", error_rate=0.05)]
    d  = fs.decide(bs, ml_weights={"b1": 0.5, "b2": 0.5})
    assert d.triggered
    assert d.reason         == FailsafeReason.ERROR_RATE_HIGH
    assert d.weights["b1"] == pytest.approx(0.0, abs=1e-9)
    assert d.weights["b2"] == pytest.approx(1.0, abs=1e-6)


# ── Policies ──────────────────────────────────────────────────────────────────

def test_least_connections_weights_inversely():
    fs = ThresholdFailsafe()
    bs = [backend("busy", conns=10), backend("idle", conns=0)]
    w  = fs._least_connections(bs)
    assert w["idle"] > w["busy"]


def test_weighted_round_robin_proportional_to_capacity():
    fs = ThresholdFailsafe()
    bs = [backend("big", cap=3.0), backend("small", cap=1.0)]
    w  = fs._weighted_round_robin(bs)
    assert w["big"] == pytest.approx(0.75, abs=1e-6)
    assert w["small"] == pytest.approx(0.25, abs=1e-6)


def test_equal_split_exact():
    fs = ThresholdFailsafe()
    bs = healthy_pool(4)
    w  = fs._equal_split(bs)
    for v in w.values():
        assert v == pytest.approx(0.25, abs=1e-6)


def test_all_policies_sum_to_one():
    fs = ThresholdFailsafe()
    bs = healthy_pool(5)
    for method in [fs._least_connections, fs._weighted_round_robin, fs._equal_split]:
        assert abs(sum(method(bs).values()) - 1.0) < 1e-6


def test_empty_backends_return_empty():
    fs = ThresholdFailsafe()
    for method in [fs._least_connections, fs._weighted_round_robin, fs._equal_split]:
        assert method([]) == {}


# ── Manual override ───────────────────────────────────────────────────────────

def test_manual_override():
    fs = ThresholdFailsafe()
    d  = fs.manual_override(healthy_pool(3), policy=FallbackPolicy.EQUAL_SPLIT)
    assert d.triggered
    assert d.reason      == FailsafeReason.MANUAL_OVERRIDE
    assert d.policy_used == FallbackPolicy.EQUAL_SPLIT


# ── Override counter ──────────────────────────────────────────────────────────

def test_override_count_increments():
    fs = ThresholdFailsafe()
    fs.decide(healthy_pool(2), ml_available=False)
    fs.decide(healthy_pool(2), ml_available=False)
    assert fs.override_count == 2


def test_last_reason_updated():
    fs = ThresholdFailsafe()
    assert fs.last_reason is None
    fs.decide(healthy_pool(2), ml_available=False)
    assert fs.last_reason == FailsafeReason.ML_UNAVAILABLE
