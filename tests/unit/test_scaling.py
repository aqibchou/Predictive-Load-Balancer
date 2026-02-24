import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../load_balancer"))

from scaling import (
    Action, LatencyStatus, LoadLevel, PredictionTrend, QLearningScaler,
    RawMetrics, SystemState, UncertaintyLevel,
)


def make_scaler():
    return QLearningScaler()


def make_state(servers=3, load=LoadLevel.MEDIUM, trend=PredictionTrend.STABLE,
               latency=LatencyStatus.OK, uncertainty=UncertaintyLevel.LOW):
    return SystemState(servers, load, trend, latency, uncertainty)


def make_metrics(servers=3, load=50.0, pred=55.0, current=50.0,
                 latency=80.0, uncertainty=10.0):
    return RawMetrics(
        current_servers=servers, avg_load_percent=load,
        predicted_load_5min=pred, current_load=current,
        p95_latency_ms=latency, prediction_uncertainty=uncertainty,
    )


# ── Discretisation ────────────────────────────────────────────────────────────

def test_discretize_load_boundaries():
    s = make_scaler()
    assert s.discretize_load(0)    == LoadLevel.LOW
    assert s.discretize_load(29.9) == LoadLevel.LOW
    assert s.discretize_load(30)   == LoadLevel.MEDIUM
    assert s.discretize_load(69.9) == LoadLevel.MEDIUM
    assert s.discretize_load(70)   == LoadLevel.HIGH
    assert s.discretize_load(100)  == LoadLevel.HIGH


def test_discretize_trend():
    s = make_scaler()
    assert s.discretize_trend(120, 100) == PredictionTrend.INCREASING   # +20%
    assert s.discretize_trend(80,  100) == PredictionTrend.DECREASING   # -20%
    assert s.discretize_trend(105, 100) == PredictionTrend.STABLE       # +5%
    assert s.discretize_trend(0,   0)   == PredictionTrend.STABLE       # zero current


def test_discretize_latency_boundaries():
    s = make_scaler()
    assert s.discretize_latency(0)     == LatencyStatus.OK
    assert s.discretize_latency(99.9)  == LatencyStatus.OK
    assert s.discretize_latency(100)   == LatencyStatus.WARNING
    assert s.discretize_latency(199.9) == LatencyStatus.WARNING
    assert s.discretize_latency(200)   == LatencyStatus.CRITICAL


def test_discretize_uncertainty_boundaries():
    s = make_scaler()
    assert s.discretize_uncertainty(0)    == UncertaintyLevel.LOW
    assert s.discretize_uncertainty(19.9) == UncertaintyLevel.LOW
    assert s.discretize_uncertainty(20)   == UncertaintyLevel.MEDIUM
    assert s.discretize_uncertainty(50)   == UncertaintyLevel.MEDIUM
    assert s.discretize_uncertainty(50.1) == UncertaintyLevel.HIGH


def test_get_state_maps_correctly():
    s = make_scaler()
    m = make_metrics(servers=2, load=50, pred=60, current=50, latency=80, uncertainty=15)
    st = s.get_state(m)
    assert st.current_servers   == 2
    assert st.load_level        == LoadLevel.MEDIUM
    assert st.latency_status    == LatencyStatus.OK
    assert st.uncertainty_level == UncertaintyLevel.LOW


# ── State tuple ───────────────────────────────────────────────────────────────

def test_state_to_tuple_has_five_dimensions():
    st = make_state()
    t  = st.to_tuple()
    assert len(t) == 5
    assert t[0] == 3  # servers
    assert isinstance(t[1], str)   # load
    assert isinstance(t[4], str)   # uncertainty


# ── Q-table update ────────────────────────────────────────────────────────────

def test_q_table_update_bellman():
    s     = make_scaler()
    state = make_state(load=LoadLevel.HIGH)
    next_ = make_state(load=LoadLevel.MEDIUM)
    # Q(s,a) = 0 + 0.1 * (5.0 + 0.95 * 0 - 0) = 0.5
    s.update_q_table(state, Action.SCALE_UP, reward=5.0, next_state=next_)
    assert s.q_table[state.to_tuple()][Action.SCALE_UP] == pytest.approx(0.5, abs=1e-6)


def test_epsilon_decays_on_update():
    s     = make_scaler()
    start = s.epsilon
    s.update_q_table(make_state(), Action.HOLD, 0.0, make_state())
    assert s.epsilon < start


def test_reset_clears_table_and_resets_epsilon():
    s = make_scaler()
    s.update_q_table(make_state(), Action.HOLD, 1.0, make_state())
    s.reset_qtable()
    assert len(s.q_table) == 0
    assert s.epsilon == 1.0


# ── Reward ────────────────────────────────────────────────────────────────────

def test_reward_sla_bonus():
    """OK latency + ≤3 servers + HOLD → reward = +5 (SLA) - 1.5 (3×cost) = 3.5"""
    s      = make_scaler()
    before = make_state()
    after  = make_state(load=LoadLevel.LOW)
    m      = make_metrics(servers=3, latency=50)
    assert s.calculate_reward(before, Action.HOLD, after, m) == pytest.approx(3.5, abs=1e-6)


def test_reward_critical_latency_penalty():
    s      = make_scaler()
    before = make_state()
    after  = make_state(latency=LatencyStatus.CRITICAL)
    m      = make_metrics(latency=300)
    reward = s.calculate_reward(before, Action.HOLD, after, m)
    assert reward < 0


def test_reward_action_penalty():
    s      = make_scaler()
    before = make_state()
    after  = make_state()
    m      = make_metrics(latency=50)
    hold_r  = s.calculate_reward(before, Action.HOLD,     after, m)
    scale_r = s.calculate_reward(before, Action.SCALE_UP, after, m)
    assert scale_r < hold_r   # scaling costs -1 extra


def test_reward_prediction_accuracy_bonus():
    s = make_scaler()
    s._prev_predicted_load = 50.0
    before = make_state()
    after  = make_state(load=LoadLevel.LOW)
    # current_load=52 → error_pct = |52-50|/50 = 4% < 25% → +2 bonus
    m = make_metrics(latency=50, current=52.0)
    reward_with_bonus = s.calculate_reward(before, Action.HOLD, after, m)
    s._prev_predicted_load = None
    reward_without       = s.calculate_reward(before, Action.HOLD, after, m)
    assert reward_with_bonus == pytest.approx(reward_without + 2.0, abs=1e-6)


# ── Q-table summary ───────────────────────────────────────────────────────────

def test_get_q_table_summary_keys():
    s       = make_scaler()
    summary = s.get_q_table_summary()
    assert "total_states"    in summary
    assert "non_zero_states" in summary
    assert "epsilon"         in summary
    assert "current_servers" in summary
    assert "states"          in summary


def test_get_q_table_summary_counts_non_zero():
    s = make_scaler()
    s.update_q_table(make_state(), Action.SCALE_UP, 5.0, make_state())
    summary = s.get_q_table_summary()
    assert summary["total_states"]    == 1
    assert summary["non_zero_states"] == 1
