"""
Unit tests for ebpf/controller.py

Tests the pure-Python helpers and the CircuitBreaker state machine.
No BCC / Linux kernel required — all kernel-specific code is in main()
which is excluded from these tests.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ebpf"))

# Import only the testable symbols — avoid importing prometheus_client at
# module level so tests don't require it installed.
from controller import ip_to_be32, mac_to_bytes, CircuitBreaker, parse_args


# ── ip_to_be32 ────────────────────────────────────────────────────────────────

def test_ip_to_be32_loopback():
    # 127.0.0.1 → 0x7F000001
    assert ip_to_be32("127.0.0.1") == 0x7F000001


def test_ip_to_be32_private():
    # 172.17.0.2 → 0xAC110002
    assert ip_to_be32("172.17.0.2") == 0xAC110002


def test_ip_to_be32_broadcast():
    # 255.255.255.255 → 0xFFFFFFFF
    assert ip_to_be32("255.255.255.255") == 0xFFFFFFFF


def test_ip_to_be32_zero():
    assert ip_to_be32("0.0.0.0") == 0


def test_ip_to_be32_returns_int():
    result = ip_to_be32("192.168.1.1")
    assert isinstance(result, int)


def test_ip_to_be32_invalid_raises():
    with pytest.raises(Exception):
        ip_to_be32("not_an_ip")


# ── mac_to_bytes ──────────────────────────────────────────────────────────────

def test_mac_to_bytes_length():
    assert len(mac_to_bytes("AA:BB:CC:DD:EE:FF")) == 6


def test_mac_to_bytes_values():
    result = mac_to_bytes("AA:BB:CC:DD:EE:FF")
    assert result == [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]


def test_mac_to_bytes_zeros():
    result = mac_to_bytes("00:00:00:00:00:00")
    assert result == [0, 0, 0, 0, 0, 0]


def test_mac_to_bytes_broadcast():
    result = mac_to_bytes("FF:FF:FF:FF:FF:FF")
    assert all(b == 0xFF for b in result)


def test_mac_to_bytes_lowercase():
    result = mac_to_bytes("aa:bb:cc:dd:ee:ff")
    assert result == [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]


# ── CircuitBreaker — initial state ────────────────────────────────────────────

def test_circuit_starts_closed():
    cb = CircuitBreaker()
    assert cb.circuit_open is False


def test_circuit_starts_no_failures():
    cb = CircuitBreaker()
    assert cb.consecutive_health_failures == 0
    assert cb.consecutive_prom_failures   == 0


def test_circuit_default_thresholds():
    cb = CircuitBreaker()
    assert cb.failure_threshold  == 3
    assert cb.p95_threshold_ms   == 500.0


def test_circuit_custom_thresholds():
    cb = CircuitBreaker(failure_threshold=5, p95_threshold_ms=200.0)
    assert cb.failure_threshold == 5
    assert cb.p95_threshold_ms  == 200.0


# ── CircuitBreaker — health polling ──────────────────────────────────────────

def test_health_failure_increments_counter():
    cb = CircuitBreaker()
    cb.record_health_failure()
    cb.record_health_failure()
    assert cb.consecutive_health_failures == 2


def test_health_success_resets_counter():
    cb = CircuitBreaker()
    cb.record_health_failure()
    cb.record_health_failure()
    cb.record_health_success()
    assert cb.consecutive_health_failures == 0


def test_health_success_does_not_affect_prom_failures():
    cb = CircuitBreaker()
    cb.record_p95(9999)
    cb.record_health_success()
    assert cb.consecutive_prom_failures == 1


# ── CircuitBreaker — Prometheus P95 polling ───────────────────────────────────

def test_p95_below_threshold_returns_false():
    cb = CircuitBreaker(p95_threshold_ms=500)
    assert cb.record_p95(499) is False


def test_p95_above_threshold_returns_true():
    cb = CircuitBreaker(p95_threshold_ms=500)
    assert cb.record_p95(501) is True


def test_p95_above_threshold_increments_counter():
    cb = CircuitBreaker(p95_threshold_ms=500)
    cb.record_p95(600)
    cb.record_p95(700)
    assert cb.consecutive_prom_failures == 2


def test_p95_drop_below_threshold_resets_counter():
    cb = CircuitBreaker(p95_threshold_ms=500)
    cb.record_p95(600)
    cb.record_p95(600)
    cb.record_p95(400)   # drops below → reset
    assert cb.consecutive_prom_failures == 0


# ── CircuitBreaker — should_open / should_close predicates ───────────────────

def test_should_open_false_when_healthy():
    cb = CircuitBreaker(failure_threshold=3)
    assert cb.should_open is False


def test_should_open_true_after_health_failures():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_health_failure()
    cb.record_health_failure()
    assert cb.should_open is False   # only 2 — not yet
    cb.record_health_failure()
    assert cb.should_open is True    # reached threshold


def test_should_open_true_after_prom_failures():
    cb = CircuitBreaker(failure_threshold=3, p95_threshold_ms=500)
    cb.record_p95(999)
    cb.record_p95(999)
    cb.record_p95(999)
    assert cb.should_open is True


def test_should_close_true_when_all_zeros():
    cb = CircuitBreaker()
    assert cb.should_close is True


def test_should_close_false_when_health_failures_present():
    cb = CircuitBreaker()
    cb.record_health_failure()
    assert cb.should_close is False


def test_should_close_false_when_prom_failures_present():
    cb = CircuitBreaker(p95_threshold_ms=500)
    cb.record_p95(999)
    assert cb.should_close is False


# ── CircuitBreaker — transition() ────────────────────────────────────────────

def test_transition_unchanged_when_healthy_and_closed():
    cb = CircuitBreaker()
    assert cb.transition() == "unchanged"


def test_transition_opened_after_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_health_failure()
    cb.record_health_failure()
    cb.record_health_failure()
    result = cb.transition()
    assert result == "opened"
    assert cb.circuit_open is True


def test_transition_unchanged_when_already_open():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_health_failure()
    cb.record_health_failure()
    cb.record_health_failure()
    cb.transition()   # opens
    result = cb.transition()   # still failing — stays open
    assert result == "unchanged"
    assert cb.circuit_open is True


def test_transition_closed_after_recovery():
    cb = CircuitBreaker(failure_threshold=3)
    # open the circuit
    for _ in range(3):
        cb.record_health_failure()
    cb.transition()
    # simulate recovery
    cb.record_health_success()
    result = cb.transition()
    assert result == "closed"
    assert cb.circuit_open is False


def test_transition_full_lifecycle():
    """opened → unchanged → closed → unchanged"""
    cb = CircuitBreaker(failure_threshold=2)

    # healthy → unchanged
    assert cb.transition() == "unchanged"

    # 2 failures → opens
    cb.record_health_failure()
    cb.record_health_failure()
    assert cb.transition() == "opened"

    # still failing → unchanged
    assert cb.transition() == "unchanged"

    # recover
    cb.record_health_success()
    assert cb.transition() == "closed"

    # healthy again → unchanged
    assert cb.transition() == "unchanged"


def test_transition_prom_opens_circuit():
    cb = CircuitBreaker(failure_threshold=2, p95_threshold_ms=200)
    cb.record_p95(999)
    cb.record_p95(999)
    assert cb.transition() == "opened"


def test_transition_mixed_health_and_prom():
    """One health failure + two prom failures → opens (prom hits threshold first)."""
    cb = CircuitBreaker(failure_threshold=2, p95_threshold_ms=200)
    cb.record_health_failure()
    cb.record_p95(999)
    cb.record_p95(999)  # prom hits 2
    assert cb.should_open is True


# ── parse_args defaults ───────────────────────────────────────────────────────

def test_parse_args_defaults():
    # Verify parse_args() works with no input — delegates to helper below
    _parse_args_with_no_input()


def _parse_args_with_no_input():
    import argparse
    # Re-parse with empty argv to get defaults
    sys_argv_backup = sys.argv
    sys.argv = ["controller.py"]
    try:
        return parse_args()
    finally:
        sys.argv = sys_argv_backup


def test_default_interface():
    args = _parse_args_with_no_input()
    assert args.interface == "eth0"


def test_default_lb_host():
    args = _parse_args_with_no_input()
    assert args.lb_host == "172.17.0.2"


def test_default_failure_threshold():
    args = _parse_args_with_no_input()
    assert args.failure_threshold == 3


def test_default_p95_threshold():
    args = _parse_args_with_no_input()
    assert args.p95_threshold_ms == 500.0


def test_default_health_interval():
    args = _parse_args_with_no_input()
    assert args.health_interval == 1.0


def test_default_prom_interval():
    args = _parse_args_with_no_input()
    assert args.prom_interval == 5.0
