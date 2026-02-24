"""
ebpf/controller.py — Userspace controller for the XDP failsafe.

Loads xdp_failsafe.c via BCC, polls the load balancer health endpoint and
Prometheus P95 latency, and flips lb_health_map[0] between 0 (circuit
closed / healthy) and 1 (circuit open / redirect traffic to fallback).

Also exposes a Prometheus gauge ``ebpf_circuit_state`` on :9100/metrics.

Usage (Linux only, requires root or CAP_NET_ADMIN + CAP_BPF):
  python3 controller.py \
      --interface eth0 \
      --lb-host 172.17.0.2 \
      --fallback-ip 172.17.0.3 \
      --fallback-mac AA:BB:CC:DD:EE:FF

See ebpf/README.md for Lima VM setup on macOS.
"""

import argparse
import ctypes
import os
import signal
import socket
import struct
import sys
import time
import threading

import requests
from prometheus_client import Gauge, start_http_server

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="XDP Failsafe Controller")
    p.add_argument("--interface",   default="eth0",          help="NIC to attach XDP program to")
    p.add_argument("--lb-host",     default="172.17.0.2",    help="Load balancer IP or hostname")
    p.add_argument("--lb-port",     default=8000, type=int,  help="Load balancer HTTP port")
    p.add_argument("--fallback-ip", default="172.17.0.3",    help="Fallback server IP")
    p.add_argument("--fallback-mac",default="02:00:00:00:00:01", help="Fallback server MAC (AA:BB:CC:DD:EE:FF)")
    p.add_argument("--prometheus-port", default=9100, type=int, help="Port to expose /metrics")
    p.add_argument("--health-interval",   default=1.0, type=float, help="Health poll interval (s)")
    p.add_argument("--prom-interval",     default=5.0, type=float, help="Prometheus poll interval (s)")
    p.add_argument("--failure-threshold", default=3,   type=int,   help="Consecutive failures before opening circuit")
    p.add_argument("--p95-threshold-ms",  default=500, type=float, help="P95 latency threshold to open circuit (ms)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ip_to_be32(ip_str: str) -> int:
    """Convert dotted-decimal IP to big-endian 32-bit integer."""
    packed = socket.inet_aton(ip_str)
    return struct.unpack("!I", packed)[0]   # network byte order


def mac_to_bytes(mac_str: str) -> list:
    """Convert 'AA:BB:CC:DD:EE:FF' → list of 6 ints."""
    return [int(b, 16) for b in mac_str.split(":")]


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Pure-Python circuit breaker used by the controller main loop.

    Decoupled from BCC so it can be unit-tested without kernel access.
    """

    def __init__(self, failure_threshold: int = 3, p95_threshold_ms: float = 500.0):
        self.failure_threshold   = failure_threshold
        self.p95_threshold_ms    = p95_threshold_ms
        self.consecutive_health_failures = 0
        self.consecutive_prom_failures   = 0
        self.circuit_open = False

    # ── Observation methods ────────────────────────────────────────────────────

    def record_health_success(self):
        self.consecutive_health_failures = 0

    def record_health_failure(self):
        self.consecutive_health_failures += 1

    def record_p95(self, p95_ms: float):
        """Update Prometheus P95 counter; returns True if threshold exceeded."""
        if p95_ms > self.p95_threshold_ms:
            self.consecutive_prom_failures += 1
            return True
        self.consecutive_prom_failures = 0
        return False

    # ── State predicates ──────────────────────────────────────────────────────

    @property
    def should_open(self) -> bool:
        return (
            self.consecutive_health_failures >= self.failure_threshold or
            self.consecutive_prom_failures   >= self.failure_threshold
        )

    @property
    def should_close(self) -> bool:
        return (
            self.consecutive_health_failures == 0 and
            self.consecutive_prom_failures   == 0
        )

    def transition(self) -> str:
        """Compute next state; returns 'opened', 'closed', or 'unchanged'."""
        if self.should_open and not self.circuit_open:
            self.circuit_open = True
            return "opened"
        if self.should_close and self.circuit_open:
            self.circuit_open = False
            return "closed"
        return "unchanged"


def main():
    args = parse_args()

    # Start Prometheus metrics server
    start_http_server(args.prometheus_port)
    circuit_gauge = Gauge(
        "ebpf_circuit_state",
        "XDP failsafe circuit state: 1=open (redirecting), 0=closed (healthy)",
        ["state"],
    )
    circuit_gauge.labels(state="open").set(0)
    circuit_gauge.labels(state="closed").set(1)

    # Load BCC / BPF
    try:
        from bcc import BPF
    except ImportError:
        print("ERROR: python3-bpfcc not found. Install on Ubuntu: apt install bpfcc-tools python3-bpfcc", file=sys.stderr)
        sys.exit(1)

    prog_path = os.path.join(os.path.dirname(__file__), "xdp_failsafe.c")
    print(f"[controller] Loading XDP program from {prog_path} ...")
    b = BPF(src_file=prog_path)
    fn = b.load_func("xdp_failsafe", BPF.XDP)
    b.attach_xdp(args.interface, fn, flags=BPF.XDP_FLAGS_SKB_MODE)
    print(f"[controller] XDP attached to {args.interface}")

    # Pre-populate BPF maps
    lb_health_map    = b["lb_health_map"]
    fallback_ip_map  = b["fallback_ip_map"]
    fallback_mac_map = b["fallback_mac_map"]

    # Circuit starts closed
    lb_health_map[ctypes.c_int(0)] = ctypes.c_uint(0)

    # Set fallback IP (big-endian)
    fallback_ip_map[ctypes.c_int(0)] = ctypes.c_uint(ip_to_be32(args.fallback_ip))

    # Set fallback MAC (byte by byte)
    for i, byte in enumerate(mac_to_bytes(args.fallback_mac)):
        fallback_mac_map[ctypes.c_int(i)] = ctypes.c_ubyte(byte)

    # Graceful shutdown
    _running = [True]
    def _shutdown(sig, frame):
        _running[0] = False
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    lb_url   = f"http://{args.lb_host}:{args.lb_port}"
    prom_url = f"http://{args.lb_host}:9090"

    cb = CircuitBreaker(
        failure_threshold=args.failure_threshold,
        p95_threshold_ms=args.p95_threshold_ms,
    )
    last_prom_check   = 0.0
    last_health_check = 0.0

    print(f"[controller] Monitoring {lb_url} — circuit opens after "
          f"{args.failure_threshold} consecutive failures")

    try:
        while _running[0]:
            now = time.monotonic()

            # ── Health poll ───────────────────────────────────────────────
            if now - last_health_check >= args.health_interval:
                last_health_check = now
                try:
                    r = requests.get(f"{lb_url}/health", timeout=1.0)
                    r.raise_for_status()
                    cb.record_health_success()
                except Exception:
                    cb.record_health_failure()
                    print(f"[controller] Health check failed "
                          f"({cb.consecutive_health_failures}/{cb.failure_threshold})")

            # ── Prometheus P95 poll ───────────────────────────────────────
            if now - last_prom_check >= args.prom_interval:
                last_prom_check = now
                try:
                    query = (
                        'histogram_quantile(0.95,'
                        'rate(http_request_duration_seconds_bucket[1m]))'
                    )
                    r = requests.get(
                        f"{prom_url}/api/v1/query",
                        params={"query": query},
                        timeout=2.0,
                    )
                    result = r.json().get("data", {}).get("result", [])
                    if result:
                        p95_ms = float(result[0]["value"][1]) * 1000
                        if cb.record_p95(p95_ms):
                            print(f"[controller] P95={p95_ms:.0f}ms > {cb.p95_threshold_ms}ms "
                                  f"({cb.consecutive_prom_failures}/{cb.failure_threshold})")
                except Exception:
                    pass   # Prometheus unavailable — ignore

            # ── Circuit state machine ─────────────────────────────────────
            transition = cb.transition()
            if transition == "opened":
                lb_health_map[ctypes.c_int(0)] = ctypes.c_uint(1)
                circuit_gauge.labels(state="open").set(1)
                circuit_gauge.labels(state="closed").set(0)
                print(f"[controller] *** CIRCUIT OPEN *** — redirecting to {args.fallback_ip}")
            elif transition == "closed":
                lb_health_map[ctypes.c_int(0)] = ctypes.c_uint(0)
                circuit_gauge.labels(state="open").set(0)
                circuit_gauge.labels(state="closed").set(1)
                print("[controller] Circuit CLOSED — normal forwarding resumed")

            time.sleep(0.2)

    finally:
        b.remove_xdp(args.interface)
        print(f"[controller] XDP removed from {args.interface}")


if __name__ == "__main__":
    main()
