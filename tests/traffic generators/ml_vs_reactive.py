import requests
import time
import statistics
import threading
from datetime import datetime

LB_URL = "http://localhost:8000"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def send_traffic(rate_per_sec, duration_sec, results, label):
    """Send traffic at a given rate for a given duration, collect latencies."""
    interval = 1.0 / rate_per_sec
    end_time = time.time() + duration_sec
    count = 0
    while time.time() < end_time:
        start = time.time()
        try:
            # Use varied paths so Prophet sees realistic traffic patterns
            path = f"api/item/{count % 20}"
            r = requests.get(f"{LB_URL}/{path}", timeout=10)
            latency = (time.time() - start) * 1000
            results.append(latency)
        except Exception as e:
            log(f"  Request error: {e}")
        elapsed = time.time() - start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        count += 1

def run_full_experiment(label, lock_servers=None):
    """
    Run one full experiment.
    lock_servers: if set, locks server count (reactive mode)
    if None, Q-learning controls scaling (predictive mode)
    """
    log(f"Starting experiment: {label}")

    # Set scaling mode
    if lock_servers is not None:
        log(f"Locking servers to {lock_servers} (reactive mode)")
        requests.post(f"{LB_URL}/scaling/override", json={"fixed_servers": lock_servers})
    else:
        log("Q-learning enabled (predictive mode)")
        requests.post(f"{LB_URL}/scaling/override", json={"fixed_servers": None})

    time.sleep(3)

    warmup_latencies = []
    varied_latencies = []
    spike_latencies = []

    # Phase 1: Warmup — 20 minutes at low traffic (2 req/sec)
    # Gives Prophet enough history to start making predictions
    log("Phase 1: Warmup (20 min, 2 req/sec)...")
    send_traffic(rate_per_sec=2, duration_sec=1200, results=warmup_latencies, label="warmup")
    log(f"  Warmup done. Avg: {statistics.mean(warmup_latencies):.1f}ms")

    # Phase 2: Varied traffic — 30 minutes ramping up and down
    # Gives Prophet real patterns to learn: low → medium → high → medium → low
    log("Phase 2: Varied traffic (30 min)...")
    phases = [
        (3, 360),   # 3 req/sec for 6 min
        (6, 360),   # 6 req/sec for 6 min
        (10, 360),  # 10 req/sec for 6 min (medium spike)
        (6, 360),   # back down to 6 req/sec
        (3, 360),   # back to baseline
    ]
    for rate, duration in phases:
        log(f"  Sending {rate} req/sec for {duration//60} min...")
        send_traffic(rate_per_sec=rate, duration_sec=duration, results=varied_latencies, label="varied")
    log(f"  Varied phase done. Avg: {statistics.mean(varied_latencies):.1f}ms")

    # Phase 3: Spike — 20 minutes at high traffic (20 req/sec)
    # This is where reactive vs predictive difference shows up
    log("Phase 3: TRAFFIC SPIKE (20 min, 20 req/sec)")
    log("  Watch Q-learning decisions in Docker logs if running predictive mode")
    send_traffic(rate_per_sec=20, duration_sec=1200, results=spike_latencies, label="spike")
    log(f"  Spike done. Avg: {statistics.mean(spike_latencies):.1f}ms")

    # Compute results
    sorted_spike = sorted(spike_latencies)
    result = {
        "label": label,
        "warmup_avg": round(statistics.mean(warmup_latencies), 1),
        "varied_avg": round(statistics.mean(varied_latencies), 1),
        "spike_avg": round(statistics.mean(spike_latencies), 1),
        "spike_p50": round(sorted_spike[int(len(sorted_spike) * 0.50)], 1),
        "spike_p95": round(sorted_spike[int(len(sorted_spike) * 0.95)], 1),
        "spike_p99": round(sorted_spike[int(len(sorted_spike) * 0.99)], 1),
        "spike_max": round(max(spike_latencies), 1),
        "total_requests": len(warmup_latencies) + len(varied_latencies) + len(spike_latencies),
    }

    log(f"Experiment complete: {label}")
    return result

# ─── RUN 1: REACTIVE 
print("\n" + "="*60)
print("RUN 1: REACTIVE (fixed servers, no ML scaling)")
print("="*60)
result_reactive = run_full_experiment("Reactive", lock_servers=2)

# ─── BREAK
log("10 minute cooldown between runs...")
time.sleep(600)

# ─── RUN 2: PREDICTIVE 
print("\n" + "="*60)
print("RUN 2: PREDICTIVE (Q-learning + Prophet enabled)")
print("="*60)
result_predictive = run_full_experiment("Predictive", lock_servers=None)

# ─── FINAL RESULTS
print("\n" + "="*60)
print("FINAL COMPARISON — SPIKE PHASE ONLY")
print("="*60)
print(f"{'Metric':<20} {'Reactive':>12} {'Predictive':>12} {'Improvement':>12}")
print("-"*58)

metrics = [
    ("spike_avg", "Spike Avg (ms)"),
    ("spike_p50", "Spike P50 (ms)"),
    ("spike_p95", "Spike P95 (ms)"),
    ("spike_p99", "Spike P99 (ms)"),
    ("spike_max", "Spike Max (ms)"),
]

for key, label in metrics:
    r = result_reactive[key]
    p = result_predictive[key]
    imp = round((r - p) / r * 100, 1) if r > 0 else 0
    direction = "better" if imp > 0 else "worse"
    print(f"{label:<20} {r:>12} {p:>12} {abs(imp):>10}% {direction}")

print(f"\nTotal requests processed: {result_reactive['total_requests'] + result_predictive['total_requests']}")
print("="*60)