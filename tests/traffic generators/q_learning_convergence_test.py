import requests
import time
import statistics
import threading
from datetime import datetime

LB_URL = "http://localhost:8000"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def send_traffic(rate_per_sec, duration_sec, results):
    interval = 1.0 / rate_per_sec
    end_time = time.time() + duration_sec
    count = 0
    while time.time() < end_time:
        start = time.time()
        try:
            path = f"api/item/{count % 20}"
            r = requests.get(f"{LB_URL}/{path}", timeout=10)
            latency = (time.time() - start) * 1000
            results.append(latency)
        except Exception as e:
            pass
        elapsed = time.time() - start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        count += 1

def get_epsilon():
    try:
        r = requests.get(f"{LB_URL}/scaling/status", timeout=5)
        return r.json().get("epsilon", "?")
    except:
        return "?"

def get_server_count():
    try:
        r = requests.get(f"{LB_URL}/scaling/status", timeout=5)
        return r.json().get("current_servers", "?")
    except:
        return "?"

def run_cycle(cycle_num, total_cycles):
    """
    One full traffic cycle:
    - 30 min low (2 req/sec)
    - 15 min ramp up (2 -> 20 req/sec in steps)
    - 20 min spike (20 req/sec)
    - 15 min ramp down (20 -> 2 req/sec in steps)
    - 20 min recovery (2 req/sec)
    Total: 100 minutes per cycle
    """
    log(f"--- CYCLE {cycle_num}/{total_cycles} START | epsilon={get_epsilon()} | servers={get_server_count()} ---")

    cycle_latencies = []
    spike_latencies = []

    # Phase 1: Low traffic - 30 min
    log(f"  [Cycle {cycle_num}] Phase 1: Low traffic (30 min, 2 req/sec)...")
    send_traffic(2, 1800, cycle_latencies)
    log(f"  [Cycle {cycle_num}] Low done. epsilon={get_epsilon()} servers={get_server_count()}")

    # Phase 2: Ramp up - 15 min (3 steps of 5 min each)
    log(f"  [Cycle {cycle_num}] Phase 2: Ramp up (15 min)...")
    for rate, label in [(5, "5 req/sec"), (10, "10 req/sec"), (15, "15 req/sec")]:
        log(f"    Ramp: {label} for 5 min...")
        send_traffic(rate, 300, cycle_latencies)
    log(f"  [Cycle {cycle_num}] Ramp done. epsilon={get_epsilon()} servers={get_server_count()}")

    # Phase 3: Spike - 20 min
    log(f"  [Cycle {cycle_num}] Phase 3: SPIKE (20 min, 20 req/sec)...")
    send_traffic(20, 1200, spike_latencies)
    spike_avg = round(statistics.mean(spike_latencies), 1) if spike_latencies else 0
    spike_p95 = round(sorted(spike_latencies)[int(len(spike_latencies) * 0.95)], 1) if spike_latencies else 0
    log(f"  [Cycle {cycle_num}] Spike done. avg={spike_avg}ms p95={spike_p95}ms epsilon={get_epsilon()} servers={get_server_count()}")

    # Phase 4: Ramp down - 15 min (3 steps of 5 min each)
    log(f"  [Cycle {cycle_num}] Phase 4: Ramp down (15 min)...")
    for rate, label in [(15, "15 req/sec"), (10, "10 req/sec"), (5, "5 req/sec")]:
        log(f"    Ramp: {label} for 5 min")
        send_traffic(rate, 300, cycle_latencies)
    log(f"  [Cycle {cycle_num}] Ramp down done. epsilon={get_epsilon()} servers={get_server_count()}")

    # Phase 5: Recovery - 20 min
    log(f"  [Cycle {cycle_num}] Phase 5: Recovery (20 min, 2 req/sec)...")
    send_traffic(2, 1200, cycle_latencies)
    log(f"  [Cycle {cycle_num}] Recovery done. epsilon={get_epsilon()} servers={get_server_count()}")

    log(f"--- CYCLE {cycle_num}/{total_cycles} COMPLETE ---")

    return {
        "cycle": cycle_num,
        "spike_avg": spike_avg,
        "spike_p95": spike_p95,
        "epsilon_end": get_epsilon(),
        "servers_end": get_server_count(),
    }


# ── SETUP ──────────────────────────────────────────────────────────────────────

NUM_CYCLES = 5  # 5 cycles about 7-8 hours 

print("=" * 60)
print("Q-LEARNING CONVERGENCE TEST")
print(f"Cycles: {NUM_CYCLES} | Est. duration: ~{NUM_CYCLES * 100 // 60}h {(NUM_CYCLES * 100) % 60}min")
print("Each cycle: 30min low -> 15min ramp up -> 20min spike -> 15min ramp down -> 20min recovery")
print("=" * 60)
print()
print("Make sure Q-learning is ENABLED (no scaling override active).")
print("Verify with: curl http://localhost:8000/scaling/status")
print()
input("Press ENTER to start...")

# Make sure scaling override is off
try:
    requests.post(f"{LB_URL}/scaling/override", json={"fixed_servers": None})
    log("Scaling override cleared - Q-learning enabled")
except:
    log("Warning: could not clear scaling override - check manually")

time.sleep(3)

start_time = datetime.now()
cycle_results = []

for i in range(1, NUM_CYCLES + 1):
    result = run_cycle(i, NUM_CYCLES)
    cycle_results.append(result)

# ── FINAL RESULTS

print()
print("=" * 60)
print("CONVERGENCE RESULTS - SPIKE PHASE PER CYCLE")
print("=" * 60)
print(f"{'Cycle':<8} {'Spike Avg':>12} {'Spike P95':>12} {'Epsilon':>10} {'Servers':>10}")
print("-" * 56)
for r in cycle_results:
    print(f"{r['cycle']:<8} {str(r['spike_avg'])+'ms':>12} {str(r['spike_p95'])+'ms':>12} {str(r['epsilon_end']):>10} {str(r['servers_end']):>10}")

print()
first_spike = cycle_results[0]["spike_avg"]
last_spike = cycle_results[-1]["spike_avg"]
if first_spike > 0:
    improvement = round((first_spike - last_spike) / first_spike * 100, 1)
    direction = "better" if improvement > 0 else "worse"
    print(f"Spike avg cycle 1 -> cycle {NUM_CYCLES}: {first_spike}ms -> {last_spike}ms ({abs(improvement)}% {direction})")

total_minutes = round((datetime.now() - start_time).total_seconds() / 60, 1)
print(f"Total runtime: {total_minutes} minutes")
print("=" * 60)