import requests
import time
import statistics

# In order to run this experiment, update main.py to allow all modes

LB_URL = "http://localhost:8000"
MODES = ["round_robin", "greedy", "astar"]
REQUESTS_PER_MODE = 100

def run_experiment(mode):
    # Switch mode
    requests.post(f"{LB_URL}/routing/mode/{mode}")
    time.sleep(2)
    
    latencies = []
    errors = 0
    
    print(f"\nRunning {mode} ({REQUESTS_PER_MODE} requests)...")
    
    for i in range(REQUESTS_PER_MODE):
        start = time.time()
        try:
            r = requests.get(f"{LB_URL}/api/test", timeout=10)
            if r.status_code == 200:
                latencies.append((time.time() - start) * 1000)
            else:
                errors += 1
        except Exception:
            errors += 1
        time.sleep(0.1)  # 10 req/sec
    
    return {
        "mode": mode,
        "requests": REQUESTS_PER_MODE,
        "errors": errors,
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }

results = []
for mode in MODES:
    result = run_experiment(mode)
    results.append(result)
    print(f"  avg={result['avg_latency_ms']}ms  p95={result['p95_ms']}ms  errors={result['errors']}")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Mode':<15} {'Avg(ms)':<12} {'P50(ms)':<12} {'P95(ms)':<12} {'P99(ms)':<12} {'Errors'}")
print("-"*60)
for r in results:
    print(f"{r['mode']:<15} {r['avg_latency_ms']:<12} {r['p50_ms']:<12} {r['p95_ms']:<12} {r['p99_ms']:<12} {r['errors']}")

# Reset to astar when done
requests.post(f"{LB_URL}/routing/mode/astar")
print("\nReset to astar mode.")