"""
Benchmark script — measures A* vs Round-Robin routing, BiStacking prediction
accuracy, cache performance, and Q-learning scaling behaviour.

Phases
------
  1. Warmup          — 60 req, mixed paths, give A* state time to populate
  2. A* baseline     — 300 req, record per-request latency
  3. Switch mode     — POST /routing/mode/round_robin
  4. Round-Robin     — 300 req, same paths, same concurrency
  5. Switch back     — POST /routing/mode/astar
  6. Spike           — 150 concurrent requests (simulate burst)
  7. Recovery        — 60 req at low rate

Metrics reported
----------------
  - Avg / P50 / P95 / P99 latency per phase
  - Cache hit rate
  - Requests routed per backend
  - BiStacking: current prediction vs traffic during test
  - Q-learning: decisions made during test
"""

import asyncio
import statistics
import time
import httpx
import json
from datetime import datetime

BASE = "http://localhost:8000"

# Paths that will exercise the cache (repeated keys → hits)
PATHS = [
    "/data?key=user_profile",
    "/data?key=product_list",
    "/data?key=homepage",
    "/compute?iterations=200",
    "/data?key=user_profile",   # repeated → cache hit
    "/data?key=product_list",   # repeated → cache hit
    "/data?key=cart_items",
    "/compute?iterations=350",
    "/data?key=homepage",       # repeated → cache hit
    "/data?key=recommendations",
]


async def req(client: httpx.AsyncClient, path: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.get(BASE + path, timeout=10.0)
        ms = (time.perf_counter() - t0) * 1000
        body = r.json() if r.status_code == 200 else {}
        return {
            "ms": ms,
            "status": r.status_code,
            "server": body.get("backend_response", {}).get("server_id", "?"),
            "cached": body.get("backend_response", {}).get("cached", False),
        }
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return {"ms": ms, "status": 0, "server": "error", "cached": False}


async def run_phase(name: str, n: int, concurrency: int = 10, delay: float = 0.05) -> list:
    print(f"\n[{name}] sending {n} requests (concurrency={concurrency})...")
    results = []
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def bounded(path):
            async with sem:
                r = await req(client, path)
                await asyncio.sleep(delay)
                return r

        tasks = [bounded(PATHS[i % len(PATHS)]) for i in range(n)]
        results = await asyncio.gather(*tasks)

    return list(results)


def stats(results: list, label: str):
    latencies = [r["ms"] for r in results if r["status"] == 200]
    errors    = sum(1 for r in results if r["status"] != 200)
    cached    = sum(1 for r in results if r.get("cached"))
    servers   = {}
    for r in results:
        s = r["server"]
        servers[s] = servers.get(s, 0) + 1

    if not latencies:
        print(f"  {label}: NO successful responses")
        return {}

    lat_sorted = sorted(latencies)
    n = len(lat_sorted)

    def pct(p):
        idx = max(0, int(p / 100 * n) - 1)
        return round(lat_sorted[idx], 1)

    result = {
        "label":    label,
        "n":        len(results),
        "success":  len(latencies),
        "errors":   errors,
        "avg_ms":   round(statistics.mean(latencies), 1),
        "p50_ms":   pct(50),
        "p95_ms":   pct(95),
        "p99_ms":   pct(99),
        "min_ms":   round(min(latencies), 1),
        "max_ms":   round(max(latencies), 1),
        "cache_hits":  cached,
        "cache_rate":  round(cached / len(results) * 100, 1),
        "servers":  servers,
    }

    print(f"  n={result['n']}  success={result['success']}  errors={errors}")
    print(f"  avg={result['avg_ms']}ms  p50={result['p50_ms']}ms  "
          f"p95={result['p95_ms']}ms  p99={result['p99_ms']}ms")
    print(f"  cache hits={cached}/{len(results)} ({result['cache_rate']}%)")
    print(f"  backends: {servers}")
    return result


async def set_routing_mode(mode: str):
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/routing/mode/{mode}", timeout=5.0)
        print(f"  Routing mode → {mode} ({r.status_code})")


async def get_snapshot(label: str) -> dict:
    async with httpx.AsyncClient() as client:
        pred    = (await client.get(f"{BASE}/prediction",     timeout=5.0)).json()
        scaling = (await client.get(f"{BASE}/scaling/status", timeout=5.0)).json()
        qtable  = (await client.get(f"{BASE}/scaling/qtable", timeout=5.0)).json()
    print(f"\n[{label}] snapshot")
    print(f"  prediction: 1min={pred.get('predicted_1min')}  "
          f"5min={pred.get('predicted_5min')}  "
          f"uncertainty={pred.get('uncertainty')}  "
          f"model_loaded={pred.get('model_loaded')}")
    print(f"  scaling: servers={scaling.get('current_servers')}  "
          f"epsilon={scaling.get('epsilon')}")
    recent = scaling.get("recent_decisions", [])
    if recent:
        last = recent[-1]
        print(f"  last decision: {last['action']}  "
              f"taken={last['action_taken']}  state={last['state']}")
    print(f"  Q-table: {qtable.get('total_states')} states  "
          f"{qtable.get('non_zero_states')} non-zero")
    return {"prediction": pred, "scaling": scaling, "qtable": qtable}


async def main():
    print("=" * 60)
    print("LOAD BALANCER BENCHMARK")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # ── Warmup ───────────────────────────────────────────────────
    print("\n>>> PHASE 1: WARMUP (populating A* state)")
    warmup = await run_phase("warmup", 60, concurrency=5, delay=0.05)
    stats(warmup, "Warmup")
    snap_before = await get_snapshot("Before A* baseline")

    await asyncio.sleep(2)

    # ── A* baseline ──────────────────────────────────────────────
    print("\n>>> PHASE 2: A* ROUTING (300 requests)")
    astar_results = await run_phase("astar", 300, concurrency=15, delay=0.03)
    astar_stats = stats(astar_results, "A* Routing")

    await asyncio.sleep(1)

    # ── Switch to Round-Robin ─────────────────────────────────────
    print("\n>>> PHASE 3: ROUND-ROBIN ROUTING (300 requests)")
    await set_routing_mode("round_robin")
    await asyncio.sleep(1)
    rr_results = await run_phase("round_robin", 300, concurrency=15, delay=0.03)
    rr_stats = stats(rr_results, "Round-Robin")
    await set_routing_mode("astar")

    await asyncio.sleep(1)

    # ── Spike ─────────────────────────────────────────────────────
    print("\n>>> PHASE 4: TRAFFIC SPIKE (150 concurrent)")
    spike_results = await run_phase("spike", 150, concurrency=50, delay=0.0)
    spike_stats = stats(spike_results, "Spike (50-concurrent)")

    await asyncio.sleep(2)

    # ── Recovery ──────────────────────────────────────────────────
    print("\n>>> PHASE 5: RECOVERY (60 requests, low rate)")
    recovery = await run_phase("recovery", 60, concurrency=3, delay=0.15)
    recovery_stats = stats(recovery, "Recovery")

    snap_after = await get_snapshot("After all phases")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    table = [astar_stats, rr_stats, spike_stats, recovery_stats]
    print(f"\n{'Phase':<25} {'Avg':>7} {'P50':>7} {'P95':>7} {'P99':>7} {'Cache%':>8} {'Errors':>7}")
    print("-" * 65)
    for row in table:
        if row:
            print(f"{row['label']:<25} {row['avg_ms']:>7} {row['p50_ms']:>7} "
                  f"{row['p95_ms']:>7} {row['p99_ms']:>7} "
                  f"{row['cache_rate']:>7}% {row['errors']:>7}")

    if astar_stats and rr_stats:
        avg_diff = round((rr_stats["avg_ms"] - astar_stats["avg_ms"]) / rr_stats["avg_ms"] * 100, 1)
        p95_diff = round((rr_stats["p95_ms"] - astar_stats["p95_ms"]) / rr_stats["p95_ms"] * 100, 1)
        print(f"\nA* vs Round-Robin: avg {avg_diff:+.1f}%  p95 {p95_diff:+.1f}%")

    pred = snap_after["prediction"]
    print(f"\nBiStacking prediction (end of test):")
    print(f"  1min={pred.get('predicted_1min')}  3min={pred.get('predicted_3min')}  "
          f"5min={pred.get('predicted_5min')}")
    print(f"  uncertainty={pred.get('uncertainty')}  model_loaded={pred.get('model_loaded')}")

    sc = snap_after["scaling"]
    print(f"\nQ-Learning (end of test):")
    print(f"  epsilon={sc.get('epsilon')}  current_servers={sc.get('current_servers')}")
    decisions = sc.get("recent_decisions", [])
    if decisions:
        counts = {}
        for d in decisions:
            counts[d["action"]] = counts.get(d["action"], 0) + 1
        print(f"  decision distribution: {counts}")
        taken = [d for d in decisions if d.get("action_taken")]
        print(f"  scaling actions executed: {len(taken)}")

    print(f"\nTest completed: {datetime.now().isoformat()}")
    print("=" * 60)

    return {
        "astar": astar_stats,
        "round_robin": rr_stats,
        "spike": spike_stats,
        "recovery": recovery_stats,
        "snapshot_after": snap_after,
    }


if __name__ == "__main__":
    results = asyncio.run(main())
