"""
Predictive Load Balancer — Locust Stress Test
==============================================
Three user classes model real-world traffic patterns:

  NormalUser  (weight 5) — steady background: mixed /data and /compute
  CacheUser   (weight 3) — hot-key repeat access: exercises LRU cache + A* locality
  SpikeUser   (weight 2) — high-frequency bursts: triggers Q-learning scale-up

Run (interactive UI):
  locust -f tests/locust/locustfile.py --host http://localhost:8000

Run (headless / CI):
  locust -f tests/locust/locustfile.py --host http://localhost:8000 \\
         --headless -u 100 -r 10 --run-time 5m \\
         --html results/locust_report.html

Run against Kubernetes:
  LB_IP=$(kubectl get svc predictive-lb-loadbalancer -n predictive-lb \\
          -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  locust -f tests/locust/locustfile.py --host http://$LB_IP:8000 \\
         --headless -u 200 -r 20 --run-time 10m \\
         --html results/k8s_stress_report.html

Metrics during the run:
  curl http://localhost:8000/prediction     → BiStacking forecast
  curl http://localhost:8000/scaling/status → Q-learning decisions
  curl http://localhost:8000/metrics        → Prometheus scrape
"""

import random

from locust import HttpUser, between, constant_pacing, events, task

# ── Key sets ──────────────────────────────────────────────────────────────────
CACHE_KEYS = [
    "user_profile", "product_list", "homepage",
    "cart_items",   "recommendations", "search_results",
    "trending",     "feed",            "settings",
]
HOT_KEYS = ["homepage", "user_profile"]   # small hot set for cache locality


# ── User classes ──────────────────────────────────────────────────────────────

class NormalUser(HttpUser):
    """Steady background traffic — 3 req/s per user, mixed workload."""

    weight    = 5
    wait_time = between(0.3, 1.0)

    @task(6)
    def data_request(self):
        key = random.choice(CACHE_KEYS)
        with self.client.get(f"/data?key={key}", name="/data", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")

    @task(2)
    def compute_request(self):
        iters = random.choice([100, 200, 300, 500])
        with self.client.get(f"/compute?iterations={iters}", name="/compute", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")

    @task(1)
    def check_prediction(self):
        self.client.get("/prediction", name="/prediction")


class CacheUser(HttpUser):
    """Hot-key access — exercises cache locality and A* cache-bonus routing."""

    weight    = 3
    wait_time = between(0.1, 0.4)

    @task(9)
    def hot_key_request(self):
        key = random.choice(HOT_KEYS)
        with self.client.get(f"/data?key={key}", name="/data[hot]", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"status {r.status_code}")

    @task(1)
    def cold_key_request(self):
        """Occasional cold miss to keep the cache from going 100% stale."""
        key = f"cold_{random.randint(1, 1000)}"
        self.client.get(f"/data?key={key}", name="/data[cold]")


class SpikeUser(HttpUser):
    """High-frequency burst — 20 req/s per user, triggers Q-learning scale-up."""

    weight    = 2
    wait_time = constant_pacing(0.05)

    @task(4)
    def burst_data(self):
        key = random.choice(CACHE_KEYS)
        with self.client.get(f"/data?key={key}", name="/data[burst]", catch_response=True) as r:
            if r.status_code not in (200, 503):
                r.failure(f"unexpected status {r.status_code}")

    @task(1)
    def burst_compute(self):
        with self.client.get("/compute?iterations=500", name="/compute[burst]", catch_response=True) as r:
            if r.status_code not in (200, 503):
                r.failure(f"unexpected status {r.status_code}")


# ── Event hooks ───────────────────────────────────────────────────────────────

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "=" * 60)
    print("  PREDICTIVE LOAD BALANCER — STRESS TEST")
    print("=" * 60)
    print(f"  Target : {environment.host}")
    print(f"  Classes: NormalUser(×5) | CacheUser(×3) | SpikeUser(×2)")
    print("  Watch  : /prediction  /scaling/status  /metrics")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.runner.stats.total
    p95   = stats.get_response_time_percentile(0.95)
    p99   = stats.get_response_time_percentile(0.99)

    print("\n" + "=" * 60)
    print("  STRESS TEST COMPLETE")
    print("=" * 60)
    print(f"  Total requests : {stats.num_requests:,}")
    print(f"  Failures       : {stats.num_failures:,}  "
          f"({stats.num_failures / max(stats.num_requests, 1) * 100:.1f}%)")
    print(f"  Avg latency    : {stats.avg_response_time:.1f} ms")
    print(f"  P95 latency    : {p95:.1f} ms")
    print(f"  P99 latency    : {p99:.1f} ms")
    peak_rps = getattr(stats, "max_rps", None) or getattr(stats, "total_rps", None) or 0
    print(f"  Peak RPS       : {peak_rps:.1f}")
    print("=" * 60 + "\n")
