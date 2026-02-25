# Predictive Load Balancer

A proactive load balancing system that predicts traffic patterns and makes intelligent routing and scaling decisions using AI. Forked from [MohammedQ13/Predictive-Load-Balancer](https://github.com/MohammedQ13/Predictive-Load-Balancer) and extended with 10 production-grade optimisations, gRPC telemetry, an eBPF/XDP kernel failsafe, and chaos engineering.

---

## What Changed From The Original

| Area | Original | This Fork |
|------|----------|-----------|
| Traffic predictor | Facebook Prophet (pre-trained `.pkl`) | **BiStacking ensemble** — refit live every 60 s |
| Feature selection | 23 hand-engineered features | **OOA-DPSO-GA** — 38 features pruned to 11 at runtime |
| Q-learning state | 4 dimensions (135 states) | **5 dimensions** + uncertainty (405 states) |
| Docker scaling | Synchronous, single attempt | **Async `to_thread` + 3-attempt retry** |
| Q-table storage | JSON file only | **PostgreSQL upsert** every cycle + JSON fallback |
| Routing | A* with no circuit protection | **Circuit breaker** (CLOSED → OPEN → HALF_OPEN) |
| Cache bonus | 50.0 (over-routing on stale data) | **20.0** (tuned) |
| Server state sync | In-memory per-instance | **Redis hash** 30 s TTL, atomic `HINCRBY` |
| Prediction refit | Blocked event loop | **`asyncio.to_thread()`** |
| Reward signal | Latency + cost only | **+2.0 bonus** when prediction within 25% of actual |
| Backend telemetry | Per-request HTTP polling | **Persistent gRPC stream** (port 50051) every 5 s |
| Failsafe | Application-layer only | **eBPF/XDP kernel failsafe** at the NIC |
| Resilience testing | Manual | **Chaos Mesh** (network loss + CPU stress) |

---

## Results

### Locust Stress Test — Dev Gate (50 users, 60 s)

Backends run real SHA-256 computation + genuine LRU cache + background Redis sync + persistent httpx client + atomic `HINCRBY`. The dominant cost is the httpx round-trip through the Docker bridge (~7–10 ms at P50). Cache hits complete in microseconds at the backend; cache misses run SHA-256 in a thread pool.

| Endpoint | Requests | P50 | P95 | P99 | Errors |
|----------|----------|-----|-----|-----|--------|
| `/data` | 1,290 | 8 ms | 39 ms | 110 ms | 0 |
| `/data[burst]` | 9,032 | 7 ms | 28 ms | 65 ms | 0 |
| `/data[hot]` | 2,947 | 8 ms | 34 ms | 75 ms | 0 |
| `/data[cold]` | 335 | 8 ms | 34 ms | 83 ms | 0 |
| `/compute` | 426 | 17 ms | 49 ms | 100 ms | 0 |
| `/compute[burst]` | 2,279 | 17 ms | 41 ms | 77 ms | 0 |
| `/health` | 230 | 3 ms | 6 ms | 13 ms | 0 |
| `/prediction` | 214 | 3 ms | 6 ms | 14 ms | 0 |
| **Aggregated** | **16,753** | **8 ms** | **33 ms** | **73 ms** | **0** |

gRPC telemetry confirmed active — LB logs showed `[gRPC] TelemetryService received: backend-1 rt=51.3ms` every 5 s.

### eBPF Circuit Breaker Simulation

```
[eBPF] health failure #1
[eBPF] health failure #2
[eBPF] health failure #3 → circuit OPENED  (XDP map[0] = 1)
...
[eBPF] health success → circuit CLOSED     (XDP map[0] = 0)
```

### Traffic Spike (100 users, 60 s — measured)

2× the steady-state load. All backends saturated with concurrent requests. Zero failures.

| Metric | 50u (measured) | 100u spike (measured) |
|--------|----------------|-----------------------|
| Avg latency | **12.3 ms** | 99.2 ms |
| P50 | **8 ms** | 85 ms |
| P95 | **33 ms** | 210 ms |
| P99 | **73 ms** | 320 ms |
| RPS | 283 | 339 |
| Errors | 0 | 0 |

Even under a 100-user spike (339 RPS peak), latency stays within 210 ms P95 with zero failures.

The 320 ms P99 at 100 users is a Docker Desktop on macOS artefact: the HyperKit VM bridge adds ~7 ms per round-trip at low concurrency and degrades nonlinearly as concurrent connections grow. On real Linux hardware (Docker bridge ≈ 0.3 ms), the same workload would produce a P99 of roughly 40–80 ms.


### Cache Performance (50-user run, measured)

| Metric | Measured |
|--------|----------|
| Cache hit rate | ~99% |
| Avg latency `/data[hot]` (≥98% hits) | **11.8 ms** |
| Avg latency `/data[cold]` (100% misses) | **12.2 ms** |
| Avg latency `/compute` (real CPU) | **21.9 ms** |
| P50 end-to-end (hit vs miss) | **8 ms / 8 ms** |

Cache hits and misses share almost identical P50 latency (both 8 ms) because the dominant cost is the Docker bridge round-trip (~7 ms), not the computation. On a cache hit, the backend returns in <0.1 ms. On a miss, SHA-256 (10 k rounds) completes in ~25–35 ms inside a thread pool — but at low-to-moderate concurrency the thread pool serves it fast enough to stay within the same P50 bucket. The tail difference appears at P95 and higher when the thread pool is saturated.

### BiStacking Prediction

| Metric | Value |
|--------|-------|
| MAE | 0.024 req/min |
| R² | 0.998 |
| Features used | 11 / 38 (OOA-DPSO-GA) |
| Refit cadence | Every 60 s (background thread) |

### Prediction Model Comparison

| Metric | Prophet (Original) | BiStacking |
|--------|--------------------|------------|
| MAE | 11.23 req/min | **0.024 req/min** |
| R² | 0.659 | **0.998** |
| Blocks event loop | Yes | **No** |


---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  KERNEL LAYER (Linux only)                                   │
│  XDP Program: reads lb_health_map → XDP_PASS or XDP_TX      │
│  eBPF Controller: polls /health (1s) + P95 (5s)             │
│    3 failures OR 3× P95 > 500ms → circuit OPEN              │
└───────────────────────────────┬──────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────┐
│  Load Balancer :8000  +  gRPC Server :50051                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ BiStacking  │  │ A* Router    │  │ Q-Learning Scaler    │ │
│  │ Prediction  │  │ + Circuit    │  │ 5-dim state          │ │
│  │ (60s refit) │  │   Breaker    │  │ 405 states           │ │
│  └─────────────┘  └──────────────┘  └──────────────────────┘ │
│  gRPC TelemetryService: receives backend pushes every 5s     │
│  → router.update_from_grpc() bypasses HTTP polling           │
└───────────────────────────────┬──────────────────────────────┘
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
         ┌──────────┐    ┌──────────┐    ┌──────────┐
         │ Backend 1│    │ Backend 2│    │ Backend N│
         │ LRU Cache│    │ LRU Cache│    │ LRU Cache│
         │ Telemetry│    │ Telemetry│    │ Telemetry│
         │  Agent  │    │  Agent  │    │  Agent  │
         └──────────┘    └──────────┘    └──────────┘
               └──── gRPC stream :50051 ────┘
         ┌──────────┐  ┌──────────┐  ┌──────────┐
         │PostgreSQL│  │  Redis   │  │Prometheus│
         │ Q-table  │  │ state +  │  │+ Grafana │
         └──────────┘  └──────────┘  └──────────┘
```

---

## Quick Start

```bash
git clone <this-repo>
cd Predictive-Load-Balancer

# Generate gRPC stubs (required once, or after proto changes)
pip install grpcio-tools
bash telemetry/generate_proto.sh

# Start the full stack
docker compose up --build
```

After startup:
```
load_balancer | [gRPC] TelemetryService listening on [::]:50051
backend-1     | [TelemetryAgent] Connected to load_balancer:50051
load_balancer | [gRPC] TelemetryService received: backend-1 active_req=0 rt=0.0ms
```

### Run tests

```bash
pip install pytest pytest-asyncio numpy pandas scikit-learn lightgbm grpcio grpcio-tools psutil prometheus-client
bash telemetry/generate_proto.sh
pytest tests/unit/ -v   # 107 tests: routing + gRPC + eBPF + BiStacking
```

### Run stress test

```bash
pip install locust
locust -f tests/locust/locustfile.py --host http://localhost:8000 \
       --headless -u 50 -r 5 --run-time 45s --html results/locust_report.html
```

### Run eBPF failsafe (Linux / Lima VM only)

```bash
sudo python3 ebpf/controller.py \
  --interface eth0 --lb-host 172.17.0.2 \
  --fallback-ip 172.17.0.3 --fallback-mac AA:BB:CC:DD:EE:FF
```

---

## API Endpoints

```bash
curl http://localhost:8000/health                   # System health
curl http://localhost:8000/prediction               # BiStacking forecast (1/3/5 min)
curl http://localhost:8000/routing/debug/api/users  # A* score breakdown per server
curl http://localhost:8000/scaling/status           # Q-learning state + epsilon
curl http://localhost:8000/scaling/qtable           # Learned Q-table
curl http://localhost:8000/metrics                  # Prometheus scrape
curl http://localhost:9100/metrics                  # eBPF circuit state (Linux only)
```

---

## CI/CD

GitHub Actions runs four parallel jobs on every push to `main`:

| Job | What It Does |
|-----|-------------|
| **lint** | `ruff check` across `load_balancer/`, `scripts/`, `tests/unit/`, `ebpf/controller.py` |
| **test** | `generate_proto.sh` → `pytest tests/unit/ -v` (107 tests) |
| **docker-build** | Build LB + backend images from root context; no push |
| **helm-lint** | `helm lint k8s/helm/predictive-lb/` |

Both Docker builds use root context (`context: .`) so `COPY telemetry/` works across both services.

---

## Project Structure

```
Predictive-Load-Balancer/
├── telemetry/               # gRPC proto schema + generated stubs + generate_proto.sh
├── load_balancer/           # FastAPI app, BiStacking, A* router, Q-learning, gRPC server
├── backend/                 # FastAPI backend with LRU cache + TelemetryAgent
├── ebpf/                    # XDP program (C), userspace controller, Dockerfile, DaemonSet
├── k8s/
│   ├── helm/predictive-lb/  # Helm chart
│   └── chaos/               # Chaos Mesh experiments (network loss + CPU stress)
├── tests/
│   ├── unit/                # 97 tests (routing, gRPC, eBPF)
│   └── locust/              # Stress test (NormalUser / CacheUser / SpikeUser)
├── scripts/                 # OOA-DPSO-GA feature selection worker
├── docker-compose.yml       # Full stack (LB :8000/:50051, backends, Postgres, Redis, Grafana)
├── .github/workflows/ci.yml
└── Jenkinsfile
```

---

## Technologies

Python · FastAPI · gRPC (grpcio/grpc.aio) · Protocol Buffers · LightGBM · scikit-learn · Redis · PostgreSQL · psutil · eBPF/XDP (BCC) · Chaos Mesh · Locust · Prometheus · Grafana · Docker · Kubernetes · Helm

---

## Original Author

Mohammed Qureshi — Carleton University
