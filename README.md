# Predictive Load Balancer

A proactive load balancing system that predicts traffic patterns and makes intelligent routing and scaling decisions using three AI techniques: time-series forecasting, heuristic search, and reinforcement learning.

---

## Results

### Predictive ML Scaling vs Reactive Scaling
Measured across a 2.5-hour controlled experiment (11,077 requests) with warmup, varied traffic, and a sustained 20-minute spike phase:

| Metric | Reactive (Fixed Servers) | Predictive (Q-Learning + Prophet) | Improvement |
|--------|--------------------------|-----------------------------------|-------------|
| Spike Avg Latency | 2163.8ms | 175.1ms | 91.9% better |
| Spike P50 Latency | 2160.0ms | 117.1ms | 94.6% better |
| Spike P95 Latency | 2200.2ms | 157.4ms | 92.8% better |
| Spike P99 Latency | 2254.3ms | 2180.4ms | 3.3% better |
| Varied Phase Avg | 2168.6ms | 978.7ms | 54.8% better |

The predictive system began outperforming reactive scaling during the varied traffic phase - before the spike arrived - because Prophet detected the increasing trend and Q-learning scaled up proactively. P99 tail latency remained similar in both runs, reflecting the inherent gap between a scaling decision firing and a new container becoming ready to serve traffic. In production this would be addressed with connection pooling or container pre-warming.

### A* Routing vs Baseline Algorithms
Measured across 4 controlled trials (100 requests each, 400 total):

| Algorithm | Avg Latency | P50 | P95 |
|-----------|-------------|-----|-----|
| Round-Robin | 130.6ms | 115.3ms | 179.9ms |
| Greedy | 185.3ms | 177.7ms | 241.7ms |
| A* | 123.4ms | 118.7ms | 157.5ms |

A* was 34% faster than greedy and 6% faster than round-robin. Greedy performed worst despite sounding intuitive - it makes an HTTP call to every server on every request to check load, adding overhead that outweighs the benefit of always picking the least loaded server. A*'s stale-state approach (refreshing every 5 seconds) trades perfect accuracy for real-world speed.

---

## Overview

Traditional load balancers are reactive:
```
Traffic spike hits -> latency degrades -> system detects overload -> scales up -> too late
```

This system is proactive:
```
Prophet predicts spike -> Q-learning scales up -> traffic arrives -> system already ready
```

Three AI components handle different decisions:

- **Facebook Prophet** - predicts traffic 1-5 minutes ahead using historical patterns
- **A* Search** - routes each request to the optimal server using a cache-aware heuristic
- **Q-Learning** - learns when to add or remove servers by balancing latency against cost
- **Prometheus + Grafana** - real-time observability across all components

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Load Balancer                            │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │   Prophet    │   │  A* Router   │   │  Q-Learning Scaler   │  │
│  │ (Prediction) │   │  (Routing)   │   │     (Scaling)        │  │
│  └──────────────┘   └──────────────┘   └──────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Prometheus Metrics (/metrics)                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
         ┌──────────┐   ┌──────────┐   ┌──────────┐
         │ Backend  │   │ Backend  │   │ Backend  │
         │Server 1  │   │Server 2  │   │Server 3  │
         │ (Cache)  │   │ (Cache)  │   │ (Cache)  │
         └──────────┘   └──────────┘   └──────────┘
               │
               ▼
         ┌──────────┐
         │PostgreSQL│
         │(Metrics) │
         └──────────┘

┌──────────────┐     scrapes /metrics      ┌──────────────┐
│  Prometheus  │ <───────────────────────  │ Load Balancer│
│  (port 9090) │                           └──────────────┘
└──────────────┘
       │
       │ datasource
       ▼
┌──────────────┐
│   Grafana    │
│  (port 3000) │
│  Dashboard   │
└──────────────┘
```

---

## Project Structure

```
project-group-101/
├── scripts/                        # ML pipeline
│   ├── 01_extract_data.py          # Parse NASA logs -> CSV
│   ├── 02_clean_data.py            # Remove duplicates/outliers
│   ├── 03_engineer_features.py     # Create 23 predictive features
│   └── 04_train_models.py          # Train Prophet, Ridge, Linear
├── load_balancer/                  # Main service
│   ├── main.py                     # FastAPI app, routing modes, metrics
│   ├── prediction_service.py       # Prophet prediction service
│   ├── routing.py                  # A* routing algorithm
│   ├── scaling.py                  # Q-learning scaler
│   ├── metrics.py                  # Prometheus metrics definitions
│   └── database.py                 # PostgreSQL connection
├── backend/                        # Backend servers
│   └── server.py                   # FastAPI with LRU cache
├── config/
│   ├── servers.yaml                # Backend server configuration
│   ├── prometheus.yml              # Prometheus scrape config
│   └── grafana/
│       ├── datasources/
│       │   └── prometheus.yml      # Grafana datasource (auto-provisioned)
│       └── dashboards/
│           ├── dashboard.yml       # Dashboard provisioning config
│           └── loadbalancer.json   # AI Load Balancer dashboard (auto-provisioned)
├── models/
│   └── prophet_model.pkl           # Trained Prophet model
├── data/
│   ├── NASA_access_log_Jul95       # Raw logs (1.89M entries)
│   ├── extracted_logs.csv
│   ├── cleaned_logs.csv
│   └── featured_traffic.csv
├── results/
│   ├── model_comparison.csv
│   └── predictions_plot.png
├── tests/                          # Test scripts
├── compare_routing.py              # Routing algorithm comparison experiment
├── ml_vs_reactive.py               # 2.5-hour reactive vs predictive experiment
├── test_prophet.py                 # Prophet prediction test
├── test_traffic.py                 # Basic traffic test
└── docker-compose.yml
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose

### 1. Clone the repository

```bash
git clone https://github.com/2025F-COMP3106/project-group-101
cd project-group-101
```

### 2. Train the ML models

```bash
pip install -r requirements.txt

python scripts/01_extract_data.py
python scripts/02_clean_data.py
python scripts/03_engineer_features.py
python scripts/04_train_models.py
```

### 3. Start the system

```bash
docker-compose up --build
```

This starts six containers: load balancer, three backend servers, PostgreSQL, Prometheus, and Grafana.

### 4. Verify everything is running

```bash
curl http://localhost:8000/health
```

---

## Monitoring - Grafana Dashboard

Open `http://localhost:3000` and login with `admin` / `admin`.

The dashboard auto-provisions on startup and contains five panels:

| Panel | What it shows |
|-------|---------------|
| Traffic: Current vs Predicted | Prophet's 5-minute prediction with confidence bounds |
| Request Latency | P50/P95/P99 latency over time |
| Q-Learning Scaling Actions | scale_up / scale_down / hold decisions over time |
| Q-Learning: Reward & Epsilon | Agent reward and epsilon decay |
| A* Server Scores | Per-server heuristic scores used for routing decisions |

Prometheus metrics are also available directly at `http://localhost:9090`.

---

## Experiments

### Reactive vs Predictive Scaling

Runs a 2.5-hour experiment comparing fixed server count (reactive) against Q-learning autoscaling (predictive) under identical traffic conditions: 20-minute warmup, 30-minute varied traffic, 20-minute spike.

```bash
python ml_vs_reactive.py
```

### Routing Algorithm Comparison

Compares round-robin, greedy, and A* under identical traffic (100 requests each). Switches routing mode at runtime without restarting Docker.

```bash
python compare_routing.py
```

### Switch routing mode manually

```bash
curl -X POST http://localhost:8000/routing/mode/round_robin
curl -X POST http://localhost:8000/routing/mode/greedy
curl -X POST http://localhost:8000/routing/mode/astar
curl http://localhost:8000/routing/mode
```

### Prophet prediction test

```bash
python test_prophet.py
```

---

## API Endpoints

```bash
curl http://localhost:8000/prediction               # Prophet 5-minute forecast
curl http://localhost:8000/routing/debug/api/users  # A* score breakdown per server
curl http://localhost:8000/scaling/status           # Q-learning agent status
curl http://localhost:8000/scaling/qtable           # Learned Q-table values
curl http://localhost:8000/stats                    # System-wide request stats
curl http://localhost:8000/metrics                  # Prometheus metrics endpoint
```

---

## AI Components

### Prophet - Traffic Prediction

Prophet decomposes traffic into trend, daily seasonality, and weekly seasonality:

```
y(t) = trend(t) + seasonality(t) + noise(t)
```

Trained on 1.89M NASA HTTP log entries. Runs every 60 seconds and outputs a 5-minute ahead prediction with confidence intervals. Wide confidence intervals signal uncertainty and trigger conservative scaling; narrow intervals allow more aggressive decisions. Prophet's prediction trend feeds directly into Q-learning's state representation, linking the forecasting and scaling components.

**Model results:** MAE = 11.23 req/min, RMSE = 14.79, R² = 0.659

### A* - Intelligent Routing

Scores each server using a multi-factor heuristic. Lowest score wins:

```
score = (active_requests x 10)
      + (avg_response_time x 0.5)
      + (1000 if unhealthy)
      - (50 if request path is cached)
```

| Factor | Weight | Rationale |
|--------|--------|-----------|
| Active requests | x10 | Penalizes busy servers |
| Avg response time | x0.5 | Penalizes historically slow servers |
| Unhealthy | +1000 | Removes from consideration |
| Cache hit | -50 | Rewards cache locality |

Server state is refreshed every 5 seconds. The cache bonus reflects that a 50ms cache hit on a loaded server beats a 200ms cache miss on an idle one. When Q-learning registers a new container, A* immediately begins routing to it - the heuristic naturally favors new servers with zero active requests and no response time penalty.

### Q-Learning - Auto-Scaling

Learns scaling policies through trial and error using a 135-state space:

| Component | Values |
|-----------|--------|
| Server count | 1-5 |
| Load level | low / medium / high |
| Prediction trend | decreasing / stable / increasing |
| Latency status | ok / warning / critical |

**Actions:** scale_up, scale_down, hold (60-second cooldown between scaling actions)

**Reward function:**

| Condition | Reward |
|-----------|--------|
| Latency > 200ms | -10 |
| Latency 100-200ms | -5 |
| Each running server | -0.5 |
| Low latency + few servers | +5 |
| Any scaling action | -1 |

Q-value update: `Q(s,a) <- Q(s,a) + a[r + y * max Q(s',a') - Q(s,a)]`
Learning rate a = 0.1, discount factor y = 0.95, epsilon decays from 1.0 -> 0.1.

The prediction trend from Prophet is one of the four state components, so Q-learning can scale up in response to a forecasted increase before current load metrics degrade. When a scaling decision fires, the new container is spun up via docker-compose and immediately registered with the A* router - closing the full loop between prediction, scaling, and routing.

---

## Key Findings

**Proactive scaling begins before the spike.** During the varied traffic phase, the predictive system averaged 978ms vs 2168ms for reactive - a 54% improvement before the spike even started. Q-learning was already adding servers because Prophet detected the upward trend.

**The middle 95% of requests were transformed.** P50 improved 94.6% and P95 improved 92.8% during the spike. 95% of requests went from over 2 seconds to under 160ms.

**Tail latency is the remaining problem.** P99 improved only 3.3% and max latency was slightly worse in the predictive run. A small number of requests still hit multi-second latency during the brief window between a scaling decision firing and the new container becoming ready. This is a known limitation of interval-based scaling and would require connection pooling or pre-warming to fully address.

**Q-learning achieved 91.9% improvement while still mostly exploring.** Epsilon was approximately 0.7 at the end of the experiment, meaning 70% of decisions were still random. The results represent a lower bound - a fully converged agent would likely perform better.

**Greedy routing is worse than round-robin in practice.** Despite sounding optimal, greedy routing makes an HTTP call to every server on every request to check current load. This overhead makes it the slowest of the three algorithms tested. A*'s periodic state refresh avoids this cost while still making informed decisions.

---

## Limitations

**The reactive baseline used a fixed server count of 2.** A reactive system with more servers pre-provisioned would perform better. The experiment demonstrates the cost of under-provisioning, which is a realistic scenario but not the only comparison point.

**Prophet requires sufficient traffic history.** During the first 20-30 minutes of operation, predictions are less reliable. The system performs best after it has observed at least one full traffic cycle.

**Heuristic weights in A* are hardcoded.** The load, response time, and cache weights were set manually. Adaptive weight tuning based on observed system behavior would improve routing decisions.

**Q-learning starts from scratch on each restart.** The Q-table is not persisted to disk between runs, so the agent begins exploring from epsilon = 1.0 every time the system restarts.

---

## Dataset

**NASA HTTP Server Logs - July 1995**

| Metric | Value |
|--------|-------|
| Raw log entries | 1,891,715 |
| Extraction success rate | 99.74% |
| Cleaning retention | 99.96% |
| Minute-level samples | 39,470 |
| Engineered features | 23 |

Train/test split: 80/20 chronological. Shuffling was explicitly avoided to prevent data leakage - a time-series model must never be evaluated on data that precedes its training window.

A critical issue discovered during development: initial models achieved near-perfect R² scores because lag features inadvertently included current traffic when predicting current traffic. The fix required shifting all lag features by one additional period so that predictions at time T only use data from T-1 and earlier.

---

## Observability

Seven custom Prometheus metrics tracked in real time:

| Metric | Type | Description |
|--------|------|-------------|
| `loadbalancer_requests_total` | Counter | Requests by server and status |
| `loadbalancer_request_latency_seconds` | Histogram | Latency distribution by route |
| `loadbalancer_active_servers` | Gauge | Current server count |
| `loadbalancer_predicted_traffic_5min` | Gauge | Prophet 5-minute prediction |
| `loadbalancer_prediction_upper/lower_bound` | Gauge | Prophet confidence interval |
| `qlearning_actions_total` | Counter | Scaling actions by type |
| `qlearning_epsilon` | Gauge | Current exploration rate |
| `astar_server_scores` | Gauge | Per-server heuristic score |

---

## Technologies

Python, FastAPI, Facebook Prophet, Docker, PostgreSQL, Prometheus, Grafana

---

## Author

Mohammed Qureshi - Carleton University
