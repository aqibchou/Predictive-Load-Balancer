# Predictive Load Balancer

A proactive load balancing system that predicts traffic patterns and makes intelligent routing and scaling decisions using three AI techniques.

---

## Overview

Traditional load balancers react to traffic after it arrives. This system predicts traffic 1-5 minutes ahead and scales resources proactively.

- **Facebook Prophet** — Time-series forecasting for traffic prediction
- **A* Search** — Intelligent request routing with cache-aware heuristics  
- **Q-Learning** — Reinforcement learning for adaptive auto-scaling
- **Prometheus + Grafana** — Real-time observability and metrics dashboards

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
│  Prometheus  │ ◄─────────────────────── │ Load Balancer│
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
│   ├── 01_extract_data.py          # Parse NASA logs → CSV
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
├── test_prophet.py                 # Prophet prediction test
├── test_traffic.py                 # Basic traffic test
├── traffic_spike.py                # Traffic spike simulation
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

## Monitoring — Grafana Dashboard

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

## Testing

### Basic routing

```bash
# Send a request (A* routes to best server)
curl http://localhost:8000/api/test

# Check which server was selected and latency
curl http://localhost:8000/stats
```

### Run the routing comparison experiment

Compares round-robin, greedy, and A* under identical traffic (100 requests each):

```bash
python compare_routing.py
```

Results from controlled experiments (100 requests × 4 trials):

| Algorithm | Avg Latency | P95 |
|-----------|-------------|-----|
| Round-Robin | 130.6ms | 179.9ms |
| Greedy | 185.3ms | 241.7ms |
| A* | 123.4ms | 157.5ms |

A* was 34% faster than greedy and 6% faster than round-robin on average latency.

### Prophet prediction test

Simulates 12 minutes of traffic with spikes and watches Prophet respond:

```bash
python test_prophet.py
```

### Traffic spike simulation

```bash
python traffic_spike.py
```

### Switch routing mode at runtime

No restart required:

```bash
# Switch to round-robin
curl -X POST http://localhost:8000/routing/mode/round_robin

# Switch to greedy
curl -X POST http://localhost:8000/routing/mode/greedy

# Switch back to A*
curl -X POST http://localhost:8000/routing/mode/astar

# Check current mode
curl http://localhost:8000/routing/mode
```

---

## API Endpoints

### Routing and stats

```bash
# Current predictions from Prophet
curl http://localhost:8000/prediction

# A* score breakdown for a specific path
curl http://localhost:8000/routing/debug/api/users

# Q-learning agent status and recent decisions
curl http://localhost:8000/scaling/status

# Q-table (learned state-action values)
curl http://localhost:8000/scaling/qtable

# Overall system stats
curl http://localhost:8000/stats

# Prometheus metrics (scraped by Prometheus every 5s)
curl http://localhost:8000/metrics
```

---

## AI Components

### Prophet — Traffic Prediction

Prophet decomposes traffic into trend, daily seasonality, and weekly seasonality:

```
y(t) = trend(t) + seasonality(t) + noise(t)
```

Trained on 1.89M NASA HTTP log entries. Runs every 60 seconds and outputs a 5-minute ahead prediction with confidence intervals. Wide confidence intervals trigger conservative scaling; narrow intervals allow more aggressive decisions.

**Results:** MAE = 11.23 req/min, RMSE = 14.79, R² = 0.659

### A* — Intelligent Routing

Scores each server using a multi-factor heuristic. Lowest score wins:

```
score = (active_requests × 10)
      + (avg_response_time × 0.5)
      + (1000 if unhealthy)
      - (50 if request path is cached)
```

| Factor | Weight | Rationale |
|--------|--------|-----------|
| Active requests | ×10 | Penalizes busy servers |
| Avg response time | ×0.5 | Penalizes historically slow servers |
| Unhealthy | +1000 | Removes from consideration |
| Cache hit | −50 | Rewards cache locality |

Server state is refreshed every 5 seconds. The cache bonus reflects that a 50ms cache hit on a loaded server beats a 200ms cache miss on an idle one.

### Q-Learning — Auto-Scaling

Learns scaling policies through trial and error using a 135-state space:

| Component | Values |
|-----------|--------|
| Server count | 1–5 |
| Load level | low / medium / high |
| Prediction trend | decreasing / stable / increasing |
| Latency status | ok / warning / critical |

**Actions:** scale_up, scale_down, hold (60-second cooldown between scaling actions)

**Reward function:**

| Condition | Reward |
|-----------|--------|
| Latency > 200ms | −10 |
| Latency 100–200ms | −5 |
| Each running server | −0.5 |
| Low latency + few servers | +5 |
| Any scaling action | −1 |

Q-value update: `Q(s,a) ← Q(s,a) + α[r + γ · max Q(s′,a′) − Q(s,a)]`  
Learning rate α = 0.1, discount factor γ = 0.95, epsilon decays from 1.0 → 0.1.

---

## Dataset

**NASA HTTP Server Logs — July 1995**

| Metric | Value |
|--------|-------|
| Raw log entries | 1,891,715 |
| Extraction success rate | 99.74% |
| Cleaning retention | 99.96% |
| Minute-level samples | 39,470 |
| Engineered features | 23 |

Train/test split: 80/20 chronological (no shuffling — prevents data leakage in time-series).

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

Mohammed Qureshi 
