# Predictive Load Balancer with AI Techniques

A proactive load balancing system that uses machine learning to predict traffic patterns and make intelligent routing and scaling decisions.

**[Project Report](./docs/COMP3106_Project_Report.pdf)**

## Overview

Traditional load balancers react to traffic after it arrives. This system predicts traffic 1-5 minutes ahead and scales resources proactively using three AI techniques:

- **Facebook Prophet** - Time-series forecasting for traffic prediction
- **A* Search** - Intelligent request routing with cache-aware heuristics
- **Q-Learning** - Reinforcement learning for adaptive auto-scaling

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Prophet   │  │  A* Router  │  │  Q-Learning Scaler  │  │
│  │ (Prediction)│  │  (Routing)  │  │     (Scaling)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Backend  │   │ Backend  │   │ Backend  │
        │ Server 1 │   │ Server 2 │   │ Server 3 │
        │ (Cache)  │   │ (Cache)  │   │ (Cache)  │
        └──────────┘   └──────────┘   └──────────┘
```

## Project Structure
```
project-group-101/
├── scripts/                    # ML Pipeline
│   ├── 01_extract_data.py      # Parse NASA logs → CSV
│   ├── 02_clean_data.py        # Remove duplicates/outliers
│   ├── 03_engineer_features.py # Create 23 predictive features
│   └── 04_train_models.py      # Train Prophet, Ridge, Linear
├── load_balancer/              # Main Service
│   ├── main.py                 # FastAPI application
│   ├── prediction.py           # Prophet prediction service
│   ├── routing.py              # A* routing algorithm
│   └── scaling.py              # Q-learning scaler
├── backend/                    # Backend Servers
│   └── server.py               # FastAPI with LRU cache
├── models/                     # Trained Models
│   └── prophet_model.pkl
├── data/                       # Data Files
│   ├── NASA_access_log_Jul95   # Raw logs
│   ├── extracted_logs.csv
│   ├── cleaned_logs.csv
│   └── featured_traffic.csv
├── results/                    # Evaluation Results
│   ├── model_comparison.csv
│   └── predictions_plot.png
├── config/
│   └── servers.yaml            # Server configuration
└── docker-compose.yml          # Container orchestration
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose

### 1. Train the ML Models
```bash
pip install pandas numpy prophet scikit-learn matplotlib

python scripts/01_extract_data.py
python scripts/02_clean_data.py
python scripts/03_engineer_features.py
python scripts/04_train_models.py
```

### 2. Start the System
```bash
docker-compose up
```
### 3. Test the Endpoints

#### Basic Request Routing
```bash
# Send a request (A* routes to best server)
curl http://localhost:8000/api/users

# Response:
{
  "server_id": "server_1",
  "path": "api/users",
  "cached": false,
  "response_time_ms": 203,
  "routed_to": "http://backend_1:8000",
  "timestamp": "2025-12-05T14:23:45.123456"
}

# Send same request again (should hit cache)
curl http://localhost:8000/api/users

# Response (note: cached=true, faster response):
{
  "server_id": "server_1",
  "path": "api/users",
  "cached": true,
  "response_time_ms": 52,
  "routed_to": "http://backend_1:8000",
  "timestamp": "2025-12-05T14:23:47.654321"
}
```

#### Prophet Predictions
```bash
# Get current traffic predictions
curl http://localhost:8000/prediction

# Response:
{
  "current_traffic": 45.2,
  "predicted_1min": 47.8,
  "predicted_3min": 52.1,
  "predicted_5min": 58.4,
  "confidence_lower": 41.2,
  "confidence_upper": 75.6,
  "trend": "increasing",
  "last_updated": "2025-12-05T14:23:00.000000"
}
```

#### A* Routing Debug
```bash
# See how A* scores each server for a specific path
curl http://localhost:8000/routing/debug/api/users

# Response:
{
  "request_path": "api/users",
  "server_scores": [
    {
      "server_id": "server_1",
      "active_requests": 2,
      "avg_response_time_ms": 98,
      "has_cached": true,
      "is_healthy": true,
      "score": 19.0,
      "score_breakdown": {
        "load_penalty": 20,
        "response_penalty": 49.0,
        "health_penalty": 0,
        "cache_bonus": -50
      }
    },
    {
      "server_id": "server_2",
      "active_requests": 0,
      "avg_response_time_ms": 102,
      "has_cached": false,
      "is_healthy": true,
      "score": 51.0,
      "score_breakdown": {
        "load_penalty": 0,
        "response_penalty": 51.0,
        "health_penalty": 0,
        "cache_bonus": 0
      }
    },
    {
      "server_id": "server_3",
      "active_requests": 1,
      "avg_response_time_ms": 95,
      "has_cached": false,
      "is_healthy": true,
      "score": 57.5,
      "score_breakdown": {
        "load_penalty": 10,
        "response_penalty": 47.5,
        "health_penalty": 0,
        "cache_bonus": 0
      }
    }
  ],
  "selected_server": "server_1",
  "reason": "lowest score (cache hit bonus)"
}
```

#### Q-Learning Status
```bash
# Check Q-learning agent status
curl http://localhost:8000/scaling/status

# Response:
{
  "current_servers": 3,
  "current_state": {
    "servers": 3,
    "load_level": "medium",
    "prediction_trend": "increasing",
    "latency_status": "ok"
  },
  "epsilon": 0.342,
  "last_action": "hold",
  "last_reward": 2.5,
  "last_scale_time": "2025-12-05T14:20:00.000000",
  "cooldown_remaining_seconds": 0,
  "recent_decisions": [
    {"time": "14:20:00", "state": "(3, medium, stable, ok)", "action": "hold", "reward": 3.0},
    {"time": "14:19:00", "state": "(3, low, stable, ok)", "action": "hold", "reward": 4.5},
    {"time": "14:18:00", "state": "(4, low, decreasing, ok)", "action": "scale_down", "reward": 2.0}
  ]
}

# View the Q-table (learned values)
curl http://localhost:8000/scaling/qtable

# Response:
{
  "total_states_visited": 47,
  "q_table": {
    "(3, low, stable, ok)": {
      "scale_up": -2.3,
      "scale_down": 1.8,
      "hold": 4.2
    },
    "(2, high, increasing, warning)": {
      "scale_up": 5.1,
      "scale_down": -8.2,
      "hold": -1.5
    },
    "(3, medium, increasing, ok)": {
      "scale_up": 2.1,
      "scale_down": -3.4,
      "hold": 3.8
    }
  }
}
```

#### System Stats
```bash
# Get overall system statistics
curl http://localhost:8000/stats

# Response:
{
  "uptime_seconds": 3842.5,
  "total_requests": 1547,
  "requests_per_server": {
    "server_1": 612,
    "server_2": 498,
    "server_3": 437
  },
  "avg_response_time_ms": 87.3,
  "cache_hit_rate": 0.42,
  "current_load": {
    "server_1": 3,
    "server_2": 1,
    "server_3": 2
  }
}
```

#### Backend Server Metrics
```bash
# Check individual backend server metrics
curl http://localhost:8000/backend/server_1/metrics

# Response:
{
  "server_id": "server_1",
  "active_requests": 2,
  "total_requests": 612,
  "cache_size": 47,
  "cache_hit_rate": 0.58,
  "avg_response_time_ms": 76.2,
  "is_healthy": true
}

# Check what's in a server's cache
curl http://localhost:8000/backend/server_1/cache

# Response:
{
  "server_id": "server_1",
  "cached_paths": [
    "api/users",
    "api/products",
    "images/logo.png",
    "static/main.css"
  ],
  "cache_size": 47,
  "max_cache_size": 100
}
```

#### Health Check
```bash
# System health check
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "components": {
    "load_balancer": "healthy",
    "prophet_model": "loaded",
    "backend_servers": {
      "server_1": "healthy",
      "server_2": "healthy",
      "server_3": "healthy"
    },
    "database": "connected"
  }
}
```

---

### End-to-End Testing

#### Test 1: Cache Locality

Verify A* routes repeated requests to the same server for cache hits.
```bash
# First request (cache miss)
curl http://localhost:8000/products/123
# Note which server handled it (e.g., server_2)

# Second request (should hit same server)
curl http://localhost:8000/products/123
# Should route to server_2 with cached=true

# Verify with routing debug
curl http://localhost:8000/routing/debug/products/123
# server_2 should have lowest score due to cache bonus
```

#### Test 2: Load Distribution

Send multiple different requests and check distribution.
```bash
# Send 30 unique requests
for i in {1..30}; do
  curl -s http://localhost:8000/test/path$i > /dev/null
done

# Check distribution
curl http://localhost:8000/stats
# requests_per_server should be roughly balanced
```

#### Test 3: Prophet + Q-Learning Integration

Watch prediction influence scaling decisions.
```bash
# Terminal 1: Watch predictions
watch -n 5 'curl -s http://localhost:8000/prediction | jq .'

# Terminal 2: Watch Q-learning decisions
watch -n 5 'curl -s http://localhost:8000/scaling/status | jq .'

# Terminal 3: Generate increasing traffic
for i in {1..100}; do
  curl -s http://localhost:8000/test/$RANDOM > /dev/null
  sleep 0.1
done

# Observe: As Prophet predicts increasing traffic,
# Q-learning should consider scaling up
```

#### Test 4: Full Load Test
```bash
# Generate sustained traffic (requires apache bench or similar)
ab -n 1000 -c 10 http://localhost:8000/api/test

# Or with curl loop
for i in {1..500}; do
  curl -s http://localhost:8000/api/item/$((RANDOM % 50)) > /dev/null &
done
wait

# Check results
curl http://localhost:8000/stats
curl http://localhost:8000/scaling/status
```

#### Test 5: Verify A* vs Round-Robin Difference
```bash
# Request same resource 10 times
for i in {1..10}; do
  echo "Request $i:"
  curl -s http://localhost:8000/api/users | jq '{server: .server_id, cached: .cached, time: .response_time_ms}'
  sleep 0.5
done

# Expected with A*: After first request, all subsequent
# requests go to same server with cached=true

# Round-robin would alternate: server_1, server_2, server_3, server_1...
# resulting in repeated cache misses
```

## AI Components

### Prophet Prediction

Predicts traffic volume 1-5 minutes ahead using historical patterns.

- Trained on 1.89M NASA HTTP log entries
- Captures daily and weekly seasonality
- Provides confidence intervals for risk-aware scaling

**Results:** MAE = 11.23 req/min, R² = 0.659

### A* Routing

Selects optimal server using a multi-factor heuristic:
```
score = (load × 10) + (response_time × 0.5) + health_penalty - cache_bonus
```

| Factor | Weight | Effect |
|--------|--------|--------|
| Active requests | ×10 | Penalizes busy servers |
| Avg response time | ×0.5 | Penalizes slow servers |
| Unhealthy | +1000 | Removes from consideration |
| Cache hit | -50 | Rewards cache locality |

### Q-Learning Scaling

Learns optimal scaling policies through reinforcement learning.

- **State:** (server_count, load_level, prediction_trend, latency_status)
- **Actions:** Scale up, Scale down, Hold
- **Reward:** Balances latency SLAs against server costs

## Dataset

**NASA HTTP Server Logs (July 1995)**

| Metric | Value |
|--------|-------|
| Raw log entries | 1,891,715 |
| Extraction success | 99.74% |
| Minute-level samples | 39,470 |
| Engineered features | 23 |

## Results

| Model | MAE (req/min) | R² |
|-------|---------------|-----|
| Prophet | 11.23 | 0.659 |
| Ridge | 11.50 | 0.648 |
| Linear | 11.50 | 0.648 |

## Technologies

- Python, FastAPI, Facebook Prophet, Docker, PostgreSQL

## How It Works

### System Overview

This load balancer combines three AI techniques to make smarter infrastructure decisions:

1. **Predict** - Forecast traffic before it arrives
2. **Route** - Send requests to the best server
3. **Scale** - Add/remove servers automatically

### The Problem with Traditional Load Balancers

Traditional load balancers are reactive:
```
Traffic spike happens → Latency increases → System detects overload → Scales up → Too late
```

This system is proactive:
```
Prophet predicts spike → Q-learning scales up → Traffic arrives → System ready
```

---

## Component 1: Traffic Prediction (Prophet)

### What It Does

Every 60 seconds, Prophet predicts how much traffic will arrive in the next 1-5 minutes.

### How It Works

Prophet decomposes traffic into patterns:
```
traffic(t) = trend + daily_seasonality + weekly_seasonality + noise
```

- **Daily pattern**: High at 2pm, low at 3am
- **Weekly pattern**: Busy on weekdays, quiet on weekends
- **Trend**: Overall growth or decline

### Training Pipeline
```
Raw NASA Logs (1.89M requests)
        ↓
    Extraction (regex parsing)
        ↓
    Cleaning (remove duplicates/outliers)
        ↓
    Feature Engineering (23 features)
        ↓
    Prophet Training (80/20 split)
        ↓
    Saved Model (prophet_model.pkl)
```

### Features Created

| Category | Features | Purpose |
|----------|----------|---------|
| Time | hour, day_of_week, is_weekend | Capture temporal patterns |
| Rolling | 5/15/30/60 min averages | Capture trends |
| Lag | traffic 1/5/15/60 min ago | Recent history |
| Statistical | rate of change, deviation from norm | Detect anomalies |

### Output
```json
{
  "predicted_1min": 45.2,
  "predicted_5min": 52.8,
  "confidence_lower": 38.1,
  "confidence_upper": 67.5
}
```

Wide confidence interval = uncertain prediction = scale conservatively.

---

## Component 2: Intelligent Routing (A*)

### What It Does

For each incoming request, A* picks the best server considering load, cache, and health.

### How It Works

A* calculates a score for each server. Lowest score wins.
```python
score = (active_requests × 10)
      + (avg_response_time × 0.5)
      + (1000 if unhealthy)
      - (50 if request is cached)
```

### Example

Request arrives for `/api/users`:

| Server | Active | Response | Cached | Score |
|--------|--------|----------|--------|-------|
| Server 1 | 2 | 100ms | Yes | (2×10) + (100×0.5) - 50 = **20** |
| Server 2 | 0 | 100ms | No | (0×10) + (100×0.5) - 0 = **50** |
| Server 3 | 1 | 150ms | No | (1×10) + (150×0.5) - 0 = **85** |

**Winner: Server 1** (score 20)

Even though Server 2 has zero load, Server 1 wins because cache hit (50ms) beats cache miss (200ms).

### Why Cache Matters
```
Without cache awareness (round-robin):
  Request 1 → Server 1 (200ms cache miss)
  Request 2 → Server 2 (200ms cache miss)
  Request 3 → Server 3 (200ms cache miss)
  
With A* cache awareness:
  Request 1 → Server 1 (200ms cache miss, now cached)
  Request 2 → Server 1 (50ms cache hit)
  Request 3 → Server 1 (50ms cache hit)
```

---

## Component 3: Auto-Scaling (Q-Learning)

### What It Does

Learns when to add or remove servers by trial and error.

### How It Works

Q-learning builds a table mapping states to actions:
```
State: (3 servers, high load, traffic increasing, latency warning)
  → Best action: SCALE_UP (learned Q-value: +4.1)

State: (4 servers, low load, traffic stable, latency ok)
  → Best action: SCALE_DOWN (learned Q-value: +3.2)
```

### State Space

| Component | Values | Description |
|-----------|--------|-------------|
| Servers | 1-5 | Current server count |
| Load | low/medium/high | Avg CPU across servers |
| Trend | decreasing/stable/increasing | Prophet's prediction |
| Latency | ok/warning/critical | P95 response time |

Total: 5 × 3 × 3 × 3 = **135 possible states**

### Action Space
```
SCALE_UP   - Add one server
SCALE_DOWN - Remove one server
HOLD       - Do nothing
```

### Reward Function

The agent learns from rewards after each action:

| Outcome | Reward | Why |
|---------|--------|-----|
| Latency > 200ms | -10 | Users are unhappy |
| Latency 100-200ms | -5 | Warning zone |
| Each running server | -0.5 | Servers cost money |
| Low latency + few servers | +5 | Efficient operation |
| Any scaling action | -1 | Prevent thrashing |

### Learning Process
```
1. Start with random actions (exploration)
2. Observe rewards
3. Update Q-table: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
4. Gradually shift to best known actions (exploitation)
5. Epsilon decays: 1.0 → 0.1 over time
```

### Integration with Prophet

Q-learning uses Prophet's predictions as part of its state:
```
Current load: 45%
Prophet prediction: 70% in 5 minutes
Trend: INCREASING

→ Q-learning sees increasing trend
→ Learned policy: scale up now
→ New server ready before traffic arrives
```

---

## Request Flow
```
1. Client sends request
        ↓
2. Load balancer receives it
        ↓
3. A* evaluates all servers
   - Fetches current load from each
   - Checks cache contents
   - Calculates heuristic scores
        ↓
4. Routes to lowest-score server
        ↓
5. Backend processes request
   - Cache hit: 50ms
   - Cache miss: 200ms, then cache it
        ↓
6. Response returns to client
        ↓
7. Metrics logged to PostgreSQL

(Meanwhile, every 60 seconds)
8. Prophet generates new predictions
        ↓
9. Q-learning observes state
        ↓
10. Decides: scale up / down / hold
```

---

## Why This Design?

| Component | Problem Solved | AI Technique |
|-----------|----------------|--------------|
| Prophet | When will traffic spike? | Time-series forecasting |
| A* | Which server is best right now? | Heuristic search |
| Q-learning | When should we scale? | Reinforcement learning |

Each component handles a different decision:
- Prophet = **prediction** (what will happen)
- A* = **routing** (where to send this request)
- Q-learning = **capacity** (how many servers to run)

Together, they create a system that anticipates demand, routes intelligently, and scales efficiently.

## Author

Mohammed - Carleton University