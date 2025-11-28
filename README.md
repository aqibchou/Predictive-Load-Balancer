# Predictive Load Balancer (initial deisgn - POC)

An AI-driven load balancing system that uses machine learning to predict traffic patterns and proactively scale resources before demand spikes occur.

## Overview

Traditional load balancers are **reactive** - they respond to problems after they happen. This system is **proactive** - it predicts traffic spikes using Facebook Prophet and scales resources in advance, while using A* search for intelligent request routing and Q-learning for optimal scaling decisions.

**Built for:** COMP 3106 (Artificial Intelligence) - Carleton University

## Key Features

- **Traffic Prediction**: Facebook Prophet forecasts traffic 5 minutes ahead using historical patterns
- **Intelligent Routing**: A* algorithm routes requests considering server load, cache locality, and health
- **Adaptive Scaling**: Q-learning agent learns optimal scaling policies balancing cost and performance
- **Real-time Dashboard**: Live visualization of predictions, server status, and system metrics

## Architecture

```
Synthetic Traffic → Load Balancer → Backend Server Pool → Metrics Database
                         ↓
                    [Prophet] → Predicts traffic
                    [A* Router] → Routes requests  
                    [Q-Learning] → Scales servers
```

## Project Structure

```
predictive-load-balancer/
├── data/                      # Processed NASA HTTP logs
├── scripts/                   # ML pipeline (extract, clean, train)
├── models/                    # Trained models (Prophet, Ridge)
├── backend/                   # FastAPI backend servers
├── load_balancer/             # Main load balancer service
├── traffic_generator/         # Synthetic traffic generation
├── experiments/               # Performance comparison tests
└── results/                   # Metrics and visualizations
```

## ML Pipeline

### Phase 1: Data Processing

**Input:** NASA HTTP access logs (Aug 1995, ~2M requests)

**Steps:**
1. **Extract** - Parse raw Apache logs into structured CSV
2. **Clean** - Remove duplicates, outliers, invalid entries
3. **Feature Engineering** - Create time-based, rolling, and lag features
4. **Train** - Build Prophet, Ridge, and Linear models
5. **Validate** - Evaluate using MAE, RMSE, R²

**Output:** Trained models ready for deployment

### Phase 2: Model Training

**Models:**
- **Facebook Prophet** (Primary) - Time-series forecasting with seasonality
- **Ridge Regression** - Linear model with L2 regularization
- **Linear Regression** - Simple baseline

**Evaluation Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

**Train/Test Split:** 80/20 time-ordered (no shuffling)

## System Components

### Load Balancer

**Core Services:**
- **Prediction Service**: Runs Prophet every minute, predicts next 5 minutes
- **A* Router**: Multi-factor heuristic considers load, cache, health, latency
- **Q-Learning Scaler**: Reinforcement learning for scaling decisions
- **Metrics Collector**: Logs all requests and decisions to PostgreSQL

**Technologies:** FastAPI, Docker, PostgreSQL

### Backend Servers

**Features:**
- In-memory LRU cache
- Realistic processing simulation (50ms cache hit, 200ms miss)
- Health monitoring
- Load tracking

**Technologies:** FastAPI, Docker

### Traffic Generator

**Purpose:** Generate synthetic traffic patterns for testing

**Patterns:**
- Normal daily cycle (low at night, high during day)
- Sudden traffic spikes
- Gradual load increases

**Note:** Uses synthetic data, NOT replaying training logs

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- PostgreSQL 14+

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/predictive-load-balancer.git
cd predictive-load-balancer

# Install Python dependencies
pip install -r requirements.txt
```

### ML Pipeline

```bash
# 1. Extract data
python scripts/01_extract_data.py

# 2. Clean data
python scripts/02_clean_data.py

# 3. Engineer features
python scripts/03_engineer_features.py

# 4. Train models
python scripts/04_train_models.py

# 5. Validate models
python scripts/05_validate_models.py
```

### Run System

```bash
# Start all services
docker-compose up

# In another terminal, generate traffic
python traffic_generator/synthetic_generator.py

# View dashboard
open http://localhost:3000
```

### Run Experiments

```bash
# Compare all algorithms
python experiments/run_experiments.py

# Analyze results
python experiments/analyze_results.py
```

## Experimental Results

**Algorithms Compared:**
1. Round-Robin + Reactive Scaling (Baseline)
2. Greedy (Least Connections) + Reactive Scaling
3. A* Routing + Reactive Scaling
4. **A* Routing + Q-Learning Scaling** (Full AI System)

**Expected Improvements:**
- ~15-20% lower latency vs baseline
- ~40-50% higher cache hit rate
- ~15% fewer servers needed (cost savings)
- Proactive scaling prevents latency spikes

## Key Insights

### Why This Approach Works

**Prophet learns from historical patterns:**
- Trained on NASA logs (Aug 1-25, 1995)
- Learns "2pm = high traffic, 3am = low traffic"
- Captures daily and weekly seasonality

**Testing on NEW data:**
- Synthetic traffic generator creates unseen patterns
- Tests Prophet's ability to generalize, not memorize
- Proves system works on real-world scenarios

**AI components work together:**
- Prophet: "Traffic spike coming in 5 minutes"
- Q-learning: "Scale up now to prepare"
- A*: "Route to servers with cached data"

## Technologies Used

**Machine Learning:**
- Facebook Prophet (time-series forecasting)
- scikit-learn (Ridge regression, metrics)
- NumPy, Pandas (data processing)

**Backend:**
- FastAPI (async web framework)
- PostgreSQL (metrics storage)
- Docker (containerization)

**Frontend (Optional):**
- React (dashboard)
- Chart.js (visualizations)
- WebSockets (real-time updates)

## Dataset

**Source:** NASA HTTP access logs (publicly available)

**Details:**
- Time period: August 1995
- ~2 million HTTP requests
- Contains real daily/weekly patterns
- Includes traffic spikes from shuttle launches

**Usage:**
- Training only (NOT used for testing)
- Test with synthetic generated traffic

## Future Enhancements

- [ ] Add more ML models (LSTM, ARIMA)
- [ ] Implement geographic routing
- [ ] Add circuit breaker patterns
- [ ] Support for Kubernetes deployment
- [ ] Advanced caching strategies
- [ ] Multi-region load balancing


## Author

Mohammed - COMP 3106 Final Project
