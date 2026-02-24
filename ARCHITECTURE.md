# Architecture & UML Diagrams

## Full System Pipeline

```mermaid
graph TD
    subgraph SLOW["Slow Path — OOA-DPSO-GA (CronJob / async_worker.py)"]
        A1[Raw Traffic CSV] --> A2[Feature Engineering\n38 columns]
        A2 --> A3[OOA-DPSO-GA Selector\nn_particles=30, n_iterations=50]
        A3 -->|LHS init + PSO velocity\n+ GA mutation & crossover| A4[Binary Mask\n11 of 38 features]
        A4 --> A5[(Redis\nfeature_mask key)]
        A4 --> A6[JSON Fallback\nmodels/feature_mask.json]
    end

    subgraph LB["Load Balancer — FastAPI on port 8000"]
        subgraph PRED["Prediction Service (prediction_service.py)"]
            B1[PostgreSQL\ntraffic_aggregates\n120 min window] --> B2[Feature Engineering\n38 columns]
            B2 --> B3[Apply Mask\nX cols mask → 11 features]
            A5 -->|load_mask O1| B3
            A6 -.->|fallback| B3
            B3 --> B4[BiStacking Ensemble\nasyncio.to_thread]
            subgraph BS["BiStacking — bistacking.py"]
                B4 --> C1[Level 0 — OOF Folds]
                C1 --> C2[LightGBM]
                C1 --> C3[Ridge α=10]
                C1 --> C4[ExtraTrees\n200 trees]
                C1 --> C5[ElasticNet]
                C2 & C3 & C4 & C5 --> C6[OOF Stack\nn_rows × 4]
                C6 --> C7[Level 1 Meta\nLightGBM]
            end
            C7 --> B5[pred_1min\npred_3min\npred_5min]
            C2 & C3 & C4 & C5 --> B6[Uncertainty\nstd of base learners]
        end

        subgraph ROUTE["A* Router (routing.py)"]
            D1[Incoming Request] --> D2{Circuit\nBreaker}
            D2 -->|OPEN| D3[Skip — score +1000]
            D2 -->|HALF_OPEN| D4[Probe — score +500]
            D2 -->|CLOSED| D5[Score Heuristic]
            D5 --> D6["score = active_req×10\n+ resp_time×0.5\n+ 1000 if unhealthy\n- 20 if cached"]
            D6 --> D7[Select Lowest Score]
            D7 --> D8{State\nFresh?}
            D8 -->|Redis < 5s old| D9[Skip HTTP refresh]
            D8 -->|Stale| D10[update_all_servers\ndouble-checked lock]
            D10 --> D11[HTTP /health\nper backend]
            D11 -->|Write| D12[(Redis\nserver_state:id\nTTL 30s)]
        end

        subgraph SCALE["Q-Learning Scaler (scaling.py)"]
            E1[Run every 60s] --> E2[Build RawMetrics\nfrom router + pred]
            E2 --> E3[get_state]
            E3 --> E4["SystemState (5-dim)\n servers count\n load LOW/MED/HIGH\n trend DEC/STAB/INC\n latency OK/WARN/CRIT\n uncertainty LOW/MED/HIGH"]
            B6 -->|prediction_uncertainty| E4
            E4 --> E5{ε-greedy\nε→0.05}
            E5 -->|explore| E6[Random Action]
            E5 -->|exploit| E7[argmax Q-table]
            E6 & E7 --> E8{Action}
            E8 -->|scale_up / scale_down| E9[execute_scaling_action\nasyncio.to_thread + 3 retries]
            E9 --> E10["docker-compose\n-p predictive-load-balancer\n--no-deps backend\n--scale backend=N"]
            E8 -->|hold| E11[No-op]
            E9 --> E12[calculate_reward]
            B5 -->|prev predicted load| E12
            E12 --> E13["reward:\n-10 latency CRIT\n-5 latency WARN\n-0.5 per server\n+5 SLA bonus\n-1 action penalty\n+2 pred accuracy"]
            E13 --> E14[update Q-table\nBellman equation]
            E14 --> E15[(PostgreSQL\nqtable_state\nupsert)]
            E14 --> E16[JSON checkpoint\nfallback]
        end
    end

    subgraph BACK["Backend Servers (backend/server.py)"]
        F1[backend-1\nLRU Cache]
        F2[backend-2\nLRU Cache]
        F3[backend-N\nLRU Cache]
    end

    subgraph OBS["Observability"]
        G1[Prometheus\nport 9090]
        G2[Grafana\nport 3000]
    end

    D7 --> F1 & F2 & F3
    F1 & F2 & F3 -->|log_request| H[(PostgreSQL\nrequest_metrics)]
    H -->|aggregate every 60s| B1
    LB -->|scrape /metrics| G1
    G1 --> G2

    style SLOW fill:#f0f4ff,stroke:#6b7280
    style LB fill:#fef9f0,stroke:#6b7280
    style BS fill:#f0fff4,stroke:#6b7280
    style BACK fill:#fff0f0,stroke:#6b7280
    style OBS fill:#f5f0ff,stroke:#6b7280
```

---

## Request Lifecycle Sequence

```mermaid
sequenceDiagram
    participant Client
    participant LB as Load Balancer
    participant Redis
    participant A* as A* Router
    participant CB as Circuit Breaker
    participant Backend
    participant DB as PostgreSQL

    Client->>LB: GET /data?key=foo
    LB->>A*: select_server("/data")
    A*->>Redis: HGETALL server_state:{id}
    Redis-->>A*: {active_requests, resp_time, ...} age=2s
    Note over A*: age < 5s → skip HTTP refresh
    A*->>CB: check circuit_state
    CB-->>A*: CLOSED → no penalty
    A*->>A*: score = active×10 + resp×0.5 - 20 (cached)
    A*-->>LB: backend-1 (lowest score)
    LB->>Backend: proxy GET /data?key=foo
    Backend-->>LB: 200 OK (cache hit, 50ms)
    LB->>Redis: HINCRBY active_requests -1
    LB->>DB: log_request(server_id, path, 50ms, 200)
    LB-->>Client: 200 OK

    Note over LB: Every 60 seconds...
    LB->>DB: get_recent_traffic(120 min)
    DB-->>LB: 120 rows
    LB->>LB: engineer_features → 38 cols
    LB->>LB: apply_mask → 11 cols
    LB->>LB: asyncio.to_thread(BiStacking.fit)
    LB->>LB: predict → (41, 101, 161) req/min
    LB->>LB: uncertainty = std(base learners)

    Note over LB: Every 60 seconds (offset)...
    LB->>LB: build SystemState(5-dim)
    LB->>LB: ε-greedy → action
    alt action = scale_up
        LB->>LB: asyncio.to_thread(docker-compose --scale backend=N)
        LB->>Redis: update server_state for new backend
    end
    LB->>DB: upsert qtable_state
```

---

## Circuit Breaker State Machine

```mermaid
stateDiagram-v2
    [*] --> CLOSED

    CLOSED --> CLOSED : request succeeds
    CLOSED --> OPEN : consecutive_failures ≥ 3
    note right of OPEN : score += 1000\n(effectively removed\nfrom routing)

    OPEN --> HALF_OPEN : 30 seconds elapsed
    note right of HALF_OPEN : score += 500\n(limited probe traffic)

    HALF_OPEN --> CLOSED : probe request succeeds
    HALF_OPEN --> OPEN : probe request fails
```

---

## BiStacking Training Flow

```mermaid
flowchart LR
    subgraph INPUT["Input (59 rows × 11 features)"]
        I1[X_train\nmasked features]
        I2[y_train\nrequest_count shifted +1]
    end

    subgraph OOF["OOF Phase — 5 Expanding Folds"]
        F1["Fold 1\ntrain on rows 0-11\npredict rows 12-23"]
        F2["Fold 2\ntrain on rows 0-23\npredict rows 24-35"]
        F3["Fold 3\ntrain on rows 0-35\npredict rows 36-47"]
        F4["Fold 4\ntrain on rows 0-47\npredict rows 48-58"]
        F5["Fold 5\ntrain on all\n(used for final fit)"]

        F1 & F2 & F3 & F4 --> STACK["OOF Stack\n59 × 4 matrix\n(LGB | Ridge | ET | EN)"]
    end

    subgraph L1["Level 1"]
        STACK --> META["LightGBM\nMeta-learner\nfit on OOF stack"]
    end

    subgraph FULLFIT["Final Full Fit (all 59 rows)"]
        LGB["LightGBM"] & RDG["Ridge"] & ET["ExtraTrees"] & EN["ElasticNet"]
    end

    subgraph INFER["Inference (last row = current minute)"]
        X_LAST["X_last (1 × 11)"]
        X_LAST --> LGB & RDG & ET & EN
        LGB & RDG & ET & EN --> STACK2["[lgb, ridge, et, en]"]
        STACK2 --> META2["Meta-learner\n.predict()"]
        META2 --> PRED["pred_1min"]
        LGB & RDG & ET & EN --> STD["std() → uncertainty"]
    end

    INPUT --> OOF
    INPUT --> FULLFIT
    FULLFIT --> INFER
```

---

## Q-Learning State Space

```mermaid
graph LR
    S["SystemState\n(5 dimensions)"] --> D1["servers\n1 | 2 | 3 | 4 | 5"]
    S --> D2["load_level\nlow | medium | high"]
    S --> D3["prediction_trend\ndecreasing | stable | increasing"]
    S --> D4["latency_status\nok | warning | critical"]
    S --> D5["uncertainty_level ← NEW\nlow | medium | high"]

    D1 & D2 & D3 & D4 & D5 --> TOTAL["5 × 3 × 3 × 3 × 3\n= 405 states\n(was 135)"]

    TOTAL --> QTABLE["Q-Table\n405 states × 3 actions\n= 1215 Q-values"]
    QTABLE --> ACTIONS["scale_up\nscale_down\nhold"]
```
