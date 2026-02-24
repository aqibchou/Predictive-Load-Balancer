import os
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Response
from contextlib import asynccontextmanager
from pydantic import BaseModel
import httpx
import yaml

from prediction_service import prediction_service
from database import init_database, close_database, log_request
from routing import router
from scaling import scaler, RawMetrics
import grpc_server
from metrics import (
    requests_total, request_latency, active_servers,
    predicted_traffic, prediction_confidence_lower, prediction_confidence_upper,
    scaling_actions, qlearning_reward, qlearning_epsilon,
    astar_server_scores, cache_hits, cache_misses, get_metrics
)

CONFIG_PATH  = os.getenv("CONFIG_PATH", "/app/config/servers.yaml")
routing_mode = "astar"

# Shared persistent HTTP client — reuses connections across all requests.
# Created once in lifespan; never recreated per-request.
_http_client: Optional[httpx.AsyncClient] = None


class LoadBalancerState:
    def __init__(self):
        self.servers:              List[Dict[str, str]] = []
        self.current_index         = 0
        self.total_requests        = 0
        self.server_request_counts: Dict[str, int] = {}
        self.start_time            = datetime.now()

    def get_next_server(self) -> Dict[str, str]:
        if not self.servers:
            raise HTTPException(status_code=503, detail="No backend servers available")
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server

    def record_request(self, server_id: str):
        self.total_requests += 1
        self.server_request_counts.setdefault(server_id, 0)
        self.server_request_counts[server_id] += 1

    def get_uptime_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()


state = LoadBalancerState()


class StatsResponse(BaseModel):
    total_requests:      int
    uptime_seconds:      float
    num_servers:         int
    requests_per_server: Dict[str, int]
    routing_algorithm:   str

class RouteResponse(BaseModel):
    load_balancer_id: str
    routed_to:        str
    backend_response: Dict
    total_time_ms:    float
    timestamp:        str

class PredictionResponse(BaseModel):
    timestamp:        str
    predicted_1min:   float
    predicted_3min:   float
    predicted_5min:   float
    lower_bound_5min: float
    upper_bound_5min: float
    uncertainty:      float
    model_loaded:     bool


def load_server_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        state.servers = config.get('servers', [])
        for server in state.servers:
            state.server_request_counts[server['id']] = 0
        print(f"Loaded {len(state.servers)} servers from config")
    except Exception as e:
        print(f"Error loading config: {e}")
        state.servers = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_server_config()
    await init_database()

    for server in state.servers:
        router.register_server(server['id'], server['url'])

    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
    await router.connect_redis(REDIS_URL)
    await router.start_background_sync()
    await scaler.load_qtable()

    prediction_service.load_model()
    await prediction_service.start()
    active_servers.set(len(state.servers))

    await grpc_server.start(router)

    async def scaling_loop():
        await asyncio.sleep(30)
        while True:
            try:
                decision = await run_scaling_decision()
                print(f"Scaling: {decision['action']} (ε={decision['epsilon']})")
            except Exception as e:
                print(f"Scaling error: {e}")
            await asyncio.sleep(60)

    global _http_client
    _http_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=20),
    )

    scaling_task = asyncio.create_task(scaling_loop())
    print(f"Load balancer started with {len(state.servers)} backend servers")
    yield

    await _http_client.aclose()
    scaling_task.cancel()
    await router.stop_background_sync()
    await grpc_server.stop()
    await prediction_service.stop()
    await close_database()
    print("Load balancer shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.post("/scaling/save")
async def save_qtable():
    await scaler.save_qtable()
    return {"status": "saved", "epsilon": scaler.epsilon, "states": len(scaler.q_table)}

@app.post("/scaling/load")
async def load_qtable():
    await scaler.load_qtable()
    return {"status": "loaded", "epsilon": scaler.epsilon, "states": len(scaler.q_table)}

@app.post("/scaling/reset")
def reset_qtable():
    scaler.reset_qtable()
    return {"status": "reset", "epsilon": scaler.epsilon}

@app.get("/stats")
async def get_stats():
    return StatsResponse(
        total_requests=state.total_requests,
        uptime_seconds=state.get_uptime_seconds(),
        num_servers=len(state.servers),
        requests_per_server=state.server_request_counts,
        routing_algorithm=routing_mode
    )

@app.get("/routing/debug/{path:path}")
async def debug_routing(path: str):
    return {"request_path": path, "server_scores": router.get_routing_decision_info(path)}

@app.post("/routing/mode/{mode}")
async def set_routing_mode(mode: str):
    global routing_mode
    if mode not in ["round_robin", "greedy", "astar"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use: round_robin, greedy, astar")
    routing_mode = mode
    return {"routing_mode": routing_mode}

@app.get("/routing/mode")
async def get_routing_mode():
    return {"routing_mode": routing_mode}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "num_servers": len(state.servers), "uptime_seconds": state.get_uptime_seconds()}

@app.get("/prediction")
async def get_prediction():
    pred = prediction_service.get_current_prediction()
    if pred is None:
        return {"status": "no prediction available yet"}
    return PredictionResponse(**pred.to_dict())

@app.get("/prediction/history")
async def get_prediction_history():
    return {
        "count": len(prediction_service.prediction_history),
        "predictions": [p.to_dict() for p in prediction_service.prediction_history[-10:]]
    }

@app.get("/scaling/status")
async def get_scaling_status():
    return {
        "current_servers": scaler.current_servers,
        "epsilon":         round(scaler.epsilon, 3),
        "last_scale_time": scaler.last_scale_time.isoformat() if scaler.last_scale_time else None,
        "recent_decisions": scaler.decision_history[-5:]
    }

@app.get("/scaling/qtable")
async def get_q_table():
    return scaler.get_q_table_summary()

@app.get("/metrics")
async def metrics():
    return Response(content=get_metrics(), media_type="text/plain")


async def run_scaling_decision():
    pred           = prediction_service.get_current_prediction()
    predicted_load = pred.predicted_5min if pred else 0
    uncertainty    = pred.uncertainty    if pred else 0

    total_active = sum(s.active_requests for s in router.servers.values())
    num_servers  = len(router.servers)
    avg_load     = (total_active / num_servers * 100) if num_servers > 0 else 0
    avg_latency  = (sum(s.avg_response_time_ms for s in router.servers.values()) / num_servers
                    if num_servers > 0 else 0)

    metrics_data = RawMetrics(
        current_servers       = scaler.current_servers,
        avg_load_percent      = avg_load,
        predicted_load_5min   = predicted_load,
        current_load          = total_active,
        p95_latency_ms        = avg_latency,
        prediction_uncertainty= uncertainty,
    )
    decision = await scaler.make_decision(metrics_data, router=router)

    scaling_actions.labels(action=decision['action']).inc()
    qlearning_epsilon.set(scaler.epsilon)
    active_servers.set(scaler.current_servers)

    if pred:
        predicted_traffic.set(pred.predicted_5min)
        prediction_confidence_lower.set(pred.lower_bound_5min)
        prediction_confidence_upper.set(pred.upper_bound_5min)

    return decision


@app.get("/{path:path}")
async def route_request(path: str):
    start_time   = time.time()
    server_state = await router.select_server(path)
    if server_state is None:
        raise HTTPException(status_code=503, detail="No backend servers available")

    server_id  = server_state.server_id
    server_url = server_state.url
    state.record_request(server_id)
    await router.record_request_start(server_id)

    try:
        response     = await _http_client.get(f"{server_url}/{path}")
        backend_data = response.json()
        total_time   = time.time() - start_time

        await router.record_request_end(server_id, total_time * 1000, path)

        cached = backend_data.get('cached', False)
        requests_total.labels(backend_server=server_id, status="success").inc()
        request_latency.labels(route=path).observe(total_time)
        (cache_hits if cached else cache_misses).labels(server=server_id).inc()

        for info in router.get_routing_decision_info(path):
            astar_server_scores.labels(server=info['server_id']).set(info['score'])

        await log_request(
            server_id=server_id, path=path,
            response_time_ms=total_time * 1000,
            status_code=response.status_code,
            cache_hit=cached, bytes_sent=backend_data.get('bytes', 0)
        )

        return RouteResponse(
            load_balancer_id="load_balancer_main",
            routed_to=server_id,
            backend_response=backend_data,
            total_time_ms=total_time * 1000,
            timestamp=datetime.now().isoformat()
        )

    except httpx.RequestError as e:
        await router.record_request_end(server_id, 0, path)
        requests_total.labels(backend_server=server_id, status="error").inc()
        raise HTTPException(status_code=503, detail=f"Backend {server_id} unavailable: {e}")
    except Exception as e:
        await router.record_request_end(server_id, 0, path)
        requests_total.labels(backend_server=server_id, status="error").inc()
        raise HTTPException(status_code=500, detail=f"Error routing to {server_id}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
