import os
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager # new release of fastapi
from pydantic import BaseModel
import httpx
import yaml

# Making predictions ahead of time 
from prediction_service import prediction_service
from database import init_database, close_database, log_request

# A* implementation
from routing import router

# Scaling
from scaling import scaler, RawMetrics

app = FastAPI()

# Configuration
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config/servers.yaml")

# Global state
class LoadBalancerState:
    def __init__(self):
        self.servers: List[Dict[str, str]] = [] #  List of backend servers from config
        self.current_index = 0 # round robin distrubition 
        self.total_requests = 0
        self.server_request_counts: Dict[str, int] = {} # How many requests each server has handled
        self.start_time = datetime.now()
    
    #  Round-robin logic for distrubted systems 
    def get_next_server(self) -> Dict[str, str]:
        if not self.servers:
            raise HTTPException(status_code=503, detail="No backend servers available")
        
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
    
    def record_request(self, server_id: str):
        self.total_requests += 1
        if server_id not in self.server_request_counts:
            self.server_request_counts[server_id] = 0
        self.server_request_counts[server_id] += 1
    
    def get_uptime_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

state = LoadBalancerState()

# Response models
class StatsResponse(BaseModel):
    total_requests: int
    uptime_seconds: float
    num_servers: int
    requests_per_server: Dict[str, int]
    routing_algorithm: str

class RouteResponse(BaseModel):
    load_balancer_id: str
    routed_to: str
    backend_response: Dict
    total_time_ms: float
    timestamp: str

class PredictionResponse(BaseModel):
    timestamp: str
    predicted_1min: float
    predicted_3min: float
    predicted_5min: float
    lower_bound_5min: float
    upper_bound_5min: float
    uncertainty: float
    model_loaded: bool

# Load server configuration
# Reads from servers.yaml
def load_server_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            state.servers = config.get('servers', [])
            for server in state.servers:
                state.server_request_counts[server['id']] = 0
            print(f"Loaded {len(state.servers)} servers from config")

    # Should not happen, but just in case 
    except FileNotFoundError:
        print(f"Config file not found: {CONFIG_PATH}")
        state.servers = []
    except Exception as e:
        print(f"Error loading config: {e}")
        state.servers = []

# fastAPI
# Runs when fast api starts (before any requests are sent)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs once when app starts
    load_server_config()
    await init_database()

    for server in state.servers:
        router.register_server(server['id'], server['url'])
    
    # load the model
    prediction_service.load_model()
    await prediction_service.start()

    # Start scaling loop
    async def scaling_loop():
        await asyncio.sleep(30)  # Wait for system to stabilize
        while True:
            try:
                decision = await run_scaling_decision()
                print(f"Scaling decision: {decision['action']} (ε={decision['epsilon']})")
            except Exception as e:
                print(f"Scaling error: {e}")
            await asyncio.sleep(60)  # Run every 60 seconds
    
    scaling_task = asyncio.create_task(scaling_loop())

    print(f"Load balancer started with {len(state.servers)} backend servers")
    yield

    scaling_task.cancel()
    await prediction_service.stop()
    await close_database()
    print("Load Balancer shutdown complete")

app = FastAPI(lifespan=lifespan)
# System statisics
# shows distribution across servers
# shows routing algo (round robin to start)
@app.get("/stats")
async def get_stats():
    return StatsResponse(
        total_requests=state.total_requests,
        uptime_seconds=state.get_uptime_seconds(),
        num_servers=len(state.servers),
        requests_per_server=state.server_request_counts,
        routing_algorithm="a-star" # deprecated round robin to use A*
    )

# Debugging purposes
@app.get("/routing/debug/{path:path}")
async def debug_routing(path: str):
    return {
        "request_path": path,
        "server_scores": router.get_routing_decision_info(path)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "num_servers": len(state.servers),
        "uptime_seconds": state.get_uptime_seconds()
    }


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

# Q learning endpoints 
@app.get("/scaling/status")
async def get_scaling_status():
    return {
        "current_servers": scaler.current_servers,
        "epsilon": round(scaler.epsilon, 3),
        "last_scale_time": scaler.last_scale_time.isoformat() if scaler.last_scale_time else None,
        "recent_decisions": scaler.decision_history[-5:]
    }


@app.get("/scaling/qtable")
async def get_q_table():
    return scaler.get_q_table_summary()

async def run_scaling_decision():
    #Collect metrics and make scaling decision
    from routing import router
    
    # Get prediction from Prophet
    pred = prediction_service.get_current_prediction()
    predicted_load = pred.predicted_5min if pred else 0
    uncertainty = pred.uncertainty if pred else 0
    
    # Get current load from router state
    total_active = 0
    for s in router.servers.values():
        total_active += s.active_requests

    num_servers = len(router.servers)
    avg_load = (total_active / num_servers * 100) if num_servers > 0 else 0
    
    # Get latency from recent requests (simplified - use avg response time)
    if num_servers > 0:
        total_latency = 0
        for s in router.servers.values():
            total_latency += s.avg_response_time_ms
        avg_latency = total_latency / num_servers
    else:
        avg_latency = 0
    
    metrics = RawMetrics(
        current_servers=scaler.current_servers,
        avg_load_percent=avg_load,
        predicted_load_5min=predicted_load,
        current_load=total_active,
        p95_latency_ms=avg_latency,
        prediction_uncertainty=uncertainty
    )
    
    decision = scaler.make_decision(metrics)
    return decision

# Main routing logic 
# Round robin logic (deprecated to use A* instead)
'''@app.get("/{path:path}")
async def route_request(path: str):
    start_time = time.time()

    # Pick next backend server 
    server = state.get_next_server()
    state.record_request(server['id'])
    
    backend_url = f"{server['url']}/{path}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Make HTTP  request and wait for a resposne 
        try:
            response = await client.get(backend_url)
            backend_data = response.json()
            
            total_time = time.time() - start_time

            # Log to database
            await log_request(
                server_id=server['id'],
                path=path,
                response_time_ms=total_time * 1000,
                status_code=response.status_code,
                cache_hit=backend_data.get('cache_hit', False),
                bytes_sent=backend_data.get('bytes', 0)
            )
            
            # Send back to client 
            return RouteResponse(
                load_balancer_id="load_balancer_main",
                routed_to=server['id'],
                backend_response=backend_data,
                total_time_ms=total_time * 1000,
                timestamp=datetime.now().isoformat()
            )
        
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Backend server {server['id']} unavailable: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error routing to {server['id']}: {str(e)}"
            )
'''
# A* routing 
@app.get("/{path:path}")
async def route_request(path: str):
    start_time = time.time()

    server_state = await router.select_server(path)
    if server_state is None:
        raise HTTPException(status_code=503, detail="No backend servers available")
    
    server_id = server_state.server_id
    server_url = server_state.url
    
    state.record_request(server_id)
    router.record_request_start(server_id)
    
    backend_url = f"{server_url}/{path}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(backend_url)
            backend_data = response.json()
            
            total_time = time.time() - start_time
            
            router.record_request_end(server_id, total_time * 1000, path)

            await log_request(
                server_id=server_id,
                path=path,
                response_time_ms=total_time * 1000,
                status_code=response.status_code,
                cache_hit=backend_data.get('cached', False),
                bytes_sent=backend_data.get('bytes', 0)
            )
            
            return RouteResponse(
                load_balancer_id="load_balancer_main",
                routed_to=server_id,
                backend_response=backend_data,
                total_time_ms=total_time * 1000,
                timestamp=datetime.now().isoformat()
            )
        
        except httpx.RequestError as e:
            router.record_request_end(server_id, 0, path)
            raise HTTPException(
                status_code=503,
                detail=f"Backend server {server_id} unavailable: {str(e)}"
            )
        except Exception as e:
            router.record_request_end(server_id, 0, path)
            raise HTTPException(
                status_code=500,
                detail=f"Error routing to {server_id}: {str(e)}"
            )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)