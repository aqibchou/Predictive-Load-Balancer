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
    
    # load the model
    prediction_service.load_model()
    await prediction_service.start()

    print(f"Load balancer started with {len(state.servers)} backend servers")
    yield
    await prediction_service.stop()
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
        routing_algorithm="round-robin"
    )

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

# Main routing logic 
@app.get("/{path:path}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)