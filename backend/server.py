import os
import time
import asyncio
from datetime import datetime
from collections import OrderedDict
from typing import Optional
from fastapi import FastAPI, Response
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Server configuration
SERVER_ID = os.getenv("SERVER_ID", "server_unknown")
CACHE_MAX_SIZE = 100
CACHE_HIT_LATENCY = 0.05  # 50ms
CACHE_MISS_LATENCY = 0.20  # 200ms

# Metrics storage
class ServerMetrics:
    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.active_requests = 0
        self.response_times = []
        self.start_time = datetime.now()
    
    def record_request(self, response_time: float, cache_hit: bool):
        self.total_requests += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        self.response_times.append(response_time)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    def get_cache_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    def get_avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_uptime_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

# LRU Cache implementation
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> bool:
        if key in self.cache:
            self.cache.move_to_end(key)
            return True
        return False
    
    def put(self, key: str):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = True
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def size(self) -> int:
        return len(self.cache)
    
    def contains(self, key: str) -> bool:
        return key in self.cache

# Global state
metrics = ServerMetrics()
cache = LRUCache(CACHE_MAX_SIZE)

# Response models
class HealthResponse(BaseModel):
    status: str
    server_id: str
    active_requests: int
    load_percentage: float
    uptime_seconds: float

# might need to remove
class MetricsResponse(BaseModel):
    server_id: str
    total_requests: int
    active_requests: int
    cache_size: int
    cache_hit_rate: float
    avg_response_time_ms: float
    uptime_seconds: float

class RequestResponse(BaseModel):
    server_id: str
    path: str
    cached: bool
    response_time_ms: float
    timestamp: str

@app.get("/health")
async def health_check():
    load_percentage = min(100, (metrics.active_requests / 10) * 100)
    
    return HealthResponse(
        status="healthy",
        server_id=SERVER_ID,
        active_requests=metrics.active_requests,
        load_percentage=load_percentage,
        uptime_seconds=metrics.get_uptime_seconds()
    )

@app.get("/metrics")
async def get_metrics():
    return MetricsResponse(
        server_id=SERVER_ID,
        total_requests=metrics.total_requests,
        active_requests=metrics.active_requests,
        cache_size=cache.size(),
        cache_hit_rate=metrics.get_cache_hit_rate(),
        avg_response_time_ms=metrics.get_avg_response_time() * 1000,
        uptime_seconds=metrics.get_uptime_seconds()
    )

@app.get("/{path:path}")
async def handle_request(path: str):
    start_time = time.time()
    metrics.active_requests += 1
    
    try:
        is_cached = cache.get(path)
        
        if is_cached:
            await asyncio.sleep(CACHE_HIT_LATENCY)
        else:
            await asyncio.sleep(CACHE_MISS_LATENCY)
            cache.put(path)
        
        response_time = time.time() - start_time
        metrics.record_request(response_time, is_cached)
        
        return RequestResponse(
            server_id=SERVER_ID,
            path=path,
            cached=is_cached,
            response_time_ms=response_time * 1000,
            timestamp=datetime.now().isoformat()
        )
    
    finally:
        metrics.active_requests -= 1

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)