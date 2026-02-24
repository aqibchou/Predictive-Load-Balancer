import hashlib
import os
import time
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from collections import OrderedDict
from typing import Optional
from fastapi import FastAPI, Response
from pydantic import BaseModel
import uvicorn

# Server configuration
SERVER_ID = os.getenv("SERVER_ID", "server_unknown") # docker will set this up
CACHE_MAX_SIZE = 100
HASH_ROUNDS = 10_000  # ~20-40ms of real CPU on first call; <1ms from cache

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
    
    def get(self, key: str):
        """Return cached value, or None on miss."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: str):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def size(self) -> int:
        return len(self.cache)
    
    def contains(self, key: str) -> bool:
        return key in self.cache

# Global state
metrics = ServerMetrics()
cache = LRUCache(CACHE_MAX_SIZE)


def _compute_hash(path: str) -> str:
    """Chain SHA-256 HASH_ROUNDS times.

    First call: ~20-40ms of real CPU work.
    Subsequent calls: returned from LRUCache in <1ms — never reaches here.
    Runs in asyncio.to_thread() so the event loop is never blocked.
    """
    data = path.encode()
    for _ in range(HASH_ROUNDS):
        data = hashlib.sha256(data).digest()
    return data.hex()


def _run_iterations(n: int) -> float:
    """Sum of square roots, scaled by n.

    iterations=100  → 100 000 ops ≈ 10-20 ms
    iterations=500  → 500 000 ops ≈ 50-100 ms
    Runs in asyncio.to_thread() so the event loop is never blocked.
    """
    return sum(i ** 0.5 for i in range(n * 1_000))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start gRPC telemetry agent (best-effort: skip if grpcio/psutil unavailable)
    _agent = None
    try:
        from telemetry_agent import TelemetryAgent
        _agent = TelemetryAgent(metrics, cache)
        _agent.start()
    except Exception as exc:
        print(f"[backend] TelemetryAgent not started: {exc}")
    yield
    if _agent is not None:
        _agent.stop()


app = FastAPI(lifespan=lifespan)


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

# Cache state used for A* routing 
@app.get("/cache")
async def get_cache_contents():
    return {
        "server_id": SERVER_ID,
        "cached_paths": list(cache.cache.keys()),
        "cache_size": cache.size()
    }

@app.get("/compute")
async def handle_compute(iterations: int = 100):
    """Real CPU work proportional to iterations. Never cached."""
    start_time = time.time()
    metrics.active_requests += 1
    try:
        result = await asyncio.to_thread(_run_iterations, iterations)
        response_time = time.time() - start_time
        metrics.record_request(response_time, False)
        return {
            "server_id": SERVER_ID,
            "iterations": iterations,
            "result": result,
            "response_time_ms": response_time * 1000,
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        metrics.active_requests -= 1


@app.get("/{path:path}")
async def handle_request(path: str):
    """
    Cache hit  → O(1) dict lookup, <1 ms.
    Cache miss → _compute_hash in thread pool, ~20-40 ms, then stored.
    """
    start_time = time.time()
    metrics.active_requests += 1
    try:
        cached_value = cache.get(path)

        if cached_value is not None:
            is_cached = True
            result = cached_value
        else:
            result = await asyncio.to_thread(_compute_hash, path)
            cache.put(path, result)
            is_cached = False

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