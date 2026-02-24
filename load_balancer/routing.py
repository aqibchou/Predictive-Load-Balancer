import asyncio
import httpx
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

WEIGHT_ACTIVE_REQUESTS = 10.0
WEIGHT_RESPONSE_TIME   = 0.5
WEIGHT_HEALTH_PENALTY  = 1000.0
CACHE_HIT_BONUS        = 20.0
CIRCUIT_TIMEOUT_SECONDS = 30


class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


@dataclass
class ServerState:
    server_id:            str
    url:                  str
    is_healthy:           bool           = True
    active_requests:      int            = 0
    avg_response_time_ms: float          = 0.0
    cached_paths:         Set[str]       = field(default_factory=set)
    last_updated:         datetime       = field(default_factory=datetime.now)
    consecutive_failures: int            = 0
    fetched:              bool           = False
    circuit_state:        CircuitState   = field(default=CircuitState.CLOSED)
    circuit_open_at:      Optional[datetime] = field(default=None)


class AStarRouter:
    def __init__(self):
        self.servers: Dict[str, ServerState] = {}
        self.state_ttl_seconds = 5
        self._update_lock = asyncio.Lock()
        self._redis = None
        # In-memory mirror of Redis — refreshed every 100 ms by _sync_loop.
        # select_server() reads from here (O(1) dict access, no network).
        self._local_state: Dict[str, dict] = {}
        self._state_poll_task: Optional[asyncio.Task] = None

    def register_server(self, server_id: str, url: str):
        self.servers[server_id] = ServerState(server_id=server_id, url=url)

    async def connect_redis(self, redis_url: str):
        try:
            import aioredis
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            await self._redis.ping()
            print(f"Connected to Redis at {redis_url}")
        except Exception as e:
            print(f"Redis unavailable ({e}), using in-memory state")
            self._redis = None

    async def start_background_sync(self):
        """Start the 100 ms background task that mirrors Redis → _local_state."""
        self._state_poll_task = asyncio.create_task(
            self._sync_loop(), name="redis-state-sync"
        )

    async def stop_background_sync(self):
        if self._state_poll_task and not self._state_poll_task.done():
            self._state_poll_task.cancel()
            try:
                await self._state_poll_task
            except asyncio.CancelledError:
                pass

    async def _sync_loop(self):
        """Background task — runs every 100 ms.

        1. Pull latest state from Redis → _local_state.
        2. Apply _local_state to in-memory ServerState objects so
           select_server() never touches Redis or makes HTTP calls.
        3. If any server is stale (gRPC absent / Redis cold), trigger
           update_all_servers() here in the background, not on the hot path.
        """
        while True:
            await asyncio.sleep(0.1)

            # Step 1: Redis → _local_state
            if self._redis:
                for sid in list(self.servers):
                    try:
                        data = await self._redis.hgetall(f"server_state:{sid}")
                        if data:
                            self._local_state[sid] = {
                                "active_requests":      int(data.get("active_requests", 0)),
                                "avg_response_time_ms": float(data.get("avg_response_time_ms", 0.0)),
                                "is_healthy":           bool(int(data.get("is_healthy", 1))),
                                "consecutive_failures": int(data.get("consecutive_failures", 0)),
                                "cached_paths":         (
                                    set(data["cached_paths"].split(","))
                                    if data.get("cached_paths") else set()
                                ),
                                "updated_at":           float(data.get("updated_at", 0.0)),
                            }
                    except Exception:
                        pass

            # Step 2: _local_state → ServerState objects
            now_ts = datetime.now().timestamp()
            for sid, server in self.servers.items():
                local = self._local_state.get(sid)
                if local and now_ts - local["updated_at"] < 30:
                    server.active_requests      = local["active_requests"]
                    server.avg_response_time_ms = local["avg_response_time_ms"]
                    server.is_healthy           = local["is_healthy"]
                    server.consecutive_failures = local["consecutive_failures"]
                    server.cached_paths         = local["cached_paths"]
                    server.fetched              = True
                    # Only update last_updated from Redis if Redis is newer than
                    # gRPC (which updates last_updated directly on the object).
                    redis_ts = datetime.fromtimestamp(local["updated_at"])
                    if redis_ts > server.last_updated:
                        server.last_updated = redis_ts

            # Step 3: HTTP fallback poll if gRPC is absent or Redis is cold
            if any(self.is_state_stale(s) or not s.fetched for s in self.servers.values()):
                await self.update_all_servers()

    def calculate_heuristic(self, server: ServerState, request_path: str) -> float:
        score = server.active_requests * WEIGHT_ACTIVE_REQUESTS
        if not server.is_healthy:
            score += WEIGHT_HEALTH_PENALTY
        if server.circuit_state == CircuitState.OPEN:
            score += WEIGHT_HEALTH_PENALTY
        elif server.circuit_state == CircuitState.HALF_OPEN:
            score += 500
        score += server.avg_response_time_ms * WEIGHT_RESPONSE_TIME
        if request_path in server.cached_paths:
            score -= CACHE_HIT_BONUS
        return score

    async def update_server_state(self, server_id: str):
        if server_id not in self.servers:
            return
        server = self.servers[server_id]
        async with httpx.AsyncClient(timeout=2.0) as client:
            try:
                metrics = (await client.get(f"{server.url}/metrics")).json()
                server.active_requests      = metrics.get('active_requests', 0)
                server.avg_response_time_ms = metrics.get('avg_response_time_ms', 0)
                server.is_healthy           = True
                server.consecutive_failures = 0

                if server.circuit_state == CircuitState.HALF_OPEN:
                    server.circuit_state  = CircuitState.CLOSED
                    server.circuit_open_at = None
                    print(f"{server_id} circuit: HALF_OPEN → CLOSED")

                cache_data          = (await client.get(f"{server.url}/cache")).json()
                server.cached_paths = set(cache_data.get('cached_paths', []))
                server.fetched      = True
                server.last_updated = datetime.now()

                if self._redis:
                    try:
                        key = f"server_state:{server_id}"
                        await self._redis.hset(key, mapping={
                            'active_requests':      server.active_requests,
                            'avg_response_time_ms': server.avg_response_time_ms,
                            'is_healthy':           int(server.is_healthy),
                            'consecutive_failures': server.consecutive_failures,
                            'cached_paths':         ','.join(server.cached_paths),
                            'updated_at':           datetime.now().timestamp(),
                        })
                        await self._redis.expire(key, 30)
                    except Exception:
                        pass

            except Exception:
                server.consecutive_failures += 1
                if server.circuit_state == CircuitState.HALF_OPEN:
                    server.circuit_state   = CircuitState.OPEN
                    server.circuit_open_at = datetime.now()
                    print(f"{server_id} circuit: HALF_OPEN → OPEN (probe failed)")
                elif server.consecutive_failures >= 3 and server.circuit_state == CircuitState.CLOSED:
                    server.circuit_state   = CircuitState.OPEN
                    server.circuit_open_at = datetime.now()
                    server.is_healthy      = False
                    print(f"{server_id} circuit: CLOSED → OPEN ({server.consecutive_failures} failures)")

    async def update_all_servers(self):
        # Double-checked locking: skip refresh if another coroutine already updated
        async with self._update_lock:
            if not any(self.is_state_stale(s) or not s.fetched for s in self.servers.values()):
                return
            await asyncio.gather(*[self.update_server_state(sid) for sid in self.servers], return_exceptions=True)

    def is_state_stale(self, server: ServerState) -> bool:
        return (datetime.now() - server.last_updated).total_seconds() > self.state_ttl_seconds

    def _maybe_try_half_open(self, server: ServerState):
        if server.circuit_state == CircuitState.OPEN and server.circuit_open_at:
            if (datetime.now() - server.circuit_open_at).total_seconds() > CIRCUIT_TIMEOUT_SECONDS:
                server.circuit_state = CircuitState.HALF_OPEN
                print(f"{server.server_id} circuit: OPEN → HALF_OPEN")

    async def select_server(self, request_path: str) -> Optional[ServerState]:
        """Pure in-memory scoring — zero I/O.

        All Redis reads and HTTP polls are handled by _sync_loop in the
        background. This function only reads from already-populated ServerState
        objects and returns immediately.
        """
        if not self.servers:
            return None

        for server in self.servers.values():
            self._maybe_try_half_open(server)

        scored = sorted(
            ((self.calculate_heuristic(s, request_path), s) for s in self.servers.values()),
            key=lambda x: x[0]
        )
        return scored[0][1]

    async def record_request_start(self, server_id: str):
        if server_id in self.servers:
            self.servers[server_id].active_requests += 1
        if self._redis:
            try:
                await self._redis.hincrby(f"server_state:{server_id}", 'active_requests', 1)
            except Exception:
                pass

    async def record_request_end(self, server_id: str, response_time_ms: float, path: str):
        if server_id in self.servers:
            self.servers[server_id].active_requests = max(0, self.servers[server_id].active_requests - 1)
        if self._redis:
            try:
                await self._redis.hincrby(f"server_state:{server_id}", 'active_requests', -1)
            except Exception:
                pass

    def update_from_grpc(self, server_id: str, data: dict):
        """Apply a telemetry payload received over gRPC.

        Directly writes live metrics into ServerState and marks the server
        healthy, resetting the circuit breaker failure counter.  The
        ``fetched`` flag is set to True so the HTTP polling fallback skips
        this server while its gRPC stream is active.
        """
        if server_id not in self.servers:
            return
        s = self.servers[server_id]
        s.active_requests      = data.get("active_requests",      s.active_requests)
        s.avg_response_time_ms = data.get("avg_response_time_ms", s.avg_response_time_ms)
        s.is_healthy           = True
        s.consecutive_failures = 0
        s.last_updated         = datetime.now()
        s.fetched              = True   # suppresses HTTP fallback polling

        if s.circuit_state == CircuitState.HALF_OPEN:
            s.circuit_state  = CircuitState.CLOSED
            s.circuit_open_at = None
            print(f"{server_id} circuit: HALF_OPEN → CLOSED (gRPC heartbeat)")

    def get_routing_decision_info(self, request_path: str) -> List[Dict]:
        return sorted([{
            "server_id":            s.server_id,
            "score":                round(self.calculate_heuristic(s, request_path), 2),
            "active_requests":      s.active_requests,
            "avg_response_time_ms": round(s.avg_response_time_ms, 2),
            "cache_hit":            request_path in s.cached_paths,
            "is_healthy":           s.is_healthy,
            "circuit_state":        s.circuit_state.value,
        } for s in self.servers.values()], key=lambda x: x['score'])


router = AStarRouter()
