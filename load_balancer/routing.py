import asyncio
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


# TODO: Future enhancement - make weights configurable via environment variables (hardcoded for now)
WEIGHT_ACTIVE_REQUESTS = 10.0
WEIGHT_RESPONSE_TIME = 0.5
WEIGHT_HEALTH_PENALTY = 1000.0
CACHE_HIT_BONUS = 50.0


@dataclass
class ServerState:
    server_id: str
    url: str
    is_healthy: bool = True
    active_requests: int = 0
    avg_response_time_ms: float = 0.0
    cached_paths: Set[str] = field(default_factory=set)
    last_updated: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0

# A* implemenation 
class AStarRouter:
    def __init__(self):
        self.servers: Dict[str, ServerState] = {}
        self.state_ttl_seconds = 5
        self._update_lock = asyncio.Lock()
    
    def register_server(self, server_id: str, url: str):
        self.servers[server_id] = ServerState(server_id=server_id, url=url)
    
    def calculate_heuristic(self, server: ServerState, request_path: str) -> float:
        score = 0.0
        
        score += server.active_requests * WEIGHT_ACTIVE_REQUESTS
        
        if not server.is_healthy:
            score += WEIGHT_HEALTH_PENALTY
        
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
                metrics_resp = await client.get(f"{server.url}/metrics")
                metrics = metrics_resp.json()
                
                server.active_requests = metrics.get('active_requests', 0)
                server.avg_response_time_ms = metrics.get('avg_response_time_ms', 0)
                server.is_healthy = True
                server.consecutive_failures = 0
                
                cache_resp = await client.get(f"{server.url}/cache")
                cache_data = cache_resp.json()
                server.cached_paths = set(cache_data.get('cached_paths', []))
                
                server.last_updated = datetime.now()
                
            except Exception as e:
                server.consecutive_failures += 1
                if server.consecutive_failures >= 3:
                    server.is_healthy = False
    
    async def update_all_servers(self):
        async with self._update_lock:
            tasks = [self.update_server_state(sid) for sid in self.servers]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def is_state_stale(self, server: ServerState) -> bool:
        age = (datetime.now() - server.last_updated).total_seconds()
        return age > self.state_ttl_seconds
    
    async def select_server(self, request_path: str) -> Optional[ServerState]:
        if not self.servers:
            return None
        
        any_stale = any(self.is_state_stale(s) for s in self.servers.values())
        if any_stale:
            await self.update_all_servers()
        
        scored_servers = []
        for server in self.servers.values():
            score = self.calculate_heuristic(server, request_path)
            scored_servers.append((score, server))
        
        scored_servers.sort(key=lambda x: x[0])
        
        best_score, best_server = scored_servers[0]
        
        return best_server
    
    def record_request_start(self, server_id: str):
        if server_id in self.servers:
            self.servers[server_id].active_requests += 1
    
    def record_request_end(self, server_id: str, response_time_ms: float, path: str):
        if server_id in self.servers:
            server = self.servers[server_id]
            server.active_requests = max(0, server.active_requests - 1)
    
    def get_routing_decision_info(self, request_path: str) -> List[Dict]:
        info = []
        for server in self.servers.values():
            score = self.calculate_heuristic(server, request_path)
            cache_hit = request_path in server.cached_paths
            info.append({
                "server_id": server.server_id,
                "score": round(score, 2),
                "active_requests": server.active_requests,
                "avg_response_time_ms": round(server.avg_response_time_ms, 2),
                "cache_hit": cache_hit,
                "is_healthy": server.is_healthy
            })
        return sorted(info, key=lambda x: x['score'])


router = AStarRouter()