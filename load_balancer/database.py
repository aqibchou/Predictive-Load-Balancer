import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import asyncpg

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@postgres:5432/loadbalancer"
)

pool: Optional[asyncpg.Pool] = None


async def init_database():
    global pool
    
    try:
        pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10
        )
        print(f"Database pool created")
        
        await create_tables()
        return True
        
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


async def close_database():
    global pool
    if pool:
        await pool.close()
        print("Database pool closed")


async def create_tables():
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS request_metrics (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                server_id VARCHAR(50),
                path VARCHAR(500),
                response_time_ms FLOAT,
                status_code INT,
                cache_hit BOOLEAN,
                bytes_sent INT
            )
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
            ON request_metrics(timestamp DESC)
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS traffic_aggregates (
                id SERIAL PRIMARY KEY,
                minute_bucket TIMESTAMPTZ UNIQUE,
                request_count INT,
                avg_response_time FLOAT,
                total_bytes BIGINT,
                success_count INT,
                total_count INT
            )
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_aggregates_bucket 
            ON traffic_aggregates(minute_bucket DESC)
        """)
        
        print("Database tables created")


async def log_request(
    server_id: str,
    path: str,
    response_time_ms: float,
    status_code: int,
    cache_hit: bool = False,
    bytes_sent: int = 0
):
    if not pool:
        return
    
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO request_metrics 
                (server_id, path, response_time_ms, status_code, cache_hit, bytes_sent)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, server_id, path, response_time_ms, status_code, cache_hit, bytes_sent)
    except Exception as e:
        print(f"Failed to log request: {e}")


async def update_minute_aggregate():
    if not pool:
        return
    
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO traffic_aggregates (minute_bucket, request_count, avg_response_time, total_bytes, success_count, total_count)
                SELECT 
                    DATE_TRUNC('minute', timestamp) as minute_bucket,
                    COUNT(*) as request_count,
                    AVG(response_time_ms) as avg_response_time,
                    SUM(bytes_sent) as total_bytes,
                    SUM(CASE WHEN status_code < 400 THEN 1 ELSE 0 END) as success_count,
                    COUNT(*) as total_count
                FROM request_metrics
                WHERE timestamp >= DATE_TRUNC('minute', NOW()) - INTERVAL '1 minute'
                  AND timestamp < DATE_TRUNC('minute', NOW())
                GROUP BY DATE_TRUNC('minute', timestamp)
                ON CONFLICT (minute_bucket) DO UPDATE SET
                    request_count = EXCLUDED.request_count,
                    avg_response_time = EXCLUDED.avg_response_time,
                    total_bytes = EXCLUDED.total_bytes,
                    success_count = EXCLUDED.success_count,
                    total_count = EXCLUDED.total_count
            """)
    except Exception as e:
        print(f"Failed to update aggregate: {e}")


async def get_recent_traffic(minutes: int = 60) -> List[Dict]:
    if not pool:
        return []
    
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    minute_bucket,
                    request_count,
                    avg_response_time,
                    total_bytes,
                    success_count,
                    total_count
                FROM traffic_aggregates
                WHERE minute_bucket >= NOW() - INTERVAL '%s minutes'
                ORDER BY minute_bucket ASC
            """ % minutes)
            
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"Failed to get recent traffic: {e}")
        return []


async def get_traffic_at_lag(minutes_ago: int) -> Optional[int]:
    if not pool:
        return None
    
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT request_count
                FROM traffic_aggregates
                WHERE minute_bucket = DATE_TRUNC('minute', NOW() - INTERVAL '%s minutes')
            """ % minutes_ago)
            
            return row['request_count'] if row else None
    except Exception as e:
        print(f"Failed to get traffic at lag: {e}")
        return None