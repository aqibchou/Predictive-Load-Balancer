"""
TelemetryAgent — runs inside each backend pod.

Opens a bidirectional gRPC stream to the load balancer's TelemetryService
and pushes a TelemetryPayload every STREAM_INTERVAL_S seconds.  If the
stream is dropped (network error, LB restart) it waits RECONNECT_DELAY_S
and reconnects automatically.
"""

import asyncio
import os
import sys
import time
import grpc
import grpc.aio
import psutil

# Allow `from telemetry import telemetry_pb2` — needs the PROJECT ROOT in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from telemetry import telemetry_pb2, telemetry_pb2_grpc

GRPC_LB_HOST      = os.getenv("GRPC_LB_HOST", "localhost:50051")
SERVER_ID         = os.getenv("SERVER_ID", "backend-unknown")
STREAM_INTERVAL_S = float(os.getenv("TELEMETRY_INTERVAL_S", "5"))
RECONNECT_DELAY_S = float(os.getenv("TELEMETRY_RECONNECT_S", "5"))


class TelemetryAgent:
    """Streams backend telemetry to the load balancer over gRPC."""

    def __init__(self, metrics_ref, cache_ref):
        """
        Parameters
        ----------
        metrics_ref : ServerMetrics
            Live metrics object from server.py.
        cache_ref : LRUCache
            Live cache object from server.py.
        """
        self._metrics = metrics_ref
        self._cache   = cache_ref
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Schedule the streaming loop as a background asyncio task."""
        self._task = asyncio.create_task(self._run(), name="telemetry-agent")
        print(f"[TelemetryAgent] Started for {SERVER_ID} → {GRPC_LB_HOST}")

    def stop(self):
        """Cancel the streaming loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            print("[TelemetryAgent] Stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run(self):
        """Outer reconnect loop."""
        while True:
            try:
                await self._stream_loop()
            except asyncio.CancelledError:
                print("[TelemetryAgent] Cancelled — exiting")
                raise
            except Exception as exc:
                print(f"[TelemetryAgent] Connection error: {exc!r} — "
                      f"reconnecting in {RECONNECT_DELAY_S}s")
            await asyncio.sleep(RECONNECT_DELAY_S)

    async def _stream_loop(self):
        """Open one gRPC channel and stream until it breaks."""
        print(f"[TelemetryAgent] Connecting to {GRPC_LB_HOST} …")
        async with grpc.aio.insecure_channel(GRPC_LB_HOST) as channel:
            stub = telemetry_pb2_grpc.TelemetryServiceStub(channel)
            print(f"[TelemetryAgent] Streaming telemetry to {GRPC_LB_HOST}")
            call = stub.StreamTelemetry(self._payload_generator())
            async for ack in call:
                if ack.received:
                    pass  # ACK confirmed; payload generator drives the cadence

    async def _payload_generator(self):
        """Yield one TelemetryPayload every STREAM_INTERVAL_S seconds."""
        while True:
            m = self._metrics
            c = self._cache
            yield telemetry_pb2.TelemetryPayload(
                server_id            = SERVER_ID,
                cpu_percent          = psutil.cpu_percent(interval=None),
                memory_percent       = psutil.virtual_memory().percent,
                active_requests      = m.active_requests,
                avg_response_time_ms = m.get_avg_response_time() * 1000,
                cache_hit_rate       = m.get_cache_hit_rate(),
                timestamp_ns         = time.time_ns(),
                total_requests       = m.total_requests,
                cache_size           = c.size(),
            )
            await asyncio.sleep(STREAM_INTERVAL_S)
