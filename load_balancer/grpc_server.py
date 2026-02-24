"""
gRPC Telemetry Server — runs inside the load balancer process on port 50051.

Backends connect and stream TelemetryPayload messages via the bidirectional
StreamTelemetry RPC.  Each received payload is forwarded to
router.update_from_grpc() so routing decisions use live binary telemetry
instead of HTTP-polled JSON.
"""

import sys
import os
import grpc
import grpc.aio

# Allow `from telemetry import telemetry_pb2` — needs the PROJECT ROOT in sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from telemetry import telemetry_pb2, telemetry_pb2_grpc  # noqa: E402

GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))

_server = None


class TelemetryServicer(telemetry_pb2_grpc.TelemetryServiceServicer):
    """Receives streaming telemetry from backend agents."""

    def __init__(self, router):
        self._router = router

    async def StreamTelemetry(self, request_iterator, context):
        peer = context.peer()
        async for payload in request_iterator:
            self._router.update_from_grpc(payload.server_id, {
                "active_requests":      payload.active_requests,
                "avg_response_time_ms": payload.avg_response_time_ms,
                "cache_hit_rate":       payload.cache_hit_rate,
                "cpu_percent":          payload.cpu_percent,
                "memory_percent":       payload.memory_percent,
                "total_requests":       payload.total_requests,
                "cache_size":           payload.cache_size,
            })
            print(f"[gRPC] TelemetryService received: {payload.server_id} "
                  f"(active={payload.active_requests}, "
                  f"rt={payload.avg_response_time_ms:.1f}ms, peer={peer})")
            yield telemetry_pb2.Ack(received=True, server_id=payload.server_id)


async def start(router):
    """Create and start the gRPC server; store reference globally for stop()."""
    global _server
    _server = grpc.aio.server()
    telemetry_pb2_grpc.add_TelemetryServiceServicer_to_server(
        TelemetryServicer(router), _server
    )
    listen_addr = f"[::]:{GRPC_PORT}"
    _server.add_insecure_port(listen_addr)
    await _server.start()
    print(f"[gRPC] TelemetryService listening on {listen_addr}")


async def stop():
    """Gracefully stop the gRPC server."""
    global _server
    if _server is not None:
        await _server.stop(grace=5)
        _server = None
        print("[gRPC] TelemetryService stopped")
