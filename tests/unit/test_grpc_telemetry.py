"""
Unit tests for gRPC telemetry pipeline:
  - AStarRouter.update_from_grpc() (routing.py)
  - TelemetryServicer.StreamTelemetry() (grpc_server.py)
  - TelemetryAgent lifecycle (telemetry_agent.py)

No real gRPC server/client is started — all network I/O is mocked.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── sys.path setup ─────────────────────────────────────────────────────────────
_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_LB    = os.path.join(_ROOT, "load_balancer")
_BE    = os.path.join(_ROOT, "backend")

for p in (_ROOT, _LB, _BE):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Shared imports ─────────────────────────────────────────────────────────────
from routing import AStarRouter, CircuitState, ServerState  # noqa: E402


# =============================================================================
# Section 1 — AStarRouter.update_from_grpc
# =============================================================================

def make_router(*server_ids):
    r = AStarRouter()
    for sid in server_ids:
        r.register_server(sid, f"http://localhost:800{server_ids.index(sid)+1}")
    return r


class TestUpdateFromGrpc:
    def test_updates_active_requests(self):
        r = make_router("s1")
        r.update_from_grpc("s1", {"active_requests": 7})
        assert r.servers["s1"].active_requests == 7

    def test_updates_avg_response_time(self):
        r = make_router("s1")
        r.update_from_grpc("s1", {"avg_response_time_ms": 123.4})
        assert r.servers["s1"].avg_response_time_ms == pytest.approx(123.4)

    def test_marks_server_healthy(self):
        r = make_router("s1")
        r.servers["s1"].is_healthy = False
        r.update_from_grpc("s1", {})
        assert r.servers["s1"].is_healthy is True

    def test_resets_consecutive_failures(self):
        r = make_router("s1")
        r.servers["s1"].consecutive_failures = 5
        r.update_from_grpc("s1", {})
        assert r.servers["s1"].consecutive_failures == 0

    def test_sets_fetched_true(self):
        r = make_router("s1")
        assert r.servers["s1"].fetched is False
        r.update_from_grpc("s1", {})
        assert r.servers["s1"].fetched is True

    def test_updates_last_updated(self):
        r = make_router("s1")
        before = r.servers["s1"].last_updated
        time.sleep(0.01)
        r.update_from_grpc("s1", {})
        assert r.servers["s1"].last_updated > before

    def test_unknown_server_id_is_noop(self):
        r = make_router("s1")
        # Should not raise
        r.update_from_grpc("does_not_exist", {"active_requests": 99})
        assert r.servers["s1"].active_requests == 0   # unchanged

    def test_resets_half_open_circuit_to_closed(self):
        r = make_router("s1")
        r.servers["s1"].circuit_state   = CircuitState.HALF_OPEN
        r.servers["s1"].circuit_open_at = datetime.now()
        r.update_from_grpc("s1", {})
        assert r.servers["s1"].circuit_state    == CircuitState.CLOSED
        assert r.servers["s1"].circuit_open_at  is None

    def test_closed_circuit_stays_closed(self):
        r = make_router("s1")
        r.update_from_grpc("s1", {})
        assert r.servers["s1"].circuit_state == CircuitState.CLOSED

    def test_open_circuit_not_changed(self):
        """OPEN circuit is not touched by gRPC update (stays open until probe succeeds)."""
        r = make_router("s1")
        r.servers["s1"].circuit_state = CircuitState.OPEN
        r.update_from_grpc("s1", {})
        assert r.servers["s1"].circuit_state == CircuitState.OPEN

    def test_partial_data_preserves_existing_values(self):
        r = make_router("s1")
        r.servers["s1"].active_requests      = 10
        r.servers["s1"].avg_response_time_ms = 50.0
        # Only send avg_response_time_ms
        r.update_from_grpc("s1", {"avg_response_time_ms": 99.9})
        assert r.servers["s1"].active_requests      == 10    # unchanged
        assert r.servers["s1"].avg_response_time_ms == pytest.approx(99.9)

    def test_multiple_servers_isolated(self):
        r = make_router("s1", "s2")
        r.update_from_grpc("s1", {"active_requests": 5})
        r.update_from_grpc("s2", {"active_requests": 10})
        assert r.servers["s1"].active_requests == 5
        assert r.servers["s2"].active_requests == 10

    def test_fetched_suppresses_http_polling_flag(self):
        """After gRPC update, fetched=True means select_server won't HTTP-poll this server."""
        r = make_router("s1")
        r.update_from_grpc("s1", {"active_requests": 3, "avg_response_time_ms": 20.0})
        s = r.servers["s1"]
        assert s.fetched is True
        # Stale check: if last_updated is recent, is_state_stale returns False
        assert r.is_state_stale(s) is False


# =============================================================================
# Section 2 — TelemetryServicer (grpc_server.py)
# =============================================================================

@pytest.mark.asyncio
class TestTelemetryServicer:
    """Tests for grpc_server.TelemetryServicer.StreamTelemetry"""

    @staticmethod
    def _get_servicer():
        """Import TelemetryServicer with grpc.aio available."""
        import grpc_server as gs
        mock_router = MagicMock()
        return gs.TelemetryServicer(mock_router), mock_router

    async def test_servicer_yields_ack_for_each_payload(self):
        from telemetry import telemetry_pb2
        servicer, mock_router = self._get_servicer()

        async def request_gen():
            for i in range(3):
                yield telemetry_pb2.TelemetryPayload(
                    server_id=f"s{i}", active_requests=i
                )

        context = MagicMock()
        context.peer.return_value = "ipv4:127.0.0.1:9999"

        acks = []
        async for ack in servicer.StreamTelemetry(request_gen(), context):
            acks.append(ack)

        assert len(acks) == 3

    async def test_servicer_ack_received_true(self):
        from telemetry import telemetry_pb2
        servicer, _ = self._get_servicer()

        async def request_gen():
            yield telemetry_pb2.TelemetryPayload(server_id="s1", active_requests=2)

        context = MagicMock()
        context.peer.return_value = "ipv4:127.0.0.1:9999"

        acks = []
        async for ack in servicer.StreamTelemetry(request_gen(), context):
            acks.append(ack)

        assert acks[0].received is True

    async def test_servicer_ack_echoes_server_id(self):
        from telemetry import telemetry_pb2
        servicer, _ = self._get_servicer()

        async def request_gen():
            yield telemetry_pb2.TelemetryPayload(server_id="backend-42")

        context = MagicMock()
        context.peer.return_value = "ipv4:127.0.0.1:1234"

        ack = None
        async for ack in servicer.StreamTelemetry(request_gen(), context):
            pass
        assert ack.server_id == "backend-42"

    async def test_servicer_calls_update_from_grpc(self):
        from telemetry import telemetry_pb2
        servicer, mock_router = self._get_servicer()

        async def request_gen():
            yield telemetry_pb2.TelemetryPayload(
                server_id="srv1",
                active_requests=5,
                avg_response_time_ms=77.0,
                cache_hit_rate=0.9,
                cpu_percent=45.0,
            )

        context = MagicMock()
        context.peer.return_value = "ipv4:127.0.0.1:5000"

        async for _ in servicer.StreamTelemetry(request_gen(), context):
            pass

        mock_router.update_from_grpc.assert_called_once()
        call_args = mock_router.update_from_grpc.call_args
        assert call_args[0][0] == "srv1"
        data = call_args[0][1]
        assert data["active_requests"]      == 5
        assert data["avg_response_time_ms"] == pytest.approx(77.0, abs=0.1)
        assert data["cache_hit_rate"]       == pytest.approx(0.9,  abs=0.01)

    async def test_servicer_empty_stream_yields_nothing(self):
        servicer, mock_router = self._get_servicer()

        async def request_gen():
            return
            yield  # make it an async generator

        context = MagicMock()
        context.peer.return_value = "ipv4:127.0.0.1:1"

        acks = [a async for a in servicer.StreamTelemetry(request_gen(), context)]
        assert len(acks) == 0
        mock_router.update_from_grpc.assert_not_called()


# =============================================================================
# Section 3 — TelemetryAgent lifecycle (backend/telemetry_agent.py)
# =============================================================================

class TestTelemetryAgent:
    """Tests for TelemetryAgent start/stop and payload generation."""

    @staticmethod
    def _make_metrics(active=0, total=0, cache_hits=0):
        """Build a minimal ServerMetrics-compatible mock."""
        m = MagicMock()
        m.active_requests = active
        m.total_requests  = total
        m.get_avg_response_time.return_value = 0.1
        m.get_cache_hit_rate.return_value    = 0.5
        return m

    @staticmethod
    def _make_cache(size=5):
        c = MagicMock()
        c.size.return_value = size
        return c

    def test_agent_start_creates_task(self):
        """TelemetryAgent.start() creates an asyncio.Task."""
        import telemetry_agent as ta

        m = self._make_metrics()
        c = self._make_cache()

        agent = ta.TelemetryAgent(m, c)

        with patch.object(ta, "grpc"), \
             patch.object(ta, "psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 10.0
            mock_psutil.virtual_memory.return_value.percent = 40.0

            async def run_test():
                agent.start()
                await asyncio.sleep(0)   # yield to let task start
                assert agent._task is not None
                assert not agent._task.done()
                agent.stop()
                await asyncio.sleep(0)
                assert agent._task.cancelled() or agent._task.done()

            asyncio.run(run_test())

    def test_agent_stop_cancels_task(self):
        import telemetry_agent as ta

        m = self._make_metrics()
        c = self._make_cache()
        agent = ta.TelemetryAgent(m, c)

        with patch.object(ta, "grpc"):
            async def run_test():
                agent.start()
                await asyncio.sleep(0)
                agent.stop()
                await asyncio.sleep(0)
                return agent._task.cancelled() or agent._task.done()

            result = asyncio.run(run_test())
        assert result is True

    def test_agent_stop_without_start_is_safe(self):
        import telemetry_agent as ta

        agent = ta.TelemetryAgent(self._make_metrics(), self._make_cache())
        # Should not raise
        agent.stop()

    @pytest.mark.asyncio
    async def test_payload_generator_yields_correct_fields(self):
        import telemetry_agent as ta
        from telemetry import telemetry_pb2

        m = self._make_metrics(active=3, total=100)
        c = self._make_cache(size=42)

        agent = ta.TelemetryAgent(m, c)

        with patch.object(ta, "psutil") as mock_psutil, \
             patch.object(ta, "SERVER_ID", "test-server"), \
             patch.object(ta, "STREAM_INTERVAL_S", 0):   # no sleep
            mock_psutil.cpu_percent.return_value        = 55.0
            mock_psutil.virtual_memory.return_value.percent = 72.0

            payloads = []
            # Collect just one payload from the generator
            async for payload in agent._payload_generator():
                payloads.append(payload)
                break   # stop after first

        assert len(payloads) == 1
        p = payloads[0]
        assert p.server_id        == "test-server"
        assert p.active_requests  == 3
        assert p.total_requests   == 100
        assert p.cache_size       == 42
        assert p.cpu_percent      == pytest.approx(55.0, abs=0.1)
        assert p.memory_percent   == pytest.approx(72.0, abs=0.1)
        assert p.cache_hit_rate   == pytest.approx(0.5,  abs=0.01)
        assert p.timestamp_ns     > 0


# =============================================================================
# Section 4 — grpc_server module-level helpers
# =============================================================================

@pytest.mark.asyncio
async def test_grpc_server_start_stop():
    """grpc_server.start() creates an aio server; stop() shuts it down cleanly."""
    import grpc_server as gs

    mock_router = MagicMock()

    with patch("grpc_server.grpc") as mock_grpc_mod:
        mock_server = AsyncMock()
        mock_grpc_mod.aio.server.return_value = mock_server
        mock_server.add_insecure_port.return_value = None
        mock_server.start = AsyncMock()
        mock_server.stop  = AsyncMock()

        await gs.start(mock_router)
        mock_server.start.assert_awaited_once()

        await gs.stop()
        mock_server.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_grpc_server_stop_without_start_is_safe():
    """Calling stop() before start() must not raise."""
    import grpc_server as gs

    # Reset the global _server to None
    gs._server = None
    await gs.stop()   # should be a no-op
