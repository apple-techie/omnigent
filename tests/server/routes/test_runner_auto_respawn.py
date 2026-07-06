"""Tests for host-bound runner auto-respawn (issues #1857 / #1953 gap).

Covers the conservative, host-bound-only proactive relaunch that
``_on_runner_disconnect`` schedules so an orphaned open session recovers
without a manual "click to reconnect". These are unit-level tests of the
orchestration in :mod:`omnigent.server.routes.sessions`, driving the
functions directly with fakes and stubbing the two reused helpers
(:func:`_launch_runner_on_host`, :func:`_wait_for_runner_client`) so the
safety gates are pinned in isolation from the HTTP / tunnel machinery.

The contract under test (see the module docstring in ``sessions.py``):
  * host online  -> auto-respawn happens;
  * host offline -> no respawn (today's manual path preserved);
  * user Stopped -> no respawn (Stop is non-sticky, marked out of band);
  * transient flap (original runner reconnects) -> no duplicate runner;
  * a respawn already in flight for the conversation -> no duplicate;
  * a host that keeps killing its runner -> capped, not an infinite loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from omnigent.entities import Conversation
from omnigent.server.routes import sessions as s

# ── Fakes ────────────────────────────────────────────────────────────


@dataclass
class _FakeHostRegistry:
    """Minimal ``HostRegistry`` stand-in keyed by host_id."""

    conns: dict[str, object] = field(default_factory=dict)

    def get(self, host_id: str) -> object | None:
        return self.conns.get(host_id)


@dataclass
class _FakeTunnelRegistry:
    """Runner-tunnel registry whose ``wait_for_runner`` is scripted.

    ``reconnects`` maps a runner_id to the session object its debounce
    wait should resolve to (a non-None value means "the same runner came
    back" — the flap case). Absent ids resolve to ``None`` (timed out /
    genuinely dead), and every call is recorded.
    """

    reconnects: dict[str, object] = field(default_factory=dict)
    waited: list[tuple[str, float]] = field(default_factory=list)

    async def wait_for_runner(self, runner_id: str, *, timeout_s: float) -> object | None:
        self.waited.append((runner_id, timeout_s))
        return self.reconnects.get(runner_id)

    def get(self, runner_id: str) -> object | None:
        return self.reconnects.get(runner_id)


@dataclass
class _FakeConversationStore:
    """Store exposing only the by-runner lookup + rebind the path uses."""

    convs: dict[str, Conversation] = field(default_factory=dict)
    replaced: list[tuple[str, str]] = field(default_factory=list)

    def list_conversations_by_runner_id(self, runner_id: str) -> list[Conversation]:
        return [c for c in self.convs.values() if c.runner_id == runner_id]

    def replace_runner_id(self, conversation_id: str, runner_id: str) -> None:
        self.replaced.append((conversation_id, runner_id))
        conv = self.convs.get(conversation_id)
        if conv is not None:
            self.convs[conversation_id] = _replace(conv, runner_id=runner_id)


def _replace(conv: Conversation, **changes: object) -> Conversation:
    import dataclasses

    return dataclasses.replace(conv, **changes)


def _make_conv(
    conv_id: str = "conv_1",
    *,
    host_id: str | None = "host_1",
    runner_id: str | None = "runner_dead",
    workspace: str | None = "/work/repo",
) -> Conversation:
    return Conversation(
        id=conv_id,
        created_at=1,
        updated_at=1,
        root_conversation_id=conv_id,
        agent_id="ag_test",
        host_id=host_id,
        runner_id=runner_id,
        workspace=workspace,
    )


@pytest.fixture(autouse=True)
def _clear_module_state() -> None:
    """Reset the module-level respawn state around each test."""
    s._intentionally_stopped_runners.clear()
    s._auto_respawn_in_flight.clear()
    s._auto_respawn_attempts.clear()
    yield
    s._intentionally_stopped_runners.clear()
    s._auto_respawn_in_flight.clear()
    s._auto_respawn_attempts.clear()


@pytest.fixture()
def stub_launch(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Stub ``_launch_runner_on_host`` to record calls, mint a new runner.

    Returns the list of recorded calls so tests can assert whether a
    respawn was actually attempted.
    """
    calls: list[dict[str, object]] = []

    async def _fake_launch(conv, conversation_store, host_registry, host_conn):  # type: ignore[no-untyped-def]
        calls.append({"conv_id": conv.id, "host_conn": host_conn})
        # Mirror the real helper's rebind so state stays coherent.
        conversation_store.replace_runner_id(conv.id, "runner_new")
        return s._HostLaunchAttempt(runner_id="runner_new")

    monkeypatch.setattr(s, "_launch_runner_on_host", _fake_launch)
    return calls


@pytest.fixture()
def stub_wait_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``_wait_for_runner_client`` so the fresh runner 'connects'."""

    async def _fake_wait(session_id, runner_router, tunnel_registry, **kwargs):  # type: ignore[no-untyped-def]
        return object()  # a truthy client == connected

    monkeypatch.setattr(s, "_wait_for_runner_client", _fake_wait)


# ── Per-conversation gate tests ──────────────────────────────────────


@pytest.mark.asyncio
class TestMaybeRespawnConversationRunner:
    async def test_host_online_respawns(
        self, stub_launch: list[dict[str, object]], stub_wait_connected: None
    ) -> None:
        conv = _make_conv()
        store = _FakeConversationStore({conv.id: conv})
        host_registry = _FakeHostRegistry({"host_1": object()})
        tunnel = _FakeTunnelRegistry()

        await s._maybe_respawn_conversation_runner(
            conv,
            dead_runner_id="runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        assert len(stub_launch) == 1
        assert store.replaced == [(conv.id, "runner_new")]
        # A launched respawn consumes one unit of the retry budget.
        assert len(s._auto_respawn_attempts[conv.id]) == 1
        # The in-flight guard is released once the respawn settles.
        assert conv.id not in s._auto_respawn_in_flight

    async def test_host_offline_does_not_respawn(
        self, stub_launch: list[dict[str, object]]
    ) -> None:
        conv = _make_conv()
        store = _FakeConversationStore({conv.id: conv})
        # Host tunnel is NOT live on this replica.
        host_registry = _FakeHostRegistry({})
        tunnel = _FakeTunnelRegistry()

        await s._maybe_respawn_conversation_runner(
            conv,
            dead_runner_id="runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        assert stub_launch == []
        assert store.replaced == []
        # No budget is consumed when a gate short-circuits before launch.
        assert conv.id not in s._auto_respawn_attempts or not s._auto_respawn_attempts[conv.id]

    async def test_local_session_not_host_bound_does_not_respawn(
        self, stub_launch: list[dict[str, object]]
    ) -> None:
        # A CLI / local_stranded session has no host_id — nothing to
        # relaunch on. Even with a (spurious) live host in the registry it
        # must be left to the manual path.
        conv = _make_conv(host_id=None)
        store = _FakeConversationStore({conv.id: conv})
        host_registry = _FakeHostRegistry({"host_1": object()})
        tunnel = _FakeTunnelRegistry()

        await s._maybe_respawn_conversation_runner(
            conv,
            dead_runner_id="runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        assert stub_launch == []

    async def test_user_stopped_does_not_respawn(
        self, stub_launch: list[dict[str, object]]
    ) -> None:
        conv = _make_conv()
        store = _FakeConversationStore({conv.id: conv})
        host_registry = _FakeHostRegistry({"host_1": object()})
        tunnel = _FakeTunnelRegistry()
        # The user Stopped this runner just before it disconnected.
        s.mark_runner_intentionally_stopped("runner_dead")

        await s._maybe_respawn_conversation_runner(
            conv,
            dead_runner_id="runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        assert stub_launch == []
        assert store.replaced == []

    async def test_binding_moved_on_does_not_respawn(
        self, stub_launch: list[dict[str, object]]
    ) -> None:
        # The row already points at a different (live) runner — a
        # concurrent path rebound it. Nothing to respawn.
        conv = _make_conv(runner_id="runner_current")
        store = _FakeConversationStore({conv.id: conv})
        host_registry = _FakeHostRegistry({"host_1": object()})
        tunnel = _FakeTunnelRegistry()

        await s._maybe_respawn_conversation_runner(
            conv,
            dead_runner_id="runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        assert stub_launch == []

    async def test_concurrent_guard_blocks_second_respawn(
        self, stub_launch: list[dict[str, object]]
    ) -> None:
        conv = _make_conv()
        store = _FakeConversationStore({conv.id: conv})
        host_registry = _FakeHostRegistry({"host_1": object()})
        tunnel = _FakeTunnelRegistry()
        # A respawn is already in flight for this conversation.
        s._auto_respawn_in_flight.add(conv.id)

        await s._maybe_respawn_conversation_runner(
            conv,
            dead_runner_id="runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        assert stub_launch == []
        # The pre-existing guard entry is left intact (owned by the other
        # in-flight respawn), not cleared by this bailout.
        assert conv.id in s._auto_respawn_in_flight

    async def test_retry_budget_caps_respawns(self, stub_launch: list[dict[str, object]]) -> None:
        conv = _make_conv()
        store = _FakeConversationStore({conv.id: conv})
        host_registry = _FakeHostRegistry({"host_1": object()})
        tunnel = _FakeTunnelRegistry()
        # Fill the rolling window to the cap with recent attempts.
        now = time.monotonic()
        s._auto_respawn_attempts[conv.id] = [now] * s._AUTO_RESPAWN_MAX_ATTEMPTS

        await s._maybe_respawn_conversation_runner(
            conv,
            dead_runner_id="runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        assert stub_launch == []

    async def test_harness_refusal_does_not_retry_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The host refuses (harness not configured). The respawn must
        # bail without waiting for a connect, and release the guard.
        async def _refuse(conv, conversation_store, host_registry, host_conn):  # type: ignore[no-untyped-def]
            conversation_store.replace_runner_id(conv.id, "runner_new")
            return s._HostLaunchAttempt(
                runner_id="runner_new",
                error_code=s._HARNESS_NOT_CONFIGURED_ERROR_CODE,
                error="harness 'codex' is not configured on host 'laptop'",
            )

        waited: list[str] = []

        async def _fake_wait(session_id, *a, **k):  # type: ignore[no-untyped-def]
            waited.append(session_id)
            return object()

        monkeypatch.setattr(s, "_launch_runner_on_host", _refuse)
        monkeypatch.setattr(s, "_wait_for_runner_client", _fake_wait)

        conv = _make_conv()
        store = _FakeConversationStore({conv.id: conv})
        host_registry = _FakeHostRegistry({"host_1": object()})
        tunnel = _FakeTunnelRegistry()

        await s._maybe_respawn_conversation_runner(
            conv,
            dead_runner_id="runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        # No connect wait on a deterministic refusal, guard released.
        assert waited == []
        assert conv.id not in s._auto_respawn_in_flight


# ── Debounce / flap tests (task body) ────────────────────────────────


@pytest.mark.asyncio
class TestAutoRespawnAfterDisconnect:
    async def test_transient_flap_reconnect_no_duplicate(
        self, stub_launch: list[dict[str, object]], stub_wait_connected: None
    ) -> None:
        conv = _make_conv()
        store = _FakeConversationStore({conv.id: conv})
        host_registry = _FakeHostRegistry({"host_1": object()})
        # The SAME runner re-registers during the debounce window.
        tunnel = _FakeTunnelRegistry(reconnects={"runner_dead": object()})

        await s._auto_respawn_runner_after_disconnect(
            "runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        # Debounce waited on the dead runner, then aborted — no respawn.
        assert tunnel.waited and tunnel.waited[0][0] == "runner_dead"
        assert stub_launch == []
        assert store.replaced == []

    async def test_dead_after_debounce_respawns(
        self, stub_launch: list[dict[str, object]], stub_wait_connected: None
    ) -> None:
        conv = _make_conv()
        store = _FakeConversationStore({conv.id: conv})
        host_registry = _FakeHostRegistry({"host_1": object()})
        # Runner does NOT reconnect during the debounce (absent id).
        tunnel = _FakeTunnelRegistry(reconnects={})

        await s._auto_respawn_runner_after_disconnect(
            "runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        assert len(stub_launch) == 1
        assert store.replaced == [(conv.id, "runner_new")]

    async def test_multiple_bound_convs_each_respawned(
        self, stub_launch: list[dict[str, object]], stub_wait_connected: None
    ) -> None:
        conv_a = _make_conv("conv_a")
        conv_b = _make_conv("conv_b")
        store = _FakeConversationStore({conv_a.id: conv_a, conv_b.id: conv_b})
        host_registry = _FakeHostRegistry({"host_1": object()})
        tunnel = _FakeTunnelRegistry(reconnects={})

        await s._auto_respawn_runner_after_disconnect(
            "runner_dead",
            conversation_store=store,
            host_registry=host_registry,
            tunnel_registry=tunnel,
            runner_router=None,
            runner_exit_reports=None,
        )

        assert {c["conv_id"] for c in stub_launch} == {"conv_a", "conv_b"}


# ── Scheduler inline-gate tests ──────────────────────────────────────


class TestScheduleRunnerAutoRespawn:
    def test_no_host_registry_schedules_nothing(self) -> None:
        store = _FakeConversationStore()
        # host_registry=None (host support not wired) -> no task created.
        s.schedule_runner_auto_respawn(
            "runner_dead",
            conversation_store=store,
            host_registry=None,
            tunnel_registry=_FakeTunnelRegistry(),
            runner_router=None,
            runner_exit_reports=None,
        )
        assert not s._auto_respawn_tasks

    def test_intentionally_stopped_schedules_nothing(self) -> None:
        store = _FakeConversationStore()
        s.mark_runner_intentionally_stopped("runner_dead")
        s.schedule_runner_auto_respawn(
            "runner_dead",
            conversation_store=store,
            host_registry=_FakeHostRegistry({"host_1": object()}),
            tunnel_registry=_FakeTunnelRegistry(),
            runner_router=None,
            runner_exit_reports=None,
        )
        assert not s._auto_respawn_tasks


# ── Intentional-stop marker semantics ────────────────────────────────


class TestIntentionalStopMarker:
    def test_marker_expires(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = {"t": 1000.0}
        monkeypatch.setattr(s.time, "monotonic", lambda: clock["t"])
        s.mark_runner_intentionally_stopped("runner_x")
        assert s._runner_was_intentionally_stopped("runner_x") is True
        # Advance past the TTL — the marker is treated as gone (and pruned).
        clock["t"] += s._INTENTIONAL_STOP_MARKER_TTL_S + 1
        assert s._runner_was_intentionally_stopped("runner_x") is False
        assert "runner_x" not in s._intentionally_stopped_runners

    def test_unmarked_runner_is_not_stopped(self) -> None:
        assert s._runner_was_intentionally_stopped("runner_never") is False
