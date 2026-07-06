"""Unit tests for the summarizer-model resolution order in
``_run_compact_locked`` (omnigent/server/routes/sessions.py).

Resolution order (first hit wins):

    1. spec.llm
    2. spec.executor.model
    3. get_caps().llm      <- server-level fallback (this fix)
    4. raise "Compaction requires a configured LLM model"

The caps.llm fallback lets omnigent-executor specs (polly and its
sub-agents) — which intentionally pin no ``spec.llm`` / ``executor.model``
— still compact, sourcing the summarizer from the deployment's
server-level ``llm:`` config instead of an in-spec secret env-ref. These
tests drive ``_run_compact_locked`` directly with fakes and capture the
``LLMConfig`` handed to ``compact_conversation_now``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from omnigent.errors import OmnigentError
from omnigent.runtime.caps import RuntimeCaps
from omnigent.server.routes import sessions as sessions_mod
from omnigent.spec.types import ExecutorSpec, LLMConfig


class _FakeAgentCache:
    """Minimal AgentCache stand-in returning a fixed spec."""

    def __init__(self, spec: Any) -> None:
        self._spec = spec

    def load(self, _agent_id: str, _bundle_location: str, *, expand_env: bool) -> Any:
        return SimpleNamespace(spec=self._spec)


class _FakeAgentStore:
    """Minimal AgentStore stand-in returning a fixed agent."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def get(self, _agent_id: str) -> Any:
        return self._agent


def _omnigent_spec(
    *,
    llm: LLMConfig | None,
    executor_model: str | None,
    executor_connection: dict[str, str] | None = None,
) -> Any:
    """Build a spec-like object with the fields _run_compact_locked reads."""
    return SimpleNamespace(
        llm=llm,
        executor=ExecutorSpec(
            type="omnigent",
            config={"harness": "claude-sdk"},
            model=executor_model,
            connection=executor_connection,
        ),
    )


async def _resolve_llm_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    spec: Any,
    caps_llm: LLMConfig | None,
) -> LLMConfig:
    """Drive _run_compact_locked with fakes; return the resolved LLMConfig.

    Captures the ``llm_config`` argument that _run_compact_locked passes
    to ``compact_conversation_now``. Raises whatever _run_compact_locked
    raises (e.g. the "no model configured" OmnigentError).
    """
    captured: dict[str, Any] = {}

    async def _capture(**kwargs: Any) -> None:
        captured["llm_config"] = kwargs["llm_config"]

    # compact_conversation_now is imported inside the function from
    # omnigent.runtime.workflow — patch it at the source module.
    monkeypatch.setattr("omnigent.runtime.workflow.compact_conversation_now", _capture)
    # Silence stream side effects.
    monkeypatch.setattr(sessions_mod, "_publish_status", lambda *a, **k: None)
    monkeypatch.setattr(sessions_mod, "get_caps", lambda: RuntimeCaps(llm=caps_llm))

    agent = SimpleNamespace(id="agent-1", bundle_location="/bundle", session_id=None)
    conv = SimpleNamespace(agent_id="agent-1")

    await sessions_mod._run_compact_locked(
        session_id="sess-1",
        conv=conv,
        agent_store=_FakeAgentStore(agent),
        agent_cache=_FakeAgentCache(spec),
    )
    return captured["llm_config"]


async def test_falls_back_to_caps_llm_when_spec_has_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """spec.llm and executor.model both None -> use the server-level caps LLM."""
    caps_llm = LLMConfig(
        model="gpt-5.4-mini",
        connection={"base_url": "https://openai.kainotomic.com/v1", "api_key": "resolved-key"},
    )
    resolved = await _resolve_llm_config(
        monkeypatch,
        spec=_omnigent_spec(llm=None, executor_model=None),
        caps_llm=caps_llm,
    )
    assert resolved.model == "gpt-5.4-mini"
    assert resolved.connection == {
        "base_url": "https://openai.kainotomic.com/v1",
        "api_key": "resolved-key",
    }


async def test_caps_llm_connection_env_ref_expanded_server_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ${VAR} in the caps LLM connection is expanded in the server process."""
    monkeypatch.setenv("OMNIGENT_COMPACT_TEST_KEY", "sk-server-side-value")
    caps_llm = LLMConfig(
        model="gpt-5.4-mini",
        connection={
            "base_url": "https://openai.kainotomic.com/v1",
            "api_key": "${OMNIGENT_COMPACT_TEST_KEY}",
        },
    )
    resolved = await _resolve_llm_config(
        monkeypatch,
        spec=_omnigent_spec(llm=None, executor_model=None),
        caps_llm=caps_llm,
    )
    assert resolved.model == "gpt-5.4-mini"
    # Expanded server-side; no unresolved ${...} ref survives.
    assert resolved.connection is not None
    assert "${" not in resolved.connection["api_key"]
    assert resolved.connection["api_key"] == "sk-server-side-value"


async def test_spec_llm_preferred_over_caps_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """spec.llm wins even when a caps LLM is also configured (backward compat)."""
    spec_llm = LLMConfig(model="anthropic/claude-opus-4-8")
    caps_llm = LLMConfig(model="gpt-5.4-mini")
    resolved = await _resolve_llm_config(
        monkeypatch,
        spec=_omnigent_spec(llm=spec_llm, executor_model=None),
        caps_llm=caps_llm,
    )
    assert resolved.model == "anthropic/claude-opus-4-8"


async def test_executor_model_preferred_over_caps_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """executor.model wins over caps LLM when spec.llm is None (backward compat)."""
    caps_llm = LLMConfig(model="gpt-5.4-mini")
    resolved = await _resolve_llm_config(
        monkeypatch,
        spec=_omnigent_spec(
            llm=None,
            executor_model="openai/gpt-5.4",
            executor_connection={"base_url": "https://example.test/v1"},
        ),
        caps_llm=caps_llm,
    )
    assert resolved.model == "openai/gpt-5.4"
    assert resolved.connection == {"base_url": "https://example.test/v1"}


async def test_raises_when_no_model_anywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing configured (no spec.llm, no executor.model, no caps.llm) -> raise."""
    with pytest.raises(OmnigentError, match="Compaction requires a configured LLM model"):
        await _resolve_llm_config(
            monkeypatch,
            spec=_omnigent_spec(llm=None, executor_model=None),
            caps_llm=None,
        )
