"""Regression guard for the polly example's worker-boot safety.

The polly bundle (root + its ``claude_code`` sub-agent) must parse with
``expand_env=True`` even when ``OMNIGENT_KAINOTOMIC_API_KEY`` is UNSET.

This is the exact condition that broke sub-agent boot: the runner re-parses
the whole bundle with ``expand_env=True`` at every sub-agent dispatch, in an
environment that deliberately strips deployment secrets via an allowlist
(``host/connect.py``). A spec that carried a
``${OMNIGENT_KAINOTOMIC_API_KEY}`` connection env-ref (as PR #3 added under
``executor.connection``) raised "Unresolved environment variable" at that
parse, failing boot. The summarizer model + credential now resolve
server-side from the deployment's ``llm:`` config (RuntimeCaps.llm) in
``_run_compact_locked``, so NO spec carries that env-ref — see
``tests/server/test_compact_model_resolution.py``.

Parse-only check, so it runs in the default suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.spec.parser import parse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POLLY_DIR = _REPO_ROOT / "examples" / "polly"
_CLAUDE_CODE_DIR = _POLLY_DIR / "agents" / "claude_code"


def test_polly_specs_parse_with_secret_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both polly specs parse with expand_env=True when the secret is UNSET.

    Simulates the runner's secret-stripped boot environment. Fails on the
    pre-fix state (PR #3's ``${OMNIGENT_KAINOTOMIC_API_KEY}`` env-ref raises
    "Unresolved environment variable" here); must succeed now.
    """
    monkeypatch.delenv("OMNIGENT_KAINOTOMIC_API_KEY", raising=False)

    # Root spec — recursively parses every sub-agent too. No raise = boot-safe.
    root = parse(_POLLY_DIR, expand_env=True)
    assert root.name == "polly"
    assert {sub.name for sub in root.sub_agents} >= {"claude_code"}

    # The claude_code sub-agent parses on its own as well.
    claude_code = parse(_CLAUDE_CODE_DIR, expand_env=True)
    assert claude_code.name == "claude_code"


def test_polly_specs_carry_no_compaction_env_ref() -> None:
    """No polly spec pins a summarizer model/connection or a secret env-ref.

    The summarizer is resolved server-side (caps.llm), so the bundle stays
    free of the ``${OMNIGENT_KAINOTOMIC_API_KEY}`` ref that broke boot.
    """
    for cfg in (_POLLY_DIR / "config.yaml", _CLAUDE_CODE_DIR / "config.yaml"):
        text = cfg.read_text()
        assert "OMNIGENT_KAINOTOMIC_API_KEY" not in text, (
            f"{cfg} must not carry the compaction secret env-ref — the "
            f"summarizer resolves server-side from caps.llm."
        )

    # No executor.model / executor.connection / compaction block on either spec.
    for spec_dir in (_POLLY_DIR, _CLAUDE_CODE_DIR):
        spec = parse(spec_dir, expand_env=False)
        assert spec.executor.model is None, (
            f"{spec_dir}: executor.model must be unset (summarizer is server-side)."
        )
        assert spec.executor.connection is None, (
            f"{spec_dir}: executor.connection must be unset (no in-spec secret)."
        )
        assert spec.compaction is None, f"{spec_dir}: no compaction block expected."
