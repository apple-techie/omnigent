"""Pure Codex app-server item / turn translation helpers.

Factored out of :mod:`omnigent.codex_native_forwarder` so the same normalization
— built-in tool-call extraction, turn-level terminal-error classification, resume
turn → Omnigent status derivation, stable turn/source ids, and the
``thread/resume`` item iteration — can be reused by session adoption without a
second, simplified parser. Every function here is side-effect-free (no IO, no DB,
no network); it maps native Codex payload dicts to normalized values. The
forwarder re-exports these names so existing callers and tests keep importing them
from :mod:`omnigent.codex_native_forwarder` unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


# Turn-error surfacing. A failed Codex turn arrives as ``turn/completed``
# (or ``turn/failed``) with ``turn.status == "failed"`` and a ``turn.error``
# object ``{message, codexErrorInfo?, additionalDetails?}``; keying status off
# the method alone mapped such turns to ``idle`` — a "silent success". The
# forwarder inspects ``turn.status``/``turn.error``, forces ``failed``, and
# surfaces the reason. As a fallback it also catches an ``error`` ThreadItem in
# ``turn.items``: both shapes exist in the app-server type system and the wire
# shape varies by version, so detecting either keeps the fix robust.
#
# ``codexErrorInfo`` is the app-server's structured classification (e.g.
# ``unauthorized``, ``usage_limit_exceeded``); auth-class values get a re-auth
# hint. httpStatusCode 401/403 is treated as auth too. Values are stored and
# compared case-insensitively: the app-server enum serializes as lowercase
# snake_case (``unauthorized``), but older/alternate spellings (``Unauthorized``)
# are matched too.
_CODEX_ERROR_ITEM_TYPE = "error"
_CODEX_AUTH_ERROR_INFO = frozenset({"unauthorized"})
_CODEX_AUTH_HTTP_STATUS = frozenset({401, 403})
# Message-substring fallback for app-server versions that omit codexErrorInfo.
# Surface-only, so recall is favored over precision: a false positive only
# appends a re-auth hint to an already-failed turn.
_CODEX_AUTH_ERROR_FRAGMENTS = (
    "401",
    "403",
    "unauthorized",
    "authentication",
    "not logged in",
    "not authenticated",
    "log in",
    "login",
    "sign in",
    "re-authenticate",
    "reauthenticate",
    "credentials",
    "access token",
    "token expired",
    "expired token",
    "session expired",
    "api key",
)
_CODEX_ERROR_KIND_AUTH = "auth"
_CODEX_ERROR_KIND_GENERIC = "generic"
_CODEX_REAUTH_HINT = "Codex needs you to re-authenticate. Run `codex login` and retry."


@dataclass(frozen=True)
class _CodexToolCall:
    """
    Normalized view of one completed Codex built-in tool call.

    :param call_id: Codex item id reused as the Omnigent call id, e.g.
        ``"call_abc"``.
    :param name: Omnigent function-call name, e.g. ``"shell"``.
    :param arguments: Tool arguments dict, e.g. ``{"command": "pwd"}``.
    :param output: Tool result text rendered as the
        ``function_call_output``, e.g. ``"/repo\n"``.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]
    output: str


@dataclass(frozen=True)
class _CodexTerminalError:
    """
    A turn-level failure surfaced from a Codex turn.

    Produced by :func:`_terminal_error_from_turn` from ``turn.error`` or an
    ``error`` ThreadItem. Forces the turn's Omnigent status to ``failed`` and
    lets :func:`_post_turn_status_edge` surface the reason (and a re-auth hint
    for auth-classified errors).

    :param message: Human-readable error text, e.g.
        ``"401 Unauthorized: ChatGPT login expired"``.
    :param kind: Classification, either ``"auth"`` or ``"generic"``.
    """

    message: str
    kind: str

    @property
    def is_auth(self) -> bool:
        """:returns: ``True`` when the error was classified as auth-related."""
        return self.kind == _CODEX_ERROR_KIND_AUTH


def _classify_codex_error(error: dict[str, Any], message: str) -> str:
    """
    Classify a Codex ``turn.error`` / ``error`` item as auth-related or generic.

    Prefers the structured ``codexErrorInfo`` (an ``unauthorized`` variant,
    case-insensitive, or an httpStatusCode of 401/403); falls back to substring
    matching against :data:`_CODEX_AUTH_ERROR_FRAGMENTS` for versions/shapes
    that omit it.

    :param error: The ``turn.error`` object.
    :param message: Its already-extracted message text.
    :returns: :data:`_CODEX_ERROR_KIND_AUTH` or
        :data:`_CODEX_ERROR_KIND_GENERIC`.
    """
    info = error.get("codexErrorInfo")
    variant: str | None = None
    http_status: Any = None
    if isinstance(info, str):
        variant = info
    elif isinstance(info, dict):
        variant = info.get("type") or info.get("kind") or info.get("variant")
        http_status = info.get("httpStatusCode")
    variant_is_auth = variant is not None and variant.lower() in _CODEX_AUTH_ERROR_INFO
    if variant_is_auth or http_status in _CODEX_AUTH_HTTP_STATUS:
        return _CODEX_ERROR_KIND_AUTH
    lowered = message.lower()
    if any(fragment in lowered for fragment in _CODEX_AUTH_ERROR_FRAGMENTS):
        return _CODEX_ERROR_KIND_AUTH
    return _CODEX_ERROR_KIND_GENERIC


def _error_payload_message(payload: dict[str, Any]) -> str:
    """
    Extract a non-empty message from a Codex ``turn.error`` or ``error`` item.

    Both shapes have surfaced the text under a few keys across app-server
    versions; reads the first non-empty one, falling back to a stable string
    so the surfaced error is never blank.

    :param payload: A ``turn.error`` object or an ``error`` ThreadItem.
    :returns: Non-empty error text.
    """
    for key in ("message", "error", "text", "detail"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Codex turn ended with an unspecified error."


def _error_item_from_turn(turn: dict[str, Any]) -> dict[str, Any] | None:
    """
    Return the first ``error`` ThreadItem in ``turn.items``, if any.

    :param turn: A Codex turn object.
    :returns: The first item whose ``type`` is :data:`_CODEX_ERROR_ITEM_TYPE`,
        or ``None``.
    """
    items = turn.get("items")
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and item.get("type") == _CODEX_ERROR_ITEM_TYPE:
            return item
    return None


def _terminal_error_from_turn(params: dict[str, Any]) -> _CodexTerminalError | None:
    """
    Return the turn-level failure carried by a Codex turn, if any.

    Prefers ``turn.error`` (the protocol's ``TurnError`` on a failed turn) and
    falls back to an ``error`` ThreadItem in ``turn.items`` — both shapes exist
    in the app-server type system and the wire shape varies by version. Single
    source of truth reused by the live terminal edge and the ``thread/resume``
    parity path.

    :param params: Codex turn params, e.g. a ``turn/completed`` payload or a
        single ``thread/resume`` turn wrapped as ``{"turn": <turn>}``.
    :returns: The classified terminal error, or ``None`` when the turn did not
        fail.
    """
    turn = params.get("turn")
    if not isinstance(turn, dict):
        return None
    payload = turn.get("error")
    if not isinstance(payload, dict):
        payload = _error_item_from_turn(turn)
    if payload is None:
        return None
    message = _error_payload_message(payload)
    return _CodexTerminalError(message=message, kind=_classify_codex_error(payload, message))


@dataclass(frozen=True)
class _CodexTurnStatusEdge:
    """
    Omnigent session-status edge derived from Codex turn lifecycle state.

    :param status: Omnigent session status, e.g. ``"running"`` or ``"idle"``.
    :param turn_id: Codex turn id that caused the edge, e.g.
        ``"turn_abc123"``.
    :param source: Lifecycle source that produced the edge, e.g.
        ``"turn/started"``.
    :param error: Turn-level error forcing this edge to ``failed``,
        or ``None`` for ordinary lifecycle edges. Surfaced as the status
        output by :func:`_post_turn_status_edge`.
    """

    status: str
    turn_id: str | None
    source: str
    error: _CodexTerminalError | None = None


# Codex ``item/completed`` item types that represent a built-in tool call.
# Each maps to a builder that extracts a normalized :class:`_CodexToolCall`.
# ``_TOOL_ITEM_BUILDERS`` is populated after the builders are defined.
_ToolItemBuilder = Callable[[str, dict[str, Any]], "_CodexToolCall | None"]


def _omnigent_status_from_resume_turn(turn: dict[str, Any]) -> str | None:
    """
    Convert an explicit Codex resume turn status to Omnigent session status.

    Applies the same ``turn.error`` check as the live terminal path
    (:func:`_terminal_turn_status_edge`) so a resumed turn that carried an
    error maps to ``failed`` even if its recorded status is not — the
    resume-path side of the "silent success" fix.

    :param turn: Codex resume turn object, e.g.
        ``{"id": "turn_123", "status": "completed"}``.
    :returns: Omnigent status literal for terminal turns, or ``None`` for active
        or unrecognized statuses.
    """
    # A ``turn.error`` forces ``failed`` regardless of the recorded status.
    if _terminal_error_from_turn({"turn": turn}) is not None:
        return "failed"
    status = turn.get("status")
    if isinstance(status, dict):
        status = status.get("type") or status.get("status")
    if status in {"completed", "interrupted", "cancelled", "canceled"}:
        return "idle"
    if status in {"failed", "errored"}:
        return "failed"
    return None


def _codex_tool_call_from_item(item: dict[str, Any]) -> _CodexToolCall | None:
    """
    Translate a completed Codex tool item into a normalized tool call.

    :param item: Codex tool item from an ``item/completed`` notification,
        e.g. ``{"type": "commandExecution", "id": "call_abc", ...}``.
    :returns: Normalized tool call, or ``None`` for a malformed item that
        should be dropped rather than mirrored with invented fields.
    """
    call_id = item.get("id")
    item_type = item.get("type")
    if not isinstance(call_id, str) or not call_id:
        _logger.warning("Codex tool item missing string id: type=%s", item_type)
        return None
    builder = _TOOL_ITEM_BUILDERS.get(item_type) if isinstance(item_type, str) else None
    if builder is None:
        return None
    return builder(call_id, item)


# Codex runs each model-issued shell command inside its OWN bwrap command
# sandbox. In a hardened container that disallows unprivileged user namespaces,
# that sandbox cannot start and every command hard-fails with this raw bwrap
# error, with no hint at how to recover. Detect the marker and append actionable
# guidance so a top-level session degrades with direction instead of an opaque
# failure. The codex
# ``--approval-mode`` presets do NOT disable this sandbox — only the "Full
# access" preset's ``danger-full-access`` (or a config ``sandbox_mode``) does.
_CODEX_SANDBOX_NAMESPACE_ERROR_MARKER = "No permissions to create new namespace"
_CODEX_SANDBOX_BYPASS_GUIDANCE = (
    "Omnigent: Codex's command sandbox could not start because this container "
    "disallows unprivileged user namespaces, so the command did not run. To run "
    'shell commands here, start a new Codex session with the "Full access" '
    "approval preset (New chat → Advanced settings), or set "
    'sandbox_mode = "danger-full-access" in ~/.codex/config.toml on the runner.'
)


def _augment_sandbox_namespace_error(output_text: str) -> str:
    """Append recovery guidance when a Codex shell command failed because its
    own command sandbox could not start (no unprivileged user namespaces).

    Returns *output_text* unchanged when the bwrap-namespace marker is absent,
    so ordinary command output is never altered. See issue #657.

    :param output_text: Aggregated command output, any exit-code suffix already
        appended, e.g. ``"bwrap: No permissions ...\\n[exit code: 1]"``.
    :returns: The output with a trailing guidance paragraph, or unchanged.
    """
    if _CODEX_SANDBOX_NAMESPACE_ERROR_MARKER not in output_text:
        return output_text
    return f"{output_text}\n\n{_CODEX_SANDBOX_BYPASS_GUIDANCE}"


def _command_execution_tool_call(call_id: str, item: dict[str, Any]) -> _CodexToolCall | None:
    """
    Build a tool call from a Codex ``commandExecution`` item.

    :param call_id: Codex item id, e.g. ``"call_abc"``.
    :param item: Codex ``commandExecution`` item, e.g.
        ``{"command": "/bin/zsh -lc 'pwd'", "cwd": "/repo",
        "aggregatedOutput": "/repo\n", "exitCode": 0}``.
    :returns: Normalized tool call, or ``None`` when the command is
        missing.
    """
    command = item.get("command")
    if not isinstance(command, str) or not command:
        _logger.warning("Codex commandExecution missing command: call_id=%s", call_id)
        return None
    arguments: dict[str, Any] = {"command": command}
    cwd = item.get("cwd")
    if isinstance(cwd, str) and cwd:
        arguments["cwd"] = cwd
    output = item.get("aggregatedOutput")
    # A command that prints nothing (e.g. ``touch x``) legitimately has no
    # aggregated output; Codex reports that as "" or null. AP's
    # function_call_output requires a string, so "" is the faithful
    # representation of "no output captured" here — not an invented default.
    output_text = output if isinstance(output, str) else ""
    exit_code = item.get("exitCode")
    # Codex reports a non-zero exit separately from stdout/stderr; surface
    # it inline so a failed command does not look successful in the UI.
    if isinstance(exit_code, int) and exit_code != 0:
        suffix = f"[exit code: {exit_code}]"
        output_text = f"{output_text}\n{suffix}" if output_text else suffix
    # Turn codex's opaque "sandbox can't start" bwrap failure into actionable
    # recovery guidance; a no-op for any other output.
    output_text = _augment_sandbox_namespace_error(output_text)
    return _CodexToolCall(call_id=call_id, name="shell", arguments=arguments, output=output_text)


def _file_change_tool_call(call_id: str, item: dict[str, Any]) -> _CodexToolCall | None:
    """
    Build a tool call from a Codex ``fileChange`` item.

    :param call_id: Codex item id, e.g. ``"call_abc"``.
    :param item: Codex ``fileChange`` item, e.g.
        ``{"changes": [{"path": "/repo/x.py", "kind": {"type": "add"},
        "diff": "print('hi')\n"}], "status": "completed"}``.
    :returns: Normalized tool call, or ``None`` when no changes are
        present.
    """
    changes = item.get("changes")
    if not isinstance(changes, list) or not changes:
        _logger.warning("Codex fileChange missing changes: call_id=%s", call_id)
        return None
    summary_lines: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        path = change.get("path")
        kind = change.get("kind")
        kind_type = kind.get("type") if isinstance(kind, dict) else None
        label = kind_type if isinstance(kind_type, str) and kind_type else "change"
        summary_lines.append(f"{label} {path}")
    output_text = "\n".join(summary_lines)
    return _CodexToolCall(
        call_id=call_id,
        name="apply_patch",
        arguments={"changes": changes},
        output=output_text,
    )


def _web_search_tool_call(call_id: str, item: dict[str, Any]) -> _CodexToolCall | None:
    """
    Build a tool call from a Codex ``webSearch`` item.

    Codex does not surface the search results, so the queries it ran are
    the only result data available and are used as the output text.

    :param call_id: Codex item id, e.g. ``"ws_abc"``.
    :param item: Codex ``webSearch`` item, e.g.
        ``{"query": "python latest version",
        "action": {"type": "search", "queries": ["python latest"]}}``.
    :returns: Normalized tool call, or ``None`` when no query is present.
    """
    query = item.get("query")
    action = item.get("action")
    queries = action.get("queries") if isinstance(action, dict) else None
    query_list = [q for q in queries if isinstance(q, str)] if isinstance(queries, list) else []
    if not query_list and isinstance(query, str) and query:
        query_list = [query]
    if not query_list:
        _logger.warning("Codex webSearch missing query: call_id=%s", call_id)
        return None
    return _CodexToolCall(
        call_id=call_id,
        name="web_search",
        arguments={"query": query_list[0]},
        output="\n".join(query_list),
    )


def _image_view_tool_call(call_id: str, item: dict[str, Any]) -> _CodexToolCall | None:
    """
    Build a tool call from a Codex ``imageView`` item.

    Codex emits an ``imageView`` item when the model opens a local image
    (e.g. a screenshot on disk) to look at it. The only datum is the
    absolute path, so it becomes both the argument and the mirrored
    output — the web UI cannot read a runner-local path, so the path is
    the faithful record of which image was viewed.

    :param call_id: Codex item id, e.g. ``"img_abc"``.
    :param item: Codex ``imageView`` item, e.g.
        ``{"type": "imageView", "id": "img_abc", "path": "/repo/shot.png"}``.
    :returns: Normalized tool call, or ``None`` when the path is missing.
    """
    path = item.get("path")
    if not isinstance(path, str) or not path:
        _logger.warning("Codex imageView missing path: call_id=%s", call_id)
        return None
    return _CodexToolCall(
        call_id=call_id,
        name="view_image",
        arguments={"path": path},
        output=path,
    )


def _image_generation_tool_call(call_id: str, item: dict[str, Any]) -> _CodexToolCall | None:
    """
    Build a tool call from a Codex ``imageGeneration`` item.

    Codex emits an ``imageGeneration`` item when the model generates an
    image. The raw ``result`` payload (base64 image bytes) is deliberately
    NOT mirrored — the web UI has no assistant-side image rendering and a
    multi-megabyte base64 string would only bloat the transcript. Instead
    the card carries the human-meaningful metadata: the revised prompt as
    the argument and the status plus on-disk save path as the output.

    :param call_id: Codex item id, e.g. ``"imggen_abc"``.
    :param item: Codex ``imageGeneration`` item, e.g.
        ``{"type": "imageGeneration", "id": "imggen_abc",
        "status": "completed", "revisedPrompt": "a red bicycle",
        "result": "<base64>", "savedPath": "/repo/out.png"}``.
    :returns: Normalized tool call, or ``None`` when the status is missing.
    """
    status = item.get("status")
    if not isinstance(status, str) or not status:
        _logger.warning("Codex imageGeneration missing status: call_id=%s", call_id)
        return None
    arguments: dict[str, Any] = {}
    revised_prompt = item.get("revisedPrompt")
    if isinstance(revised_prompt, str) and revised_prompt:
        arguments["revised_prompt"] = revised_prompt
    output_lines = [f"status: {status}"]
    saved_path = item.get("savedPath")
    if isinstance(saved_path, str) and saved_path:
        output_lines.append(f"saved to {saved_path}")
    return _CodexToolCall(
        call_id=call_id,
        name="generate_image",
        arguments=arguments,
        output="\n".join(output_lines),
    )


# Codex built-in tool item types this forwarder mirrors into Omnigent history.
# ``mcpToolCall`` is intentionally absent: its event shape has not been
# verified, so it is logged-but-skipped rather than mirrored with guessed
# fields. Add it here once its real shape is captured.
_TOOL_ITEM_BUILDERS: dict[str, _ToolItemBuilder] = {
    "commandExecution": _command_execution_tool_call,
    "fileChange": _file_change_tool_call,
    "webSearch": _web_search_tool_call,
    "imageView": _image_view_tool_call,
    "imageGeneration": _image_generation_tool_call,
}
_TOOL_ITEM_TYPES = frozenset(_TOOL_ITEM_BUILDERS)


def _turn_id_from_payload(payload: object) -> str | None:
    """
    Extract a turn id from a Codex payload.

    :param payload: Codex notification params or nested turn object.
    :returns: Turn id, or ``None`` when absent.
    """
    if not isinstance(payload, dict):
        return None
    value = payload.get("id") or payload.get("turnId")
    return value if isinstance(value, str) and value else None


def _source_id(params: dict[str, Any], item: dict[str, Any]) -> str:
    """
    Build a stable per-record label for one Codex item.

    Only used for debug-log correlation — it is not sent to the server
    and is not a dedup key (the server persists external items with a
    random primary key).

    :param params: Codex notification params.
    :param item: Codex item payload.
    :returns: Record label, e.g. ``"turn_abc:item_xyz"``.
    """
    turn_id = params.get("turnId")
    item_id = item.get("id")
    left = turn_id if isinstance(turn_id, str) and turn_id else "thread"
    right = item_id if isinstance(item_id, str) and item_id else "item"
    return f"{left}:{right}"


def iter_resume_items(response: dict[str, Any]) -> Iterator[tuple[str | None, dict[str, Any]]]:
    """Yield ``(thread_id, item/completed event)`` for each item in a resume response.

    Pure iteration over a Codex ``thread/resume`` response envelope: it walks
    ``result.thread.turns[*].items[*]`` and wraps each item as the same
    ``item/completed`` event the live stream delivers, so a caller can feed each
    one through the normal event handler and dedup gate. Yields nothing when the
    envelope lacks a well-formed ``result.thread.turns`` list.

    :param response: Codex ``thread/resume`` response envelope.
    :returns: Iterator of ``(thread_id, event)`` pairs, in turn/item order.
    """
    result = response.get("result")
    if not isinstance(result, dict):
        return
    thread = result.get("thread")
    if not isinstance(thread, dict):
        return
    turns = thread.get("turns")
    if not isinstance(turns, list):
        return
    thread_id = thread.get("id")
    thread_id = thread_id if isinstance(thread_id, str) and thread_id else None
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        turn_id = _turn_id_from_payload(turn)
        items = turn.get("items")
        if not turn_id or not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            yield (
                thread_id,
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": thread_id,
                        "turnId": turn_id,
                        "item": item,
                    },
                },
            )
