"""Pure Hermes ``messages``-row → Omnigent conversation-item translation.

This is the parser source of truth for the hermes-native harness, factored out of
:mod:`omnigent.hermes_native_forwarder` so the same conversion can be reused by
session adoption (importing a native Hermes session's history) without a second,
simplified transcript parser. Everything here is side-effect-free: it takes plain
Python values already read from Hermes' SQLite ``state.db`` and returns
:class:`HermesMirrorItem` values. No IO, no polling, no DB, no network.

The live forwarder keeps ownership of the SQLite query + poll loop; it imports
:func:`message_to_items` / :func:`assistant_row_has_tool_calls` and converts each
row it reads. Adoption reads a bounded range and converts it through the same
helpers, so the mirrored chat and the imported history stay byte-identical.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# The executor injects ``[Attached: <path>]`` markers for web-UI attachments
# before pasting into the TUI; strip them from the mirrored bubble (the path is
# an internal bridge detail).
_ATTACHMENT_MARKER_RE = re.compile(r"\[Attached:[^\]]*\]")

# Hermes injects skill content as a user message prefixed with this marker.
# The full skill prompt is not useful in the web UI — replace it with a
# short summary so the chat view stays clean.
_SKILL_INVOKE_RE = re.compile(
    r'^\[IMPORTANT: The user has invoked the "(?P<name>[^"]+)" skill',
)


@dataclass
class HermesMirrorItem:
    """One conversation item ready to POST, plus the message id that produced it."""

    msg_id: int
    item_type: str
    item_data: dict[str, object]
    response_id: str


def message_to_items(
    msg_id: int,
    role: object,
    content: object,
    tool_calls: object,
    tool_call_id: object,
    tool_name: object,  # noqa: ARG001 — reserved for future use (e.g. logging)
    agent_name: str,
) -> list[HermesMirrorItem]:
    """Convert one ``messages`` row to mirror items.

    An assistant row with ``tool_calls`` emits a ``function_call`` item per
    call, followed by a ``message`` item if it also has prose content. A tool
    row emits a ``function_call_output`` item. Returns an empty list to skip.
    """
    if not isinstance(role, str):
        return []
    text = ""
    if isinstance(content, str):
        text = _ATTACHMENT_MARKER_RE.sub("", content).strip()
    response_id = f"hermes:{msg_id}"

    if role == "user":
        if not text:
            return []
        # Hermes injects skill content as a user message — replace with
        # a short summary so the chat view stays readable.
        skill_match = _SKILL_INVOKE_RE.match(text)
        if skill_match:
            text = f"/{skill_match.group('name')}"
        return [
            HermesMirrorItem(
                msg_id=msg_id,
                item_type="message",
                item_data={"role": "user", "content": [{"type": "input_text", "text": text}]},
                response_id=response_id,
            )
        ]

    if role == "assistant":
        items: list[HermesMirrorItem] = []
        # Parse tool_calls JSON — assistant rows may include tool call requests.
        if isinstance(tool_calls, str) and tool_calls:
            try:
                calls = json.loads(tool_calls)
            except (json.JSONDecodeError, ValueError):
                calls = []
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    call_id = call.get("call_id") or call.get("id") or ""
                    func = call.get("function", {})
                    name = func.get("name", "") if isinstance(func, dict) else ""
                    arguments = func.get("arguments", "{}") if isinstance(func, dict) else "{}"
                    if call_id and name:
                        items.append(
                            HermesMirrorItem(
                                msg_id=msg_id,
                                item_type="function_call",
                                item_data={
                                    "agent": agent_name,
                                    "name": name,
                                    "arguments": arguments,
                                    "call_id": call_id,
                                },
                                response_id=response_id,
                            )
                        )
        # Also emit a message item if there's prose content.
        if text:
            items.append(
                HermesMirrorItem(
                    msg_id=msg_id,
                    item_type="message",
                    item_data={
                        "role": "assistant",
                        "agent": agent_name,
                        "content": [{"type": "output_text", "text": text}],
                    },
                    response_id=response_id,
                )
            )
        return items

    if role == "tool":
        # Tool result row — emit function_call_output.
        if isinstance(tool_call_id, str) and tool_call_id:
            output = text or ""
            return [
                HermesMirrorItem(
                    msg_id=msg_id,
                    item_type="function_call_output",
                    item_data={"call_id": tool_call_id, "output": output},
                    response_id=response_id,
                )
            ]
        return []

    return []


def assistant_row_has_tool_calls(tool_calls: object) -> bool:
    """Whether an assistant ``messages`` row carries a non-empty ``tool_calls`` list.

    Hermes writes one ``messages`` row per agentic step (complete, append-only —
    rows are never updated in place, which is why message mirroring keys off
    ``id > last_id``). An assistant row with one or more tool calls means the loop
    continues (a tool result + further assistant step follow); a row with no tool
    calls is the loop's terminal step — the model returning its final answer.
    Mirrors the ``tool_calls`` parsing in :func:`message_to_items`.
    """
    if not isinstance(tool_calls, str) or not tool_calls.strip():
        return False
    try:
        calls = json.loads(tool_calls)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(calls, list) and len(calls) > 0
