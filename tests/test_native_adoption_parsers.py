"""Characterization tests for the shared native-session parsers.

Slice 1 of session adoption extracted the pure ``native record -> Omnigent item``
translation logic out of the three live forwarders into standalone modules
(:mod:`omnigent.hermes_native_items`, :mod:`omnigent.claude_transcript_parser`,
:mod:`omnigent.codex_native_items`). These tests import the helpers *directly from
the new modules* — the way session adoption will — and pin their exact output for
representative native records, locking in behavior byte-for-byte with what the
inline forwarder path produced. The existing forwarder suites still exercise the
same helpers through the forwarders; this file is the standalone contract.
"""

from __future__ import annotations

import json

from omnigent import claude_transcript_parser as claude
from omnigent import codex_native_items as codex
from omnigent import hermes_native_items as hermes

# --------------------------------------------------------------------------- #
# Hermes: messages-row -> mirror items
# --------------------------------------------------------------------------- #


def test_hermes_user_message_item() -> None:
    items = hermes.message_to_items(7, "user", "hello world", None, None, None, "ag")
    assert items == [
        hermes.HermesMirrorItem(
            msg_id=7,
            item_type="message",
            item_data={"role": "user", "content": [{"type": "input_text", "text": "hello world"}]},
            response_id="hermes:7",
        )
    ]


def test_hermes_user_strips_attachment_marker_and_summarizes_skill() -> None:
    attached = hermes.message_to_items(
        1, "user", "look [Attached: /x/y.png] here", None, None, None, "ag"
    )
    assert attached[0].item_data["content"][0]["text"] == "look  here"

    skill = hermes.message_to_items(
        2,
        "user",
        '[IMPORTANT: The user has invoked the "deep-research" skill and ...',
        None,
        None,
        None,
        "ag",
    )
    assert skill[0].item_data["content"][0]["text"] == "/deep-research"


def test_hermes_assistant_tool_calls_then_text() -> None:
    tool_calls = json.dumps(
        [{"call_id": "c1", "function": {"name": "shell", "arguments": '{"cmd":"ls"}'}}]
    )
    items = hermes.message_to_items(9, "assistant", "done", tool_calls, None, None, "ag")
    assert [i.item_type for i in items] == ["function_call", "message"]
    assert items[0].item_data == {
        "agent": "ag",
        "name": "shell",
        "arguments": '{"cmd":"ls"}',
        "call_id": "c1",
    }
    assert items[1].item_data == {
        "role": "assistant",
        "agent": "ag",
        "content": [{"type": "output_text", "text": "done"}],
    }


def test_hermes_tool_row_and_skips() -> None:
    tool = hermes.message_to_items(3, "tool", "output text", None, "call-1", "shell", "ag")
    assert tool == [
        hermes.HermesMirrorItem(
            msg_id=3,
            item_type="function_call_output",
            item_data={"call_id": "call-1", "output": "output text"},
            response_id="hermes:3",
        )
    ]
    # Empty user text, unknown role, and tool row without id all skip.
    assert hermes.message_to_items(4, "user", "", None, None, None, "ag") == []
    assert hermes.message_to_items(5, "system", "x", None, None, None, "ag") == []
    assert hermes.message_to_items(6, "tool", "x", None, "", None, "ag") == []


def test_hermes_assistant_row_has_tool_calls() -> None:
    assert hermes.assistant_row_has_tool_calls(None) is False
    assert hermes.assistant_row_has_tool_calls("") is False
    assert hermes.assistant_row_has_tool_calls("[]") is False
    assert hermes.assistant_row_has_tool_calls("not json") is False
    assert hermes.assistant_row_has_tool_calls(json.dumps([{"id": "c1"}])) is True


# --------------------------------------------------------------------------- #
# Claude: transcript entry -> conversation items
# --------------------------------------------------------------------------- #


def test_claude_assistant_text_and_tool_use() -> None:
    entry = {
        "uuid": "u-assist",
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-8",
            "content": [
                {"type": "text", "text": "Working on it"},
                {"type": "tool_use", "id": "call_1", "name": "Bash", "input": {"command": "pwd"}},
            ],
        },
    }
    rid, items = claude._transcript_items_from_entry(
        entry,
        line_number=1,
        record_offset=0,
        agent_name="claude-native-ui",
        current_response_id=None,
    )
    assert [i.item_type for i in items] == ["message", "function_call"]
    assert items[0].data == {
        "role": "assistant",
        "agent": "claude-native-ui",
        "content": [{"type": "output_text", "text": "Working on it"}],
    }
    assert items[1].data == {
        "agent": "claude-native-ui",
        "name": "Bash",
        "arguments": '{"command":"pwd"}',
        "call_id": "call_1",
    }
    # Both items share the response id derived from the record source key.
    assert rid == items[0].response_id == items[1].response_id


def test_claude_user_message_and_tool_result() -> None:
    user = {"uuid": "u1", "message": {"role": "user", "content": "hi there"}}
    _, items = claude._transcript_items_from_entry(
        user, line_number=1, record_offset=0, agent_name="ag", current_response_id=None
    )
    assert items[0].item_type == "message"
    assert items[0].data == {
        "role": "user",
        "content": [{"type": "input_text", "text": "hi there"}],
    }

    tool_result = {
        "uuid": "u2",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "/repo"}],
        },
    }
    _, out = claude._transcript_items_from_entry(
        tool_result, line_number=2, record_offset=10, agent_name="ag", current_response_id="resp_x"
    )
    assert out[0].item_type == "function_call_output"
    assert out[0].data == {"call_id": "call_1", "output": "/repo"}
    assert out[0].response_id == "resp_x"


def test_claude_sidechain_dropped_by_default() -> None:
    entry = {"isSidechain": True, "message": {"role": "assistant", "content": "hidden"}}
    rid, items = claude._transcript_items_from_entry(
        entry, line_number=1, record_offset=0, agent_name="ag", current_response_id="resp_keep"
    )
    assert items == []
    assert rid == "resp_keep"
    # With include_sidechains the same record yields the assistant message.
    _, kept = claude._transcript_items_from_entry(
        entry,
        line_number=1,
        record_offset=0,
        agent_name="ag",
        current_response_id=None,
        include_sidechains=True,
    )
    assert kept[0].item_type == "message"


def test_claude_source_key_and_response_id_are_deterministic() -> None:
    assert claude._transcript_source_key({"uuid": "abc"}, 1, 99) == "abc"
    assert claude._transcript_source_key({}, 1, 99) == "byte-99"
    assert claude._transcript_source_key({}, 5, None) == "line-5"
    rid = claude._response_id_from_source("abc")
    assert rid.startswith("resp_claude_") and claude._response_id_from_source("abc") == rid


def test_claude_read_from_offset_over_a_file(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    records = [
        {"uuid": "r1", "message": {"role": "user", "content": "first"}},
        {"uuid": "r2", "message": {"role": "assistant", "model": "m", "content": "reply"}},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    result = claude.read_transcript_items_from_offset(path, 0, start_line=0, agent_name="ag")
    assert [i.item_type for i in result.items] == ["message", "message"]
    assert result.items[0].data["role"] == "user"
    assert result.items[1].data["role"] == "assistant"
    assert result.latest_model == "m"
    assert result.byte_offset == path.stat().st_size


# --------------------------------------------------------------------------- #
# Codex: item normalization, terminal status, resume iteration
# --------------------------------------------------------------------------- #


def test_codex_command_execution_tool_call() -> None:
    tc = codex._codex_tool_call_from_item(
        {
            "type": "commandExecution",
            "id": "c1",
            "command": "pwd",
            "cwd": "/repo",
            "aggregatedOutput": "/repo\n",
            "exitCode": 0,
        }
    )
    assert tc == codex._CodexToolCall(
        call_id="c1", name="shell", arguments={"command": "pwd", "cwd": "/repo"}, output="/repo\n"
    )
    # Non-zero exit code is surfaced inline.
    failed = codex._codex_tool_call_from_item(
        {
            "type": "commandExecution",
            "id": "c2",
            "command": "false",
            "aggregatedOutput": "",
            "exitCode": 1,
        }
    )
    assert failed.output == "[exit code: 1]"


def test_codex_web_search_and_image_tool_calls() -> None:
    ws = codex._codex_tool_call_from_item(
        {"type": "webSearch", "id": "w1", "action": {"queries": ["a", "b"]}}
    )
    assert ws.name == "web_search" and ws.arguments == {"query": "a"} and ws.output == "a\nb"

    view = codex._codex_tool_call_from_item({"type": "imageView", "id": "i1", "path": "/x.png"})
    assert view.name == "view_image" and view.output == "/x.png"

    # Unknown / malformed items drop rather than mirror invented fields.
    assert codex._codex_tool_call_from_item({"type": "mcpToolCall", "id": "m1"}) is None
    assert codex._codex_tool_call_from_item({"type": "commandExecution", "id": "c3"}) is None


def test_codex_terminal_error_classification() -> None:
    auth = codex._terminal_error_from_turn({"turn": {"error": {"message": "401 Unauthorized"}}})
    assert auth is not None and auth.kind == "auth" and auth.is_auth is True

    generic = codex._terminal_error_from_turn(
        {"turn": {"error": {"message": "disk full", "codexErrorInfo": {"type": "io"}}}}
    )
    assert generic is not None and generic.kind == "generic"

    # Falls back to an ``error`` ThreadItem in turn.items.
    item_err = codex._terminal_error_from_turn(
        {"turn": {"items": [{"type": "error", "message": "boom"}]}}
    )
    assert item_err is not None and item_err.message == "boom"

    assert codex._terminal_error_from_turn({"turn": {"items": []}}) is None


def test_codex_omnigent_status_from_resume_turn() -> None:
    assert codex._omnigent_status_from_resume_turn({"status": "completed"}) == "idle"
    assert codex._omnigent_status_from_resume_turn({"status": "interrupted"}) == "idle"
    assert codex._omnigent_status_from_resume_turn({"status": "failed"}) == "failed"
    # A turn error forces failed even on a "completed" status.
    assert (
        codex._omnigent_status_from_resume_turn({"status": "completed", "error": {"message": "x"}})
        == "failed"
    )
    assert codex._omnigent_status_from_resume_turn({"status": "in_progress"}) is None


def test_codex_turn_id_and_source_id() -> None:
    assert codex._turn_id_from_payload({"id": "turn_1"}) == "turn_1"
    assert codex._turn_id_from_payload({"turnId": "turn_2"}) == "turn_2"
    assert codex._turn_id_from_payload({}) is None
    assert codex._source_id({"turnId": "t"}, {"id": "it"}) == "t:it"
    assert codex._source_id({}, {}) == "thread:item"


def test_codex_iter_resume_items() -> None:
    response = {
        "result": {
            "thread": {
                "id": "thread_9",
                "turns": [
                    {
                        "id": "turn_1",
                        "items": [
                            {"type": "agentMessage", "id": "a1", "text": "hi"},
                            {"type": "commandExecution", "id": "c1", "command": "ls"},
                        ],
                    },
                    {
                        "id": "turn_2",
                        "items": [{"type": "agentMessage", "id": "a2", "text": "bye"}],
                    },
                ],
            }
        }
    }
    events = list(codex.iter_resume_items(response))
    assert [tid for tid, _ in events] == ["thread_9", "thread_9", "thread_9"]
    assert events[0][1] == {
        "method": "item/completed",
        "params": {
            "threadId": "thread_9",
            "turnId": "turn_1",
            "item": {"type": "agentMessage", "id": "a1", "text": "hi"},
        },
    }
    # A malformed envelope yields nothing (no terminal-status inference).
    assert list(codex.iter_resume_items({"result": {}})) == []
    assert list(codex.iter_resume_items({})) == []
