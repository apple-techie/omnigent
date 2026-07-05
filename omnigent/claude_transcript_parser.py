"""Pure Claude Code transcript → Omnigent conversation-item parser.

Factored out of :mod:`omnigent.claude_native_bridge` so the same durable
transcript parser — byte-offset cursors, sidechain handling, slash-command and
terminal-command translation, tool-call/result mapping, and per-entry usage/model
extraction — can be reused by session adoption (importing a native Claude session
from its JSONL transcript) without a second, simplified parser.

Everything here is side-effect-free apart from reading the transcript file the
caller names: it takes native Claude JSONL records and returns
:class:`ClaudeTranscriptItem` values. No polling, no DB, no network, no bridge
state. :mod:`omnigent.claude_native_bridge` re-exports these names so existing
callers and tests keep importing them from the bridge unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaudeTranscriptItem:
    """
    One Omnigent conversation item parsed from Claude's JSONL log.

    :param source_id: Stable idempotency key derived from the Claude
        transcript record UUID and content block position, e.g.
        ``"747e:0:function_call"``.
    :param item_type: Omnigent conversation item type, e.g.
        ``"message"`` or ``"function_call"``.
    :param data: Item payload shaped like ``SessionEventInput.data``.
    :param response_id: Synthetic response id used to group the
        Claude turn in AP/web UI rendering.
    """

    source_id: str
    item_type: str
    data: dict[str, Any]
    response_id: str


@dataclass(frozen=True)
class TranscriptReadResult:
    """
    Result of reading Claude transcript JSONL records.

    :param line_cursor: Count of complete newline-terminated records
        consumed from the transcript, e.g. ``12``.
    :param byte_offset: Byte offset immediately after the last
        complete record consumed, e.g. ``4096``. A partial trailing
        line is not included.
    :param current_response_id: Response id for a Claude assistant
        turn that remains active across polls.
    :param items: Parsed Omnigent conversation items from the
        complete records after the caller's cursor.
    :param latest_usage: Token-usage from the most recent assistant
        entry with a ``message.usage`` block. Keys: ``context_tokens``,
        ``input_tokens``, ``output_tokens``. ``None`` when no such
        entry was scanned.
    :param latest_model: ``message.model`` from the most recent
        assistant entry, or ``None``.
    """

    line_cursor: int
    byte_offset: int
    current_response_id: str | None
    items: list[ClaudeTranscriptItem]
    latest_usage: dict[str, int] | None = None
    latest_model: str | None = None


@dataclass(frozen=True)
class _JsonlRecord:
    """
    One complete newline-terminated JSONL record.

    :param line_number: One-based line number relative to the reader's
        line cursor, e.g. ``5``.
    :param byte_offset: Byte offset where the record starts.
    :param next_byte_offset: Byte offset immediately after the
        newline-terminated record.
    :param text: UTF-8 decoded JSONL text including the trailing
        newline, or ``None`` when the complete record was not valid
        UTF-8 and should advance cursors without being parsed.
    """

    line_number: int
    byte_offset: int
    next_byte_offset: int
    text: str | None


@dataclass(frozen=True)
class _JsonlReadResult:
    """
    Complete-record read result for an append-only JSONL file.

    :param line_cursor: Count of complete records consumed.
    :param byte_offset: Byte offset after the last complete record.
    :param records: Complete records read after the requested byte
        offset.
    """

    line_cursor: int
    byte_offset: int
    records: list[_JsonlRecord]


def read_assistant_text_since(
    transcript_path: Path,
    start_line: int,
) -> tuple[int, list[str]]:
    """
    Read assistant text blocks appended after a transcript cursor.

    :param transcript_path: Claude transcript path, e.g.
        ``"/home/user/.claude/projects/x/session.jsonl"``.
    :param start_line: Zero-based line cursor captured before a
        message is injected into the Claude terminal.
    :returns: ``(new_cursor, text_chunks)``.
    """
    texts: list[str] = []
    cursor = 0
    try:
        with transcript_path.open("r", encoding="utf-8") as handle:
            for cursor, line in enumerate(handle, start=1):
                if cursor <= start_line:
                    continue
                text = _assistant_text_from_transcript_line(line)
                if text:
                    texts.append(text)
    except FileNotFoundError:
        return start_line, []
    return cursor, texts


def read_transcript_items_since(
    transcript_path: Path,
    start_line: int,
    *,
    agent_name: str,
    current_response_id: str | None = None,
) -> tuple[int, str | None, list[ClaudeTranscriptItem]]:
    """
    Read Claude transcript records as Omnigent conversation items.

    Claude Code writes append-only JSONL records whose ``message``
    payloads include user prompts, assistant text, native tool calls,
    and native tool results. This parser intentionally ignores
    metadata records (title, file-history, permission mode, system
    bookkeeping) and raw ``thinking`` blocks, while translating the
    user-visible semantic records into Omnigent item types the web UI
    already understands.

    :param transcript_path: Claude transcript path, e.g.
        ``"/home/user/.claude/projects/x/session.jsonl"``.
    :param start_line: One-based line cursor. Lines at or before
        this cursor are skipped.
    :param agent_name: Agent/model name stamped on assistant and
        tool-call items, e.g. ``"claude-native-ui"``.
    :param current_response_id: Response id for an in-progress
        Claude assistant turn from a previous poll.
    :returns: ``(new_cursor, current_response_id, items)``.
    """
    result = read_transcript_items_since_with_position(
        transcript_path,
        start_line,
        agent_name=agent_name,
        current_response_id=current_response_id,
    )
    return result.line_cursor, result.current_response_id, result.items


def read_transcript_items_since_with_position(
    transcript_path: Path,
    start_line: int,
    *,
    agent_name: str,
    current_response_id: str | None = None,
) -> TranscriptReadResult:
    """
    Read transcript items from a line cursor and return byte position.

    This compatibility reader supports existing durable state that
    only stored a line cursor. It scans the file once, parses only
    complete newline-terminated records after ``start_line``, and
    returns the byte offset so future polls can seek directly.

    :param transcript_path: Claude transcript path, e.g.
        ``"/home/user/.claude/projects/x/session.jsonl"``.
    :param start_line: One-based line cursor. Lines at or before
        this cursor are skipped.
    :param agent_name: Agent/model name stamped on assistant and
        tool-call items, e.g. ``"claude-native-ui"``.
    :param current_response_id: Response id for an in-progress
        Claude assistant turn from a previous poll.
    :returns: Parsed items plus line and byte cursors.
    """
    read_result = _read_complete_jsonl_records(
        transcript_path,
        byte_offset=0,
        start_line=0,
        emit_after_line=start_line,
    )
    items: list[ClaudeTranscriptItem] = []
    active_response_id = current_response_id
    latest_usage: dict[str, int] | None = None
    latest_model: str | None = None
    for record in read_result.records:
        if record.text is None:
            continue
        try:
            entry = json.loads(record.text)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        active_response_id, parsed = _transcript_items_from_entry(
            entry,
            line_number=record.line_number,
            record_offset=None,
            agent_name=agent_name,
            current_response_id=active_response_id,
        )
        items.extend(parsed)
        usage = _usage_from_transcript_entry(entry)
        if usage is not None:
            latest_usage = usage
        model = _model_from_transcript_entry(entry)
        if model is not None:
            latest_model = model
    return TranscriptReadResult(
        line_cursor=read_result.line_cursor,
        byte_offset=read_result.byte_offset,
        current_response_id=active_response_id,
        items=items,
        latest_usage=latest_usage,
        latest_model=latest_model,
    )


def read_transcript_items_from_offset(
    transcript_path: Path,
    byte_offset: int,
    *,
    start_line: int,
    agent_name: str,
    current_response_id: str | None = None,
    include_sidechains: bool = False,
) -> TranscriptReadResult:
    """
    Read transcript items appended after a byte offset.

    Only complete newline-terminated JSONL records are parsed. If
    Claude is midway through writing a trailing JSON record, the
    returned byte offset remains before that partial line so the next
    poll retries it after completion.

    :param transcript_path: Claude transcript path, e.g.
        ``"/home/user/.claude/projects/x/session.jsonl"``.
    :param byte_offset: Byte offset already consumed, e.g. ``4096``.
    :param start_line: Count of complete records already consumed.
        Used only to keep legacy line cursors and diagnostics
        monotonic while byte offsets drive the actual seek.
    :param agent_name: Agent/model name stamped on assistant and
        tool-call items, e.g. ``"claude-native-ui"``.
    :param current_response_id: Response id for an in-progress
        Claude assistant turn from a previous poll.
    :param include_sidechains: Pass ``True`` when reading a
        sub-agent's own ``agent-<id>.jsonl`` — every record there is
        a sidechain by Claude's definition, and dropping them would
        leave the sub-agent's child Omnigent conversation empty. The
        default ``False`` keeps the parent-transcript path
        unchanged.
    :returns: Parsed items plus updated line and byte cursors.
    """
    read_result = _read_complete_jsonl_records(
        transcript_path,
        byte_offset=byte_offset,
        start_line=start_line,
    )
    items: list[ClaudeTranscriptItem] = []
    active_response_id = current_response_id
    latest_usage: dict[str, int] | None = None
    latest_model: str | None = None
    for record in read_result.records:
        if record.text is None:
            continue
        try:
            entry = json.loads(record.text)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        active_response_id, parsed = _transcript_items_from_entry(
            entry,
            line_number=record.line_number,
            record_offset=record.byte_offset,
            agent_name=agent_name,
            current_response_id=active_response_id,
            include_sidechains=include_sidechains,
        )
        items.extend(parsed)
        usage = _usage_from_transcript_entry(entry)
        if usage is not None:
            latest_usage = usage
        model = _model_from_transcript_entry(entry)
        if model is not None:
            latest_model = model
    return TranscriptReadResult(
        line_cursor=read_result.line_cursor,
        byte_offset=read_result.byte_offset,
        current_response_id=active_response_id,
        items=items,
        latest_usage=latest_usage,
        latest_model=latest_model,
    )


def _read_complete_jsonl_records(
    path: Path,
    *,
    byte_offset: int,
    start_line: int,
    emit_after_line: int | None = None,
) -> _JsonlReadResult:
    """
    Read complete newline-terminated records from a JSONL file.

    The reader seeks to ``byte_offset`` and stops before a trailing
    partial line. That partial line's bytes are retried by the next
    poll after the writer appends its newline.

    :param path: JSONL file path.
    :param byte_offset: Byte offset where reading should begin,
        e.g. ``4096``.
    :param start_line: Count of complete records before
        ``byte_offset``, e.g. ``12``.
    :param emit_after_line: When provided, complete records at or
        before this line number are counted for cursor migration but
        not decoded or stored.
    :returns: Complete records plus updated line and byte cursors.
    """
    if byte_offset < 0:
        raise ValueError(f"byte_offset must be non-negative, got {byte_offset}")
    if start_line < 0:
        raise ValueError(f"start_line must be non-negative, got {start_line}")
    records: list[_JsonlRecord] = []
    cursor = start_line
    position = byte_offset
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            if byte_offset > file_size:
                handle.seek(0)
                cursor = 0
                position = 0
            else:
                handle.seek(byte_offset)
            while True:
                record_start = position
                raw = handle.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    break
                position = handle.tell()
                cursor += 1
                if emit_after_line is not None and cursor <= emit_after_line:
                    continue
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = None
                records.append(
                    _JsonlRecord(
                        line_number=cursor,
                        byte_offset=record_start,
                        next_byte_offset=position,
                        text=text,
                    )
                )
    except FileNotFoundError:
        return _JsonlReadResult(
            line_cursor=start_line,
            byte_offset=byte_offset,
            records=[],
        )
    return _JsonlReadResult(
        line_cursor=cursor,
        byte_offset=position,
        records=records,
    )


def _model_from_transcript_entry(entry: dict[str, Any]) -> str | None:
    """
    Return ``message.model`` from an assistant transcript record.

    Surfaced on :class:`TranscriptReadResult.latest_model` for
    diagnostics. The ring's denominator comes from the statusLine
    stdin (see :func:`read_claude_context_state`); the JSONL model
    name is no longer used to size the ring.

    :param entry: One decoded transcript JSONL record.
    :returns: API model name, or ``None`` for non-assistant entries
        and entries missing the field.
    """
    message = entry.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    model = message.get("model")
    if isinstance(model, str) and model:
        return model
    return None


def _usage_from_transcript_entry(entry: dict[str, Any]) -> dict[str, int] | None:
    """
    Extract token-usage from one Claude assistant transcript entry.

    ``context_tokens`` is ``input + cache_creation + cache_read`` — the
    bytes that will reappear in the next call's prompt. Output tokens
    are reported separately since they don't shift the prompt forward.

    :param entry: One decoded transcript JSONL record.
    :returns: ``{"context_tokens", "input_tokens", "output_tokens"}``
        when the record is an assistant entry with usage; ``None``
        otherwise.
    """
    message = entry.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cache_creation = usage.get("cache_creation_input_tokens")
    cache_read = usage.get("cache_read_input_tokens")
    if not isinstance(input_tokens, int):
        return None
    if not isinstance(output_tokens, int):
        output_tokens = 0
    cc = cache_creation if isinstance(cache_creation, int) else 0
    cr = cache_read if isinstance(cache_read, int) else 0
    result: dict[str, int] = {
        "context_tokens": input_tokens + cc + cr,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cc:
        result["cache_creation_input_tokens"] = cc
    if cr:
        result["cache_read_input_tokens"] = cr
    return result


def _assistant_text_from_transcript_line(line: str) -> str | None:
    """
    Extract assistant text from one Claude transcript JSONL line.

    :param line: Raw JSONL record.
    :returns: Assistant text, or ``None`` when the record is not an
        assistant text message.
    """
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(entry, dict):
        return None
    message = entry.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts) or None


def _transcript_items_from_entry(
    entry: dict[str, Any],
    *,
    line_number: int,
    record_offset: int | None = None,
    agent_name: str,
    current_response_id: str | None,
    include_sidechains: bool = False,
) -> tuple[str | None, list[ClaudeTranscriptItem]]:
    """
    Convert one Claude transcript entry into Omnigent conversation items.

    :param entry: Decoded JSON object from one transcript line.
    :param line_number: One-based transcript line number.
    :param record_offset: Byte offset where the transcript record
        starts. Used for stable fallback source ids when Claude omits
        ``uuid`` and ``requestId``.
    :param agent_name: Agent/model name for assistant/tool items.
    :param current_response_id: Response id for the active Claude
        assistant turn, if a previous poll already started one.
    :param include_sidechains: When ``False`` (the default) any record
        with ``isSidechain: true`` is dropped — that's the right
        behavior when reading the parent's main transcript, where
        sub-agent records are inlined as sidechains and must not
        appear in the parent's Omnigent conversation. When ``True`` the
        flag is ignored — required when reading a sub-agent's own
        ``agent-<id>.jsonl`` (every record there is a sidechain by
        definition) so the sub-agent's items reach the child AP
        conversation. Caller is responsible for matching the flag to
        the file shape.
    :returns: Updated active response id and parsed items.
    """
    if not include_sidechains and entry.get("isSidechain") is True:
        return current_response_id, []
    if entry.get("type") == "attachment":
        return _attachment_transcript_items_from_entry(
            entry,
            line_number=line_number,
            record_offset=record_offset,
            current_response_id=current_response_id,
        )
    if entry.get("subtype") == "local_command":
        return _local_command_transcript_items_from_entry(
            entry,
            line_number=line_number,
            record_offset=record_offset,
            current_response_id=current_response_id,
        )
    message = entry.get("message")
    if not isinstance(message, dict):
        return current_response_id, []
    role = message.get("role")
    if role == "user":
        return _user_transcript_items_from_entry(
            entry,
            line_number=line_number,
            record_offset=record_offset,
            agent_name=agent_name,
            current_response_id=current_response_id,
        )
    if role == "assistant":
        return _assistant_transcript_items_from_entry(
            entry,
            line_number=line_number,
            record_offset=record_offset,
            agent_name=agent_name,
            current_response_id=current_response_id,
        )
    return current_response_id, []


def _attachment_transcript_items_from_entry(
    entry: dict[str, Any],
    *,
    line_number: int,
    record_offset: int | None,
    current_response_id: str | None,
) -> tuple[str | None, list[ClaudeTranscriptItem]]:
    """
    Parse user-visible Claude attachment transcript entries.

    Claude records prompts typed while an assistant turn is busy as
    ``attachment.type == "queued_command"`` rather than a normal
    ``role=user`` message. Treat prompt-mode queued commands as user
    messages so interruption inputs such as ``"STOP"`` appear in the
    Omnigent transcript and reset the active assistant response.

    :param entry: Decoded Claude transcript record.
    :param line_number: One-based transcript line number.
    :param record_offset: Byte offset where the transcript record
        starts, or ``None`` for legacy line-cursor reads.
    :param current_response_id: Response id for the active assistant
        turn. Ignored attachment metadata preserves this value.
    :returns: Updated active response id and parsed items.
    """
    attachment = entry.get("attachment")
    if not isinstance(attachment, dict):
        return current_response_id, []
    if attachment.get("type") != "queued_command":
        return current_response_id, []
    if attachment.get("commandMode") != "prompt":
        return current_response_id, []
    prompt = attachment.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return current_response_id, []
    source_key = _transcript_source_key(entry, line_number, record_offset)
    item = ClaudeTranscriptItem(
        source_id=_source_id(source_key, 0, "message"),
        item_type="message",
        data={
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        },
        response_id=_response_id_from_source(source_key),
    )
    return None, [item]


# Slash-command marker handling. Claude Code's embedded TUI emits
# multiple ``role=user`` records per operator action: a ``<command-
# name>`` echo, a sibling ``isMeta=true`` ``<local-command-caveat>``,
# a follow-up ``<local-command-stdout>`` (and friends), plus
# ``<bash-*>`` records when the operator types ``!cmd``. All are
# CLI scaffolding, not user-typed content — rendering any of them as
# a user bubble shows raw markup to a web viewer.
# Today: drop isMeta + every CLI-scaffolding-prefixed record; for
# ``<command-name>`` records also surface Skills as ``slash_command``
# items. The original blanket drop was reverted because it
# hid Skills; we keep the broad scaffolding filter and just
# selectively re-surface the Skill case.
_COMMAND_NAME_RE = re.compile(r"<command-name>(.*?)</command-name>", re.DOTALL)
_COMMAND_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)
_COMMAND_STDOUT_RE = re.compile(r"<local-command-stdout>(.*?)</local-command-stdout>", re.DOTALL)
_BASH_INPUT_RE = re.compile(r"<bash-input>(.*?)</bash-input>", re.DOTALL)
_BASH_STDOUT_RE = re.compile(r"<bash-stdout>(.*?)</bash-stdout>", re.DOTALL)
_BASH_STDERR_RE = re.compile(r"<bash-stderr>(.*?)</bash-stderr>", re.DOTALL)

# Markers that prefix a ``role=user`` record produced by Claude
# Code's CLI scaffolding (not user-typed content). ``<command-
# name>`` is handled separately by the slash-command parser;
# ``<bash-*>`` records are handled separately as ``terminal_command``
# items; the rest must always drop.
_CLI_SCAFFOLDING_MARKERS: tuple[str, ...] = (
    "<command-message>",
    "<command-args>",
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
)

# Claude Code's CLI built-ins (no leading ``/``). Each name is
# classified as either:
#
# - DROPPED — pure UI affordances (``/help``, ``/login``) and local
#   config (``/permissions``, ``/add-dir``). No conversation-visible
#   effect; surfacing them in the web UI is noise.
# - SURFACED — commands that change the next turn's behavior or the
#   conversation state (``/effort high``, ``/clear``, ``/compact``,
#   ``/model``, ``/ultrareview``). A web observer needs to see these,
#   otherwise the next assistant turn appears to shift unprompted.
#
# Unknown names fall through to the Skill branch — a safer default
# than silently hiding them.
_CLAUDE_CLI_DROPPED_COMMANDS: frozenset[str] = frozenset(
    {
        "add-dir",
        "agents",
        "bug",
        "config",
        "cost",
        "doctor",
        "exit",
        "fast",
        "feedback",
        "help",
        "hooks",
        "ide",
        "login",
        "logout",
        "mcp",
        "memory",
        "onboarding",
        "permissions",
        "plugin",
        "quiet",
        "quit",
        "release-notes",
        "resume",
        "save",
        "status",
        "terminal-setup",
        "upgrade",
        "verbose",
    }
)
_CLAUDE_CLI_SURFACED_COMMANDS: frozenset[str] = frozenset(
    {
        "clear",
        "compact",
        "effort",
        "model",
        "ultrareview",
    }
)


@dataclass(frozen=True)
class _SlashCommandPayload:
    """
    Parsed content of a slash-command ``role=user`` transcript record.

    :param name: Command name with leading ``/`` stripped, e.g.
        ``"dev-productivity:simplify"``.
    :param arguments: Verbatim ``<command-args>`` text; empty when none.
    :param output: Verbatim ``<local-command-stdout>`` text, or ``None``.
    """

    name: str
    arguments: str
    output: str | None


def _parse_slash_command_record(content: str) -> _SlashCommandPayload | None:
    """
    Parse a Claude Code slash-command marker blob.

    Returns ``None`` on a missing/empty/unclosed ``<command-name>``
    tag rather than raising — a single corrupt JSONL line must not
    kill the transcript poll loop.

    :param content: ``message.content`` string from a ``role=user``
        Claude Code JSONL record.
    :returns: Parsed payload, or ``None`` when no name could be
        extracted.
    """
    name_match = _COMMAND_NAME_RE.search(content)
    if name_match is None:
        return None
    raw_name = name_match.group(1).strip()
    if not raw_name:
        return None
    # Strip leading ``/`` so renderers can add their own prefix without double-rendering.
    name = raw_name.lstrip("/")
    if not name:
        return None
    args_match = _COMMAND_ARGS_RE.search(content)
    arguments = args_match.group(1).strip() if args_match else ""
    stdout_match = _COMMAND_STDOUT_RE.search(content)
    output = stdout_match.group(1) if stdout_match else None
    return _SlashCommandPayload(name=name, arguments=arguments, output=output)


def _local_command_transcript_items_from_entry(
    entry: dict[str, Any],
    *,
    line_number: int,
    record_offset: int | None,
    current_response_id: str | None,
) -> tuple[str | None, list[ClaudeTranscriptItem]]:
    """
    Parse a top-level Claude ``local_command`` transcript entry.

    Newer Claude Code builds can record shell-mode ``!cmd`` activity
    as top-level transcript records with ``subtype="local_command"``
    and a string ``content`` field instead of wrapping the same markup
    inside ``message.role=user``. Only ``<bash-*>`` records are
    conversation-visible here; slash-command local records are still
    handled by hook/fork detection and otherwise ignored.

    :param entry: Decoded Claude transcript record.
    :param line_number: One-based transcript line number.
    :param record_offset: Byte offset where the transcript record
        starts, or ``None`` for legacy line-cursor reads.
    :param current_response_id: Response id for an in-progress shell
        command group, if the input record was parsed in an earlier
        line.
    :returns: Updated active response id and parsed terminal-command
        items.
    """
    content = entry.get("content")
    if not isinstance(content, str) or not content:
        return current_response_id, []
    source_key = _transcript_source_key(entry, line_number, record_offset)
    fallback_response_id = _response_id_from_source(source_key)
    response_id = (
        fallback_response_id
        if _BASH_INPUT_RE.search(content) is not None
        else current_response_id or fallback_response_id
    )
    items = _terminal_command_items_from_content(
        content,
        source_key=source_key,
        response_id=response_id,
    )
    if not items:
        return current_response_id, []
    return response_id, items


def _terminal_command_items_from_content(
    content: str,
    *,
    source_key: str,
    response_id: str,
) -> list[ClaudeTranscriptItem]:
    """
    Parse Claude shell-mode markup into terminal-command items.

    Claude may emit shell input and output as separate records or as
    one record containing multiple ``<bash-*>`` tags. This helper
    emits at most one input item and one output item, preserving their
    order in the source record and giving both the same response id so
    the server transcript groups an invocation with its result.

    :param content: Transcript markup, e.g.
        ``"<bash-input>pwd</bash-input><bash-stdout>/tmp</bash-stdout>"``.
    :param source_key: Base transcript record key used to construct
        source ids, e.g. ``"rec_abc123"``.
    :param response_id: Synthetic response id for this terminal
        command group, e.g. ``"resp_claude_abc123"``.
    :returns: Parsed ``terminal_command`` items. Empty when no shell
        markers are present.
    """
    if not any(marker in content for marker in ("<bash-input>", "<bash-stdout>", "<bash-stderr>")):
        return []
    input_match = _BASH_INPUT_RE.search(content)
    stdout_match = _BASH_STDOUT_RE.search(content)
    stderr_match = _BASH_STDERR_RE.search(content)
    items: list[ClaudeTranscriptItem] = []
    item_index = 0
    if input_match is not None:
        items.append(
            ClaudeTranscriptItem(
                source_id=_source_id(source_key, item_index, "terminal_command"),
                item_type="terminal_command",
                data={"kind": "input", "input": input_match.group(1)},
                response_id=response_id,
            )
        )
        item_index += 1
    if stdout_match is not None or stderr_match is not None:
        items.append(
            ClaudeTranscriptItem(
                source_id=_source_id(source_key, item_index, "terminal_command"),
                item_type="terminal_command",
                data={
                    "kind": "output",
                    "stdout": stdout_match.group(1) if stdout_match is not None else None,
                    "stderr": stderr_match.group(1) if stderr_match is not None else None,
                },
                response_id=response_id,
            )
        )
    return items


def _user_transcript_items_from_entry(
    entry: dict[str, Any],
    *,
    line_number: int,
    record_offset: int | None,
    agent_name: str,
    current_response_id: str | None,
) -> tuple[str | None, list[ClaudeTranscriptItem]]:
    """
    Parse a Claude ``role=user`` transcript entry.

    :param entry: Decoded Claude transcript record.
    :param line_number: One-based transcript line number.
    :param record_offset: Byte offset where the transcript record
        starts, or ``None`` for legacy line-cursor reads.
    :param agent_name: Agent/model name attached to ``slash_command``
        items so the web UI can attribute the invocation.
    :param current_response_id: Response id for the active assistant
        turn; tool results keep this id.
    :returns: Updated active response id and parsed user/tool-result
        items.
    """
    # ``isMeta=true`` carries CLI scaffolding like
    # ``<local-command-caveat>``; no user-visible content.
    if entry.get("isMeta") is True:
        return current_response_id, []
    message = entry["message"]
    content = message.get("content") if isinstance(message, dict) else None
    source_key = _transcript_source_key(entry, line_number, record_offset)
    fallback_response_id = _response_id_from_source(source_key)
    items: list[ClaudeTranscriptItem] = []

    if isinstance(content, str):
        if not content:
            return current_response_id, []
        stripped = content.lstrip()
        # Skill invocations with args ship the tag order
        # ``<command-message>…<command-name>…<command-args>…`` — i.e.
        # ``<command-name>`` is NOT the first tag. Detect it anywhere
        # in the content, not just at the start.
        if "<command-name>" in stripped:
            payload = _parse_slash_command_record(content)
            # Drop unparseable markup rather than letting it fall through
            # to the user-bubble path — that rendered the markup verbatim
            # in the original bug.
            if payload is None or payload.name in _CLAUDE_CLI_DROPPED_COMMANDS:
                return current_response_id, []
            kind = "command" if payload.name in _CLAUDE_CLI_SURFACED_COMMANDS else "skill"
            data: dict[str, Any] = {
                "agent": agent_name,
                "kind": kind,
                "name": payload.name,
                "arguments": payload.arguments,
            }
            if payload.output is not None:
                data["output"] = payload.output
            items.append(
                ClaudeTranscriptItem(
                    source_id=_source_id(source_key, 0, "slash_command"),
                    item_type="slash_command",
                    data=data,
                    response_id=fallback_response_id,
                )
            )
            # Slash command opens a new logical turn; subsequent
            # assistant text must inherit this id so it clusters with
            # the indicator, not the prior bubble.
            return fallback_response_id, items
        # ``!cmd`` terminal commands may arrive here in older Claude
        # builds; newer builds use top-level ``local_command`` records.
        # In both shapes, surface the command and result as their own
        # transcript group instead of inheriting the previous assistant
        # response id.
        terminal_response_id = (
            fallback_response_id
            if _BASH_INPUT_RE.search(content) is not None
            else current_response_id or fallback_response_id
        )
        terminal_items = _terminal_command_items_from_content(
            content,
            source_key=source_key,
            response_id=terminal_response_id,
        )
        if terminal_items:
            return terminal_response_id, terminal_items
        # Other CLI-scaffolding records (stdout/stderr from /effort, etc.)
        # arrive as standalone ``role=user`` records and must drop instead
        # of leaking as user bubbles.
        if any(stripped.startswith(m) for m in _CLI_SCAFFOLDING_MARKERS):
            return current_response_id, []
        items.append(
            ClaudeTranscriptItem(
                source_id=_source_id(source_key, 0, "message"),
                item_type="message",
                data={
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}],
                },
                response_id=fallback_response_id,
            )
        )
        return None, items

    if not isinstance(content, list):
        return current_response_id, []

    user_blocks: list[dict[str, Any]] = []
    saw_user_text = False
    item_index = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str) or not text:
                continue
            # Defensively guard against slash-command markup or other
            # CLI-scaffolding markers ever arriving in list-form
            # content. Today these only ship in string content (the
            # branch above), but Claude Code's JSONL format is not
            # under our control — without this filter, a format
            # change would regress to rendering ``<command-name>…``
            # markup as a user bubble.
            stripped = text.lstrip()
            if "<command-name>" in stripped or any(
                stripped.startswith(m) for m in _CLI_SCAFFOLDING_MARKERS
            ):
                continue
            user_blocks.append({"type": "input_text", "text": text})
            saw_user_text = True
            continue
        if block_type != "tool_result":
            continue
        call_id = block.get("tool_use_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        response_id = current_response_id or _response_id_from_source(
            _parent_or_record_source_key(entry, line_number, record_offset)
        )
        items.append(
            ClaudeTranscriptItem(
                source_id=_source_id(source_key, item_index, "function_call_output"),
                item_type="function_call_output",
                data={
                    "call_id": call_id,
                    "output": _tool_result_output(entry, block),
                },
                response_id=response_id,
            )
        )
        item_index += 1

    if user_blocks:
        items.insert(
            0,
            ClaudeTranscriptItem(
                source_id=_source_id(source_key, item_index, "message"),
                item_type="message",
                data={
                    "role": "user",
                    "content": user_blocks,
                },
                response_id=fallback_response_id,
            ),
        )
    return (None if saw_user_text else current_response_id), items


def _assistant_transcript_items_from_entry(
    entry: dict[str, Any],
    *,
    line_number: int,
    record_offset: int | None,
    agent_name: str,
    current_response_id: str | None,
) -> tuple[str | None, list[ClaudeTranscriptItem]]:
    """
    Parse a Claude ``role=assistant`` transcript entry.

    :param entry: Decoded Claude transcript record.
    :param line_number: One-based transcript line number.
    :param record_offset: Byte offset where the transcript record
        starts, or ``None`` for legacy line-cursor reads.
    :param agent_name: Agent/model name for assistant/tool items.
    :param current_response_id: Response id for the active Claude
        assistant turn.
    :returns: Updated active response id and parsed assistant/tool
        items.
    """
    message = entry["message"]
    content = message.get("content") if isinstance(message, dict) else None
    source_key = _transcript_source_key(entry, line_number, record_offset)
    response_id = current_response_id or _response_id_from_source(source_key)
    items: list[ClaudeTranscriptItem] = []

    if isinstance(content, str):
        if content:
            items.append(
                _assistant_message_item(
                    source_key=source_key,
                    item_index=0,
                    agent_name=agent_name,
                    response_id=response_id,
                    text=content,
                )
            )
        return response_id, items

    if not isinstance(content, list):
        return current_response_id, []

    for item_index, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                items.append(
                    _assistant_message_item(
                        source_key=source_key,
                        item_index=item_index,
                        agent_name=agent_name,
                        response_id=response_id,
                        text=text,
                    )
                )
            continue
        if block_type == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            if not isinstance(tool_id, str) or not tool_id:
                continue
            if not isinstance(name, str) or not name:
                continue
            arguments = block.get("input")
            if not isinstance(arguments, dict):
                arguments = {}
            items.append(
                ClaudeTranscriptItem(
                    source_id=_source_id(source_key, item_index, "function_call"),
                    item_type="function_call",
                    data={
                        "agent": agent_name,
                        "name": name,
                        "arguments": json.dumps(arguments, separators=(",", ":")),
                        "call_id": tool_id,
                    },
                    response_id=response_id,
                )
            )
    return response_id if items else current_response_id, items


_CONTEXT_OVERFLOW_RE = re.compile(
    r"^prompt is too long",
    re.IGNORECASE,
)

_CONTEXT_OVERFLOW_REPLACEMENT = (
    "Context limit reached — the conversation has grown too long for "
    "the model’s context window. Use /compact to summarize and free up "
    "space, or /clear to start a new conversation."
)


def _assistant_message_item(
    *,
    source_key: str,
    item_index: int,
    agent_name: str,
    response_id: str,
    text: str,
) -> ClaudeTranscriptItem:
    """
    Build an assistant message item from one Claude text block.

    :param source_key: Base transcript record key.
    :param item_index: Content block index inside the record.
    :param agent_name: Agent/model name for the assistant message.
    :param response_id: Response id grouping the Claude turn.
    :param text: Assistant text block.
    :returns: Parsed transcript item.
    """
    display_text = text
    if _CONTEXT_OVERFLOW_RE.match(text.strip()):
        display_text = _CONTEXT_OVERFLOW_REPLACEMENT
    return ClaudeTranscriptItem(
        source_id=_source_id(source_key, item_index, "message"),
        item_type="message",
        data={
            "role": "assistant",
            "agent": agent_name,
            "content": [{"type": "output_text", "text": display_text}],
        },
        response_id=response_id,
    )


def _tool_result_output(entry: dict[str, Any], block: dict[str, Any]) -> str:
    """
    Return the UI-facing output string for a Claude tool result.

    :param entry: Decoded Claude transcript record containing
        optional ``toolUseResult`` metadata.
    :param block: ``tool_result`` content block from ``message``.
    :returns: String output for a ``function_call_output`` item.
    """
    content = block.get("content")
    if isinstance(content, str):
        return content
    if content is not None:
        return json.dumps(content, separators=(",", ":"))
    tool_use_result = entry.get("toolUseResult")
    if isinstance(tool_use_result, str):
        return tool_use_result
    if tool_use_result is not None:
        return json.dumps(tool_use_result, separators=(",", ":"))
    return ""


def _transcript_source_key(
    entry: dict[str, Any],
    line_number: int,
    record_offset: int | None = None,
) -> str:
    """
    Return the stable key for a Claude transcript record.

    :param entry: Decoded Claude transcript record.
    :param line_number: One-based transcript line number.
    :param record_offset: Byte offset where the transcript record
        starts, or ``None`` when unavailable.
    :returns: Claude UUID/request id, byte-offset fallback, or a
        legacy line-number fallback.
    """
    for key in ("uuid", "requestId"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    if record_offset is not None:
        return f"byte-{record_offset}"
    return f"line-{line_number}"


def _parent_or_record_source_key(
    entry: dict[str, Any],
    line_number: int,
    record_offset: int | None = None,
) -> str:
    """
    Return a parent key for tool results when Claude supplies one.

    :param entry: Decoded Claude transcript record.
    :param line_number: One-based transcript line number.
    :param record_offset: Byte offset where the transcript record
        starts, or ``None`` when unavailable.
    :returns: Parent UUID when present, otherwise the record key.
    """
    parent = entry.get("parentUuid")
    if isinstance(parent, str) and parent:
        return parent
    return _transcript_source_key(entry, line_number, record_offset)


def _response_id_from_source(source: str) -> str:
    """
    Derive a deterministic Omnigent response id from a Claude source key.

    :param source: Claude UUID/request id/line key.
    :returns: String id with the standard ``resp_`` prefix.
    """
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]
    return f"resp_claude_{digest}"


def _source_id(source_key: str, item_index: int, item_type: str) -> str:
    """
    Build a per-item idempotency key for a transcript-derived item.

    :param source_key: Base Claude record key.
    :param item_index: Content block index inside the record.
    :param item_type: Omnigent item type.
    :returns: Stable source id string.
    """
    return f"{source_key}:{item_index}:{item_type}"
