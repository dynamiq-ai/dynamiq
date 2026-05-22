"""Function-calling message sanitization, ported from litellm.

Mirrors ``litellm.litellm_core_utils.prompt_templates.factory.sanitize_messages_for_tool_calling``
but does not depend on litellm internals or its ``litellm.modify_params`` global.

Handles four cases on OpenAI-format message dicts:
  A. Missing tool_result for an assistant tool_call → inject a synthetic reply.
  B. Orphan tool_result whose tool_call_id matches no prior assistant tool_call → drop it.
  C. Empty/whitespace user or assistant text content → replace with a placeholder.
  D. Duplicate tool_results for the same tool_call_id within a contiguous block → keep the last.
"""

from __future__ import annotations

EMPTY_CONTENT_PLACEHOLDER = "[System: Empty message content sanitised to satisfy protocol]"


def _dummy_tool_result_content(tool_name: str) -> str:
    return f"[System: Tool execution skipped/interrupted by user. No result provided for tool '{tool_name}'.]"


def _tool_call_id(tool_call: object) -> str | None:
    if isinstance(tool_call, dict):
        return tool_call.get("id")
    return getattr(tool_call, "id", None)


def _tool_call_name(tool_call: object) -> str:
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        if isinstance(function, dict):
            return function.get("name", "unknown_tool")
        return getattr(function, "name", "unknown_tool")
    function = getattr(tool_call, "function", None)
    if function is not None:
        return getattr(function, "name", "unknown_tool")
    return "unknown_tool"


def _sanitize_empty_text_content(message: dict) -> dict:
    """Case C: replace empty/whitespace user or assistant text content with a placeholder."""
    if message.get("role") not in ("user", "assistant"):
        return message
    content = message.get("content")
    if not isinstance(content, str):
        return message
    if content and content.strip():
        return message
    copy = dict(message)
    copy["content"] = EMPTY_CONTENT_PLACEHOLDER
    return copy


def _is_orphaned_tool_result(current: dict, sanitized: list[dict]) -> bool:
    """Case B: tool message whose tool_call_id matches no prior assistant tool_call."""
    if current.get("role") not in ("tool", "function"):
        return False
    tool_call_id = current.get("tool_call_id")
    if not tool_call_id:
        return False

    for prev in reversed(sanitized):
        if prev.get("role") != "assistant":
            continue
        tool_calls = prev.get("tool_calls") or []
        for tc in tool_calls:
            if _tool_call_id(tc) == tool_call_id:
                return False
        # Most recent assistant message lacks the matching id → orphan.
        return True
    return True


def _add_missing_tool_results(
    current: dict, messages: list[dict], current_index: int
) -> tuple[list[dict], int]:
    """Case A: synthesize tool replies for any tool_call without a matching tool message.

    Returns the assistant message plus existing tool replies plus any synthesized
    replies, and the count of original tool messages consumed (so the caller can
    advance its index past them).
    """
    tool_calls = current.get("tool_calls") or []
    if not tool_calls:
        return [current], 0

    expected_ids: set[str] = set()
    for tc in tool_calls:
        tc_id = _tool_call_id(tc)
        if tc_id:
            expected_ids.add(tc_id)

    found_ids: set[str] = set()
    actual_replies: list[dict] = []
    j = current_index + 1
    while j < len(messages):
        nxt = messages[j]
        role = nxt.get("role")
        if role == "assistant":
            break
        if role in ("tool", "function"):
            tc_id = nxt.get("tool_call_id")
            if tc_id and tc_id in expected_ids:
                found_ids.add(tc_id)
                actual_replies.append(nxt)
        j += 1

    missing_ids = expected_ids - found_ids
    if not missing_ids:
        return [current], 0

    result: list[dict] = [current, *actual_replies]
    for missing_id in missing_ids:
        tool_name = "unknown_tool"
        for tc in tool_calls:
            if _tool_call_id(tc) == missing_id:
                tool_name = _tool_call_name(tc)
                break
        result.append(
            {
                "role": "tool",
                "tool_call_id": missing_id,
                "content": _dummy_tool_result_content(tool_name),
            }
        )
    return result, len(actual_replies)


def _dedupe_tool_results(messages: list[dict]) -> list[dict]:
    """Case D: within a contiguous block of tool messages, keep only the last
    tool_result per tool_call_id. A non-tool message resets the block.
    """
    duplicates: set[int] = set()
    seen: dict[str, int] = {}
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in ("tool", "function"):
            tcid = msg.get("tool_call_id")
            if isinstance(tcid, str) and tcid:
                if tcid in seen:
                    duplicates.add(seen[tcid])
                seen[tcid] = idx
        else:
            seen = {}

    if not duplicates:
        return messages
    return [m for i, m in enumerate(messages) if i not in duplicates]


def sanitize_fc_messages(messages: list[dict]) -> list[dict]:
    """Run Cases A/B/C/D on outgoing FC messages. See module docstring."""
    sanitized: list[dict] = []
    i = 0
    while i < len(messages):
        current = _sanitize_empty_text_content(messages[i])
        if current.get("role") == "assistant":
            result_messages, consumed = _add_missing_tool_results(current, messages, i)
            if len(result_messages) > 1:
                sanitized.extend(result_messages)
                i += 1 + consumed
                continue
        if _is_orphaned_tool_result(current, sanitized):
            i += 1
            continue
        sanitized.append(current)
        i += 1

    return _dedupe_tool_results(sanitized)
