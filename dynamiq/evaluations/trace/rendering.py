"""Render a `Run` tree from `TracingCallbackHandler` into a compact prompt-friendly text.

The renderer is the bridge between the structured trace produced by
:class:`dynamiq.callbacks.tracing.TracingCallbackHandler` and an LLM judge:
it walks the parent/child relationships, indents nested runs, and tags
each step with a short ``run_id`` anchor so the model can cite specific
steps in its findings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from dynamiq.callbacks.tracing import ExecutionRun, Run, RunStatus
from dynamiq.utils import JsonWorkflowEncoder

_EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)

DEFAULT_MAX_TRACE_CHARS = 12000
DEFAULT_FIELD_CHAR_LIMIT = 800
DEFAULT_SHORT_ID_LENGTH = 8


@dataclass
class RenderedTrace:
    """Result of rendering a trace.

    Attributes:
        text: The rendered, prompt-friendly trace.
        short_id_to_run: Mapping from short run-id (first 8 chars of UUID) to
            the originating ``Run`` object. Used to resolve LLM-cited findings
            back to real runs afMater evaluation.
    """

    text: str
    short_id_to_run: dict[str, Run]


def _short_id(run: Run, length: int = DEFAULT_SHORT_ID_LENGTH) -> str:
    return str(run.id)[:length]


_REDACTED = "[REDACTED]"

_REDACTION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # OpenAI / Anthropic / generic "sk-..." / "pk-..." / "rk-..." style keys.
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b"), _REDACTED),
    # AWS access key ids.
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), _REDACTED),
    # GitHub personal / fine-grained / OAuth / app tokens.
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), _REDACTED),
    # Google API keys.
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), _REDACTED),
    # Slack tokens.
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), _REDACTED),
    # Authorization: Bearer <token> — keep the "Bearer " prefix.
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-+/=]{10,}"), rf"\1{_REDACTED}"),
    # URL-embedded basic-auth creds: scheme://user:password@host
    (
        re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)[^/\s:@]+:[^/\s:@]+@"),
        rf"\1{_REDACTED}:{_REDACTED}@",
    ),
)


def redact_secrets(text: str) -> str:
    """Scrub common API-key / token shapes from a rendered field.

    Pattern-based and conservative — misses custom secret formats, and may
    false-positive on long random identifiers. Use as a safety net, not a
    substitute for capture-time redaction.
    """
    if not text:
        return text
    for pattern, replacement in _REDACTION_RULES:
        text = pattern.sub(replacement, text)
    return text


def _format_field(value: object, limit: int) -> str:
    if value is None:
        return ""
    try:
        rendered = json.dumps(value, cls=JsonWorkflowEncoder, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(value)
    rendered = redact_secrets(rendered)
    if len(rendered) > limit:
        return f"{rendered[:limit]}... [truncated {len(rendered) - limit} chars]"
    return rendered


def _duration_ms(run: Run) -> str:
    if run.start_time is None or run.end_time is None:
        return "?"
    delta = run.end_time - run.start_time
    return f"{int(delta.total_seconds() * 1000)}"


def _status_str(status: RunStatus | None) -> str:
    return status.value if status is not None else "unknown"


def _render_executions(executions: Iterable[ExecutionRun], indent: str) -> list[str]:
    executions = list(executions)
    if not executions:
        return []
    failed = [e for e in executions if e.status == RunStatus.FAILED]
    line = f"{indent}executions: {len(executions)} (failed: {len(failed)})"
    lines = [line]
    for exc in failed:
        message = ""
        if isinstance(exc.error, dict):
            message = str(exc.error.get("message", ""))
        elif exc.error is not None:
            message = str(exc.error)
        if message:
            lines.append(f"{indent}  - failed execution: {message}")
    return lines


def _render_run_block(
    run: Run,
    depth: int,
    field_char_limit: int,
    short_id_length: int,
) -> list[str]:
    indent = "  " * depth
    inner_indent = "  " * (depth + 1)
    header = (
        f"{indent}[run_id={_short_id(run, short_id_length)}] "
        f"{run.type.value}:{run.name or '?'} "
        f"status={_status_str(run.status)} "
        f"duration={_duration_ms(run)}ms"
    )
    lines = [header]

    input_str = _format_field(run.input, field_char_limit)
    if input_str:
        lines.append(f"{inner_indent}input: {input_str}")

    output_str = _format_field(run.output, field_char_limit)
    if output_str:
        lines.append(f"{inner_indent}output: {output_str}")

    if run.error:
        # Errors are always included verbatim — they are the most relevant
        # signal for a judge and must survive truncation pressure.
        if isinstance(run.error, dict):
            err_msg = str(run.error.get("message", run.error))
        else:
            err_msg = str(run.error)
        lines.append(f"{inner_indent}error: {redact_secrets(err_msg)}")

    metadata = run.metadata or {}
    usage = metadata.get("usage") if isinstance(metadata, dict) else None
    if usage:
        lines.append(f"{inner_indent}usage: {_format_field(usage, field_char_limit)}")

    tool_data = metadata.get("tool_data") if isinstance(metadata, dict) else None
    if tool_data:
        lines.append(f"{inner_indent}tool_data: {_format_field(tool_data, field_char_limit)}")

    lines.extend(_render_executions(run.executions, inner_indent))
    return lines


def _normalize_runs(runs: Run | Iterable[Run] | None) -> list[Run]:
    if runs is None:
        return []
    if isinstance(runs, Run):
        return [runs]
    return [r for r in runs if isinstance(r, Run)]


def _build_children_map(runs: list[Run]) -> tuple[dict, list[Run]]:
    by_id = {r.id: r for r in runs}
    children: dict = {r.id: [] for r in runs}
    roots: list[Run] = []
    for run in runs:
        parent_id = run.parent_run_id
        if parent_id is not None and parent_id in by_id:
            children[parent_id].append(run)
        else:
            roots.append(run)

    def _sort_key(r: Run) -> tuple:
        return (r.start_time or _EPOCH, str(r.id))

    for child_list in children.values():
        child_list.sort(key=_sort_key)
    roots.sort(key=_sort_key)

    # Promote any run that isn't reachable from a real root (i.e. trapped in
    # a cycle) to a synthetic root so the renderer still surfaces it; the
    # walker's visited set will then break the cycle.
    reachable: set = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node.id in reachable:
            continue
        reachable.add(node.id)
        stack.extend(children.get(node.id, []))
    for run in runs:
        if run.id not in reachable:
            roots.append(run)
            reachable.add(run.id)
            stack = list(children.get(run.id, []))
            while stack:
                node = stack.pop()
                if node.id in reachable:
                    continue
                reachable.add(node.id)
                stack.extend(children.get(node.id, []))

    return children, roots


def render_trace(
    runs: Run | Iterable[Run] | None,
    max_chars: int = DEFAULT_MAX_TRACE_CHARS,
    field_char_limit: int = DEFAULT_FIELD_CHAR_LIMIT,
    short_id_length: int = DEFAULT_SHORT_ID_LENGTH,
) -> RenderedTrace:
    """Render a captured `Run` tree into a compact text suitable for an LLM judge.

    Args:
        runs: A single root :class:`Run`, an iterable of runs (e.g.
            ``list(tracing.runs.values())``), or ``None``.
        max_chars: Soft global character budget. When exceeded, the deepest
            tail of the rendered trace is dropped and replaced with a marker.
        field_char_limit: Per-field truncation cap applied to inputs, outputs,
            usage and tool_data values before the global budget kicks in.

    Returns:
        A :class:`RenderedTrace` containing the rendered text and the
        short-id → :class:`Run` mapping needed to resolve LLM citations.
    """
    run_list = _normalize_runs(runs)
    if not run_list:
        return RenderedTrace(text="", short_id_to_run={})

    children, roots = _build_children_map(run_list)
    short_id_to_run: dict[str, Run] = {}

    sections: list[str] = []
    multi_root = len(roots) > 1

    def walk(run: Run, depth: int, visited: set) -> list[str]:
        if run.id in visited:
            indent = "  " * depth
            return [f"{indent}<cycle detected at run_id={_short_id(run, short_id_length)}>"]
        visited.add(run.id)
        short_id_to_run[_short_id(run, short_id_length)] = run
        block = _render_run_block(run, depth, field_char_limit, short_id_length)
        for child in children.get(run.id, []):
            block.extend(walk(child, depth + 1, visited))
        return block

    for idx, root in enumerate(roots, start=1):
        if multi_root:
            sections.append(f"=== Trace {idx}/{len(roots)} ===")
        sections.extend(walk(root, depth=0, visited=set()))

    text = "\n".join(sections)
    if len(text) > max_chars:
        truncated = text[:max_chars]
        # Cut at the last newline so we don't slice mid-line.
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
        omitted_chars = len(text) - len(truncated)
        text = f"{truncated}\n... [trace truncated, {omitted_chars} chars omitted]"

    return RenderedTrace(text=text, short_id_to_run=short_id_to_run)
