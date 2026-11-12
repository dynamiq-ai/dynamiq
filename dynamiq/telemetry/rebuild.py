"""Run records rebuilt from Dynamiq spans: the "Rebuilding a run" table of docs/reference/opentelemetry-tracing.md.

``rebuild_runs`` turns the finished spans of a run into the records a ``TracingCallbackHandler`` attached to the
same run collects (``Run.to_dict()``). The spans are read through the attributes that the OpenTelemetry SDK's
``ReadableSpan`` has, so the SDK is not needed here: ``spans_from_otlp_json`` gives the same attributes for spans
stored as OTLP/JSON.
"""

import base64
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.callbacks.tracing import PROMPT_MESSAGES_BASE_KEY, RunStatus, RunType
from dynamiq.runnables import RunnableStatus
from dynamiq.telemetry import attributes as attrs

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

RUN_STATUS_PAUSED = "paused"

# Status of a rebuilt run -> the RunnableStatus a flow reports for a node in its output.
_NODE_RESULT_STATUS = {
    RunStatus.SUCCEEDED.value: RunnableStatus.SUCCESS.value,
    # A paused run is recorded as failed already.
    RunStatus.FAILED.value: RunnableStatus.FAILURE.value,
    RunStatus.SKIPPED.value: RunnableStatus.SKIP.value,
    RunStatus.CANCELED.value: RunnableStatus.CANCELED.value,
}

_RUN_NAMES = {RunType.WORKFLOW.value: "Workflow", RunType.FLOW.value: "Flow"}

_HOOK_FIELDS = {
    "hook": attrs.DYNAMIQ_HOOK_NAME,
    "type": attrs.DYNAMIQ_HOOK_TYPE,
    "point": attrs.DYNAMIQ_HOOK_POINT,
    "decision": attrs.DYNAMIQ_HOOK_DECISION,
}

_MODEL_OPERATIONS = frozenset({attrs.OPERATION_CHAT, attrs.OPERATION_EMBEDDINGS})

_MODEL_USAGE = {
    "prompt_tokens": attrs.GEN_AI_USAGE_INPUT_TOKENS,
    "completion_tokens": attrs.GEN_AI_USAGE_OUTPUT_TOKENS,
    "cache_read_input_tokens": attrs.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    "cache_creation_input_tokens": attrs.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
}
_OTHER_USAGE = {
    "prompt_tokens": attrs.DYNAMIQ_USAGE_INPUT_TOKENS,
    "completion_tokens": attrs.DYNAMIQ_USAGE_OUTPUT_TOKENS,
}
_COMMON_USAGE = {
    "total_tokens": attrs.DYNAMIQ_USAGE_TOTAL_TOKENS,
    "prompt_tokens_cost_usd": attrs.DYNAMIQ_USAGE_INPUT_COST_USD,
    "completion_tokens_cost_usd": attrs.DYNAMIQ_USAGE_OUTPUT_COST_USD,
    "total_tokens_cost_usd": attrs.DYNAMIQ_USAGE_TOTAL_COST_USD,
}


class RebuiltExecution(BaseModel):
    """One execution attempt of a node run, in the shape of ``ExecutionRun.to_dict()``."""

    id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    status: RunStatus | None = None
    input: Any = None
    output: Any = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)


class RebuiltRun(BaseModel):
    """A run rebuilt from its span, in the shape of ``Run.to_dict()``."""

    id: str
    name: str | None = None
    type: RunType
    trace_id: str
    source_id: str | None = None
    session_id: str | None = None
    start_time: datetime
    end_time: datetime | None = None
    parent_run_id: str | None = None
    status: RunStatus | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    executions: list[RebuiltExecution] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    _error_type: str | None = PrivateAttr(default=None)

    model_config = ConfigDict(use_enum_values=True)


def rebuild_runs(spans: Iterable[Any], *, prompt_compaction_enabled: bool = False) -> list[dict[str, Any]]:
    """The runs a ``TracingCallbackHandler`` records for the run whose finished spans are ``spans``.

    Args:
        spans: Finished spans with the attributes of the OpenTelemetry SDK's ``ReadableSpan``: ``context``
            (``trace_id``, ``span_id``), ``parent``, ``start_time``, ``end_time``, ``attributes`` and ``events``
            (``name``, ``timestamp``, ``attributes``). Spans without ``dynamiq.run.id`` are ignored.
        prompt_compaction_enabled: Whether the handler being reproduced had ``compact_llm_prompts_enabled``. It
            decides between the two records of an LLM run that holds its whole prompt under a parent run:
            ``messages_base`` set to None, or no ``messages_base`` key.

    Returns:
        One dict per Dynamiq span, in the shape of ``Run.to_dict()``, ordered by start time. The flow run of a
        workflow has no span and is not among them; the nodes of that flow name the workflow run as parent.
    """
    all_spans = {span.context.span_id: span for span in spans}
    dynamiq_spans = [span for span in all_spans.values() if attrs.DYNAMIQ_RUN_ID in (span.attributes or {})]
    run_ids = {_format_span_id(span.context.span_id): span.attributes[attrs.DYNAMIQ_RUN_ID] for span in dynamiq_spans}
    dynamiq_spans.sort(key=lambda span: (span.start_time or 0, span.context.span_id))

    runs = [
        _rebuild_run(span, _parent_run_span(span, all_spans), run_ids, prompt_compaction_enabled)
        for span in dynamiq_spans
    ]
    for run in runs:
        if run.type in (RunType.WORKFLOW.value, RunType.FLOW.value):
            run.output = _flow_output(run, runs) if run.status == RunStatus.SUCCEEDED.value else None
    return [run.model_dump() for run in runs]


def messages_from_parts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prompt messages, as LLM nodes send them, from messages in the convention's input message shape.

    Inverts the Messages table of the spec. A message with a single text part gets a string ``content``; a
    message with tool calls and no text gets ``""``, as ``Message.content`` is a string.
    """
    return [_message_from_parts(message) for message in messages]


# OTLP/JSON


class OtlpSpanContext(BaseModel):
    """Trace and span id of a span, as integers like the SDK's ``SpanContext``."""

    trace_id: int
    span_id: int


class OtlpEvent(BaseModel):
    """A span event with the attributes of the SDK's ``Event``."""

    name: str
    timestamp: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class OtlpSpan(BaseModel):
    """A span read from OTLP/JSON, with the attributes ``rebuild_runs`` reads from the SDK's ``ReadableSpan``."""

    name: str
    context: OtlpSpanContext
    parent: OtlpSpanContext | None = None
    start_time: int | None = None
    end_time: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[OtlpEvent] = Field(default_factory=list)
    scope_name: str | None = None
    scope_version: str | None = None


def spans_from_otlp_json(request: Mapping[str, Any] | str | bytes) -> list[OtlpSpan]:
    """The spans of an OTLP/JSON ``ExportTraceServiceRequest`` (trace and span ids as hex strings)."""
    if isinstance(request, (str, bytes)):
        request = json.loads(request)
    spans = []
    for resource_spans in request.get("resourceSpans") or []:
        for scope_spans in resource_spans.get("scopeSpans") or []:
            scope = scope_spans.get("scope") or {}
            for span in scope_spans.get("spans") or []:
                trace_id = int(span["traceId"], 16)
                parent_span_id = span.get("parentSpanId")
                spans.append(
                    OtlpSpan(
                        name=span.get("name", ""),
                        context=OtlpSpanContext(trace_id=trace_id, span_id=int(span["spanId"], 16)),
                        parent=(
                            OtlpSpanContext(trace_id=trace_id, span_id=int(parent_span_id, 16))
                            if parent_span_id
                            else None
                        ),
                        start_time=_otlp_int(span.get("startTimeUnixNano")),
                        end_time=_otlp_int(span.get("endTimeUnixNano")),
                        attributes=_otlp_attributes(span.get("attributes")),
                        events=[
                            OtlpEvent(
                                name=event.get("name", ""),
                                timestamp=_otlp_int(event.get("timeUnixNano")),
                                attributes=_otlp_attributes(event.get("attributes")),
                            )
                            for event in span.get("events") or []
                        ],
                        scope_name=scope.get("name"),
                        scope_version=scope.get("version"),
                    )
                )
    return spans


def _otlp_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _otlp_attributes(key_values: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {item["key"]: _otlp_value(item.get("value") or {}) for item in key_values or []}


def _otlp_value(value: dict[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "arrayValue" in value:
        return tuple(_otlp_value(item) for item in value["arrayValue"].get("values") or [])
    if "kvlistValue" in value:
        return _otlp_attributes(value["kvlistValue"].get("values"))
    if "bytesValue" in value:
        return base64.b64decode(value["bytesValue"])
    return None


# Runs


def _format_span_id(span_id: int) -> str:
    return format(span_id, "016x")


def _time(unix_nano: int | None) -> datetime | None:
    if unix_nano is None:
        return None
    return _EPOCH + timedelta(microseconds=unix_nano // 1000)


def _json(attributes: Mapping[str, Any], key: str) -> Any:
    value = attributes.get(key)
    return None if value is None else json.loads(value)


def _parent_run_span(span: Any, all_spans: dict[int, Any]) -> Any | None:
    """Span of the run that started the run of ``span``: the nearest Dynamiq ancestor, or None for a root."""
    if span.attributes.get(attrs.DYNAMIQ_RUN_IS_ROOT):
        return None
    seen = set()
    parent = span.parent
    while parent is not None and parent.span_id not in seen:
        seen.add(parent.span_id)
        ancestor = all_spans.get(parent.span_id)
        if ancestor is None:
            return None
        if attrs.DYNAMIQ_RUN_ID in (ancestor.attributes or {}):
            return ancestor
        parent = ancestor.parent
    return None


def _event_attributes(span: Any, name: str) -> list[tuple[Any, Mapping[str, Any]]]:
    return [(event, event.attributes or {}) for event in span.events or [] if event.name == name]


def _rebuild_run(span: Any, parent: Any | None, run_ids: dict[str, str], compaction: bool) -> RebuiltRun:
    a = span.attributes
    run_type = a[attrs.DYNAMIQ_RUN_TYPE]
    status = a.get(attrs.DYNAMIQ_RUN_STATUS)
    is_tool = a.get(attrs.GEN_AI_OPERATION_NAME) == attrs.OPERATION_EXECUTE_TOOL
    run = RebuiltRun(
        id=a[attrs.DYNAMIQ_RUN_ID],
        name=_RUN_NAMES.get(run_type) or a.get(attrs.DYNAMIQ_NODE_NAME),
        type=run_type,
        trace_id=str(UUID(int=span.context.trace_id)),
        source_id=a.get(attrs.DYNAMIQ_SOURCE_ID),
        session_id=a.get(attrs.GEN_AI_CONVERSATION_ID),
        start_time=_time(span.start_time),
        end_time=_time(span.end_time),
        parent_run_id=parent.attributes[attrs.DYNAMIQ_RUN_ID] if parent is not None else None,
        # The handler has no paused status: a paused run reports an error to it.
        status=RunStatus.FAILED.value if status == RUN_STATUS_PAUSED else status,
        input=_json(a, attrs.GEN_AI_TOOL_CALL_ARGUMENTS if is_tool else attrs.DYNAMIQ_INPUT),
        output=_json(a, attrs.GEN_AI_TOOL_CALL_RESULT if is_tool else attrs.DYNAMIQ_OUTPUT),
        tags=list(a.get(attrs.DYNAMIQ_TAGS) or []),
    )
    for _, event in _event_attributes(span, attrs.EVENT_EXCEPTION):
        run._error_type = event.get(attrs.EXCEPTION_TYPE)
        run.error = {
            "message": event.get(attrs.EXCEPTION_MESSAGE),
            "traceback": event.get(attrs.EXCEPTION_STACKTRACE),
        }
    if run_type == RunType.NODE.value:
        run.metadata = _node_metadata(span, status, parent is not None, run_ids, compaction)
        run.executions = _executions(span, status)
    else:
        run.metadata = _flow_metadata(a, run_type)
    run.metadata |= {
        key.removeprefix(attrs.DYNAMIQ_METADATA_PREFIX): value
        for key, value in a.items()
        if key.startswith(attrs.DYNAMIQ_METADATA_PREFIX)
    }
    return run


def _flow_metadata(a: Mapping[str, Any], run_type: str) -> dict[str, Any]:
    if run_type == RunType.WORKFLOW.value:
        metadata = {
            "workflow": {"id": a.get(attrs.DYNAMIQ_WORKFLOW_ID), "version": a.get(attrs.DYNAMIQ_WORKFLOW_VERSION)}
        }
    else:
        metadata = {"flow": {"id": a.get(attrs.DYNAMIQ_FLOW_ID)}}
    if (failed_nodes := _json(a, attrs.DYNAMIQ_FAILED_NODES)) is not None:
        metadata["failed_nodes"] = failed_nodes
    return metadata


def _node_metadata(
    span: Any, status: str | None, has_parent: bool, run_ids: dict[str, str], compaction: bool
) -> dict[str, Any]:
    a = span.attributes
    node = _json(a, attrs.DYNAMIQ_NODE_CONFIG)
    if node is None:
        node = {
            key: value
            for key, value in (
                ("id", a.get(attrs.DYNAMIQ_NODE_ID)),
                ("name", a.get(attrs.DYNAMIQ_NODE_NAME)),
                ("type", a.get(attrs.DYNAMIQ_NODE_TYPE)),
                ("group", a.get(attrs.DYNAMIQ_NODE_GROUP)),
                ("model", a.get(attrs.DYNAMIQ_NODE_MODEL)),
            )
            if value is not None
        }
        node["depends"] = list(a.get(attrs.DYNAMIQ_NODE_DEPENDS) or [])
    if a.get(attrs.GEN_AI_OPERATION_NAME) == attrs.OPERATION_CHAT and attrs.GEN_AI_INPUT_MESSAGES in a:
        prompt = node.get("prompt") if isinstance(node.get("prompt"), dict) else {}
        prompt["messages"] = messages_from_parts(_json(a, attrs.GEN_AI_INPUT_MESSAGES))
        if attrs.DYNAMIQ_PROMPT_MESSAGES_BASE_SPAN_ID in a:
            prompt[PROMPT_MESSAGES_BASE_KEY] = {
                "run_id": run_ids.get(a[attrs.DYNAMIQ_PROMPT_MESSAGES_BASE_SPAN_ID]),
                "count": a.get(attrs.DYNAMIQ_PROMPT_MESSAGES_BASE_COUNT),
            }
        elif compaction and has_parent:
            prompt[PROMPT_MESSAGES_BASE_KEY] = None
        node["prompt"] = prompt

    metadata: dict[str, Any] = {"node": node, "run_depends": list(a.get(attrs.DYNAMIQ_RUN_DEPENDS) or [])}
    if (usage := _usage(a)) is not None:
        metadata["usage"] = usage
    if (tool_data := _json(a, attrs.DYNAMIQ_TOOL_DATA)) is not None:
        metadata["tool_data"] = tool_data
    if hooks := [_hook(event) for _, event in _event_attributes(span, attrs.EVENT_AGENT_HOOK)]:
        metadata["hooks"] = hooks
    if (skip := _json(a, attrs.DYNAMIQ_SKIP)) is not None:
        metadata["skip"] = skip
    if (feedback := a.get(attrs.DYNAMIQ_HUMAN_FEEDBACK)) is not None:
        metadata["human_feedback"] = feedback
    # The handler sets the flags where a run ends: both on success, only is_mocked on failure, none on a skip or a
    # cancellation.
    if status == RunStatus.SUCCEEDED.value:
        metadata["is_output_from_cache"] = bool(a.get(attrs.DYNAMIQ_NODE_IS_OUTPUT_FROM_CACHE, False))
    if status in (RunStatus.SUCCEEDED.value, RunStatus.FAILED.value, RUN_STATUS_PAUSED):
        metadata["is_mocked"] = bool(a.get(attrs.DYNAMIQ_NODE_IS_MOCKED, False))
    return metadata


def _usage(a: Mapping[str, Any]) -> dict[str, Any] | None:
    pairs = _MODEL_USAGE if a.get(attrs.GEN_AI_OPERATION_NAME) in _MODEL_OPERATIONS else _OTHER_USAGE
    usage = {field: a[key] for field, key in (pairs | _COMMON_USAGE).items() if key in a}
    return usage or None


def _hook(event: Mapping[str, Any]) -> dict[str, Any]:
    hook = {field: event.get(key) for field, key in _HOOK_FIELDS.items()}
    details = _json(event, attrs.DYNAMIQ_HOOK_DETAILS)
    if isinstance(details, dict):
        hook |= details
    return hook


def _executions(span: Any, status: str | None) -> list[RebuiltExecution]:
    events = _event_attributes(span, attrs.EVENT_NODE_EXECUTION)
    if events:
        executions = []
        for event, a in events:
            failed = a.get(attrs.DYNAMIQ_EXECUTION_STATUS) == RunStatus.FAILED.value
            executions.append(
                RebuiltExecution(
                    id=a.get(attrs.DYNAMIQ_EXECUTION_ID),
                    start_time=_time(a.get(attrs.DYNAMIQ_EXECUTION_START_TIME_UNIX_NANO)),
                    end_time=_time(event.timestamp),
                    status=a.get(attrs.DYNAMIQ_EXECUTION_STATUS),
                    error={"message": a.get(attrs.EXCEPTION_MESSAGE), "traceback": None} if failed else None,
                )
            )
        return executions
    a = span.attributes
    if a.get(attrs.DYNAMIQ_NODE_IS_MOCKED) or a.get(attrs.DYNAMIQ_NODE_IS_OUTPUT_FROM_CACHE):
        return []
    if status == RunStatus.SUCCEEDED.value:
        # The span is the attempt; the attempt's id is not recorded.
        return [RebuiltExecution(start_time=_time(span.start_time), end_time=_time(span.end_time), status=status)]
    if status == RUN_STATUS_PAUSED:
        # The attempt paused: the handler saw it start and never end.
        return [RebuiltExecution(start_time=_time(span.start_time))]
    return []


def _flow_output(flow: RebuiltRun, runs: list[RebuiltRun]) -> dict[str, Any]:
    """``{node id: {status, input, output[, error]}}`` of the node runs directly below a workflow or flow run.

    ``dynamiq.output`` of a workflow or flow span holds the output of the terminal nodes only.
    """
    output: dict[str, Any] = {}
    for run in runs:
        if run.type != RunType.NODE.value or run.parent_run_id != flow.id:
            continue
        entry: dict[str, Any] = {
            "status": _NODE_RESULT_STATUS.get(run.status),
            "input": run.input,
            "output": run.output,
        }
        if run.error is not None:
            entry["error"] = {"type": run._error_type, "message": run.error["message"]}
        output[run.metadata["node"].get("id")] = entry
    return output


# Messages


def _content_part_from(part: Any) -> Any:
    """One OpenAI content part from a convention part (the inverse of ``content._content_part``)."""
    if not isinstance(part, dict):
        return part
    part_type = part.get("type")
    modality = part.get("modality")
    if part_type == attrs.PART_TEXT:
        return {"type": "text", "text": part.get("content")}
    if part_type == attrs.PART_URI:
        if modality == attrs.MODALITY_IMAGE:
            return {"type": "image_url", "image_url": {"url": part.get("uri")}}
        return {"type": "file", "file": {"file_data": part.get("uri")}}
    if part_type == attrs.PART_BLOB:
        mime_type = part.get("mime_type")
        data = part.get("content") or ""
        if modality == attrs.MODALITY_AUDIO:
            audio: dict[str, Any] = {"data": data}
            if mime_type and "/" in mime_type:
                audio["format"] = mime_type.split("/", 1)[1]
            return {"type": "input_audio", "input_audio": audio}
        url = f"data:{mime_type or ''};base64,{data}"
        if modality == attrs.MODALITY_IMAGE:
            return {"type": "image_url", "image_url": {"url": url}}
        return {"type": "file", "file": {"file_data": url}}
    if part_type == attrs.PART_FILE:
        file = {"file_id": part.get("file_id")}
        if part.get("mime_type"):
            file["format"] = part["mime_type"]
        return {"type": "file", "file": file}
    return part


def _tool_call_from(part: dict[str, Any]) -> dict[str, Any]:
    call: dict[str, Any] = {}
    if part.get("id") is not None:
        call["id"] = part["id"]
    call["type"] = attrs.TOOL_TYPE_FUNCTION
    function: dict[str, Any] = {"name": part.get("name")}
    if "arguments" in part:
        arguments = part["arguments"]
        function["arguments"] = arguments if isinstance(arguments, str) else json.dumps(arguments)
    call["function"] = function
    return call


def _is_text(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == attrs.PART_TEXT


def _message_from_parts(message: dict[str, Any]) -> dict[str, Any]:
    content_parts, tool_calls = [], []
    response: dict[str, Any] | None = None
    for part in message.get("parts") or []:
        part_type = part.get("type") if isinstance(part, dict) else None
        if part_type == attrs.PART_TOOL_CALL:
            tool_calls.append(_tool_call_from(part))
        elif part_type == attrs.PART_TOOL_CALL_RESPONSE:
            response = part
        else:
            content_parts.append(part)

    out: dict[str, Any] = {}
    if response is not None:
        value = response.get("response")
        out["content"] = [_content_part_from(item) for item in value] if isinstance(value, list) else value
    elif len(content_parts) == 1 and _is_text(content_parts[0]):
        out["content"] = content_parts[0].get("content")
    elif content_parts:
        out["content"] = [_content_part_from(part) for part in content_parts]
    else:
        out["content"] = ""
    out["role"] = message.get("role")
    if tool_calls:
        out["tool_calls"] = tool_calls
    if response is not None and response.get("id") is not None:
        out["tool_call_id"] = response["id"]
    if message.get("name") is not None:
        out["name"] = message["name"]
    return out
