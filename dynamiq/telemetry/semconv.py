"""Maps Dynamiq runs to span names, kinds, attributes and events.

Every function is pure: it reads the objects it is given and returns plain values. None of them is called
unless a span is being recorded, except the two ``*_span_spec`` functions and ``parent_run_id_of``, which run
before the sampling decision and must stay cheap (no ``to_dict()``, no ``format_value()``, no JSON encoding).

Node, flow and workflow objects are duck typed: importing their modules here would be circular, because they
import the emitter that calls these functions.
"""

import traceback
import warnings
from enum import Enum
from typing import Any

from opentelemetry.trace import SpanKind

from dynamiq.telemetry import attributes as attrs
from dynamiq.telemetry import content
from dynamiq.telemetry.types import CaptureSettings, ErrorSpec, PromptRecord, RunKind, RunOutcome, SpanSpec, SpanType

SEMCONV_VERSION = "1.41.1"
SEMCONV_SCHEMA_URL = "https://opentelemetry.io/schemas/1.41.1"
SHAPE_VERSION = 1
INSTRUMENTATION_SCOPE_NAME = "dynamiq"

EXECUTION_EVENT_NAME = attrs.EVENT_NODE_EXECUTION
HOOK_EVENT_NAME = attrs.EVENT_AGENT_HOOK
EXCEPTION_EVENT_NAME = attrs.EVENT_EXCEPTION
RESUME_LINK_ATTRIBUTES = {attrs.DYNAMIQ_LINK_TYPE: attrs.LINK_TYPE_RESUMED_FROM}

_SPAN_TYPE_BY_GROUP = {
    attrs.NODE_GROUP_LLMS: SpanType.CHAT,
    attrs.NODE_GROUP_EMBEDDERS: SpanType.EMBEDDINGS,
    attrs.NODE_GROUP_RETRIEVERS: SpanType.RETRIEVAL,
    attrs.NODE_GROUP_TOOLS: SpanType.TOOL,
    attrs.NODE_GROUP_AGENTS: SpanType.AGENT,
}

_OPERATION_BY_SPAN_TYPE = {
    SpanType.WORKFLOW: attrs.OPERATION_INVOKE_WORKFLOW,
    SpanType.AGENT: attrs.OPERATION_INVOKE_AGENT,
    SpanType.CHAT: attrs.OPERATION_CHAT,
    SpanType.EMBEDDINGS: attrs.OPERATION_EMBEDDINGS,
    SpanType.RETRIEVAL: attrs.OPERATION_RETRIEVAL,
    SpanType.TOOL: attrs.OPERATION_EXECUTE_TOOL,
}

_CLIENT_SPAN_TYPES = frozenset({SpanType.CHAT, SpanType.EMBEDDINGS, SpanType.RETRIEVAL})
_MODEL_SPAN_TYPES = frozenset({SpanType.CHAT, SpanType.EMBEDDINGS})

# Provider prefix of a model string (the LiteLLM prefix Dynamiq adds through BaseLLM.MODEL_PREFIX) to the
# convention's gen_ai.provider.name value. Any other prefix is recorded without its trailing slash.
_PROVIDER_BY_PREFIX = {
    "openai": attrs.PROVIDER_OPENAI,
    "anthropic": attrs.PROVIDER_ANTHROPIC,
    "azure": attrs.PROVIDER_AZURE_AI_OPENAI,
    "bedrock": attrs.PROVIDER_AWS_BEDROCK,
    "gemini": attrs.PROVIDER_GCP_GEMINI,
    "vertex_ai": attrs.PROVIDER_GCP_VERTEX_AI,
    "cohere": attrs.PROVIDER_COHERE,
    "cohere_chat": attrs.PROVIDER_COHERE,
    "groq": attrs.PROVIDER_GROQ,
    "mistral": attrs.PROVIDER_MISTRAL_AI,
    "deepseek": attrs.PROVIDER_DEEPSEEK,
    "perplexity": attrs.PROVIDER_PERPLEXITY,
    "xai": attrs.PROVIDER_X_AI,
    "watsonx": attrs.PROVIDER_IBM_WATSONX_AI,
}

# Node classes whose provider is not told by their model prefix: they reuse another provider's prefix or add none.
_PROVIDER_BY_LLM_CLASS = {
    "AtlasCloud": "atlascloud",
    "Cohere": attrs.PROVIDER_COHERE,
}

# Embedder node classes, by class name without the DocumentEmbedder / TextEmbedder suffix. Their model strings
# carry a LiteLLM prefix for some providers only (OpenAI and Bedrock models have none).
_PROVIDER_BY_EMBEDDER_CLASS = {
    "OpenAI": attrs.PROVIDER_OPENAI,
    "Bedrock": attrs.PROVIDER_AWS_BEDROCK,
    "Cohere": attrs.PROVIDER_COHERE,
    "Gemini": attrs.PROVIDER_GCP_GEMINI,
    "HuggingFace": "huggingface",
    "Mistral": attrs.PROVIDER_MISTRAL_AI,
    "VertexAI": attrs.PROVIDER_GCP_VERTEX_AI,
    "WatsonX": attrs.PROVIDER_IBM_WATSONX_AI,
}
_EMBEDDER_CLASS_SUFFIXES = ("DocumentEmbedder", "TextEmbedder", "Embedder")
_EMBEDDER_MODEL_PREFIXES = frozenset(_PROVIDER_BY_PREFIX) | {"huggingface"}

_FINISH_REASON_ALIASES = {
    "tool_calls": attrs.FINISH_REASON_TOOL_CALL,
    "function_call": attrs.FINISH_REASON_TOOL_CALL,
}

# Vector store node fields that name what a retriever reads from, in order of preference. A pgvector node has both
# `table_name` (the data) and `index_name` (an index on that table).
_DATA_SOURCE_FIELDS = ("table_name", "collection_name", "index_name")

# Fields of an agent hook event that have their own attribute; the rest are recorded as details.
_HOOK_EVENT_FIELDS = {
    "hook": attrs.DYNAMIQ_HOOK_NAME,
    "type": attrs.DYNAMIQ_HOOK_TYPE,
    "point": attrs.DYNAMIQ_HOOK_POINT,
    "decision": attrs.DYNAMIQ_HOOK_DECISION,
}

# What dynamiq.skip keeps with content capture off (see Node._handle_skip for the shape).
_SKIP_FAILED_DEPENDENCY = "failed_dependency"
_SKIP_NODE_KEYS = ("id", "name", "type")

_STATE_FINISH_REASONS = "finish_reasons"
_STATE_OUTPUT_MESSAGES = "output_messages_recorded"
_STATE_DIMENSIONS = "dimension_count_recorded"


# Attribute values


def _scalar(value: Any) -> str | bool | int | float | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _attribute_value(value: Any) -> Any:
    """``value`` as an OTel attribute value, or None to leave the attribute out."""
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [item for item in (_scalar(item) for item in value) if item is not None]
        kinds = {type(item) for item in items}
        if len(kinds) > 1:
            if kinds <= {int, float}:
                return [float(item) for item in items]
            return [str(item) for item in items]
        return items
    if isinstance(value, dict):
        return content.dumps(value)
    return _scalar(value)


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in attributes.items():
        value = _attribute_value(value)
        if value is not None:
            out[key] = value
    return out


def _value_of(value: Any) -> Any:
    return getattr(value, "value", value)


def _get(data: Any, key: str, default: Any = None) -> Any:
    """``key`` of a dict or an attribute of an object."""
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _str_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


# Span type, provider and model


def span_type_of(node: Any) -> SpanType:
    """Span type of a node, by its group."""
    return _SPAN_TYPE_BY_GROUP.get(_value_of(getattr(node, "group", None)), SpanType.NODE)


def _split_model_prefix(model: str) -> tuple[str | None, str]:
    prefix, sep, rest = model.partition("/")
    if not sep or not prefix or not rest:
        return None, model
    return prefix, rest


def _provider_for_prefix(prefix: str) -> str:
    return _PROVIDER_BY_PREFIX.get(prefix, prefix)


def llm_provider_and_model(node: Any) -> tuple[str | None, str | None]:
    """``(gen_ai.provider.name, gen_ai.request.model)`` of an LLM node."""
    model = getattr(node, "model", None)
    model = model if isinstance(model, str) and model else None
    class_name = type(node).__name__
    model_prefix = getattr(type(node), "MODEL_PREFIX", None)

    request_model = model
    if model and model_prefix and model.startswith(model_prefix):
        request_model = model[len(model_prefix) :]

    if class_name in _PROVIDER_BY_LLM_CLASS:
        return _PROVIDER_BY_LLM_CLASS[class_name], request_model
    if model_prefix:
        return _provider_for_prefix(model_prefix.rstrip("/")), request_model
    provider_prefix = getattr(node, "provider_prefix", None)
    if isinstance(provider_prefix, str) and provider_prefix:
        return _provider_for_prefix(provider_prefix.strip("/")), request_model
    if model:
        prefix, rest = _split_model_prefix(model)
        if prefix is not None:
            return _provider_for_prefix(prefix), rest
    return class_name.lower(), request_model


def embedder_provider_and_model(node: Any) -> tuple[str | None, str | None]:
    """``(gen_ai.provider.name, gen_ai.request.model)`` of an embedder node."""
    model = getattr(node, "model", None)
    model = model if isinstance(model, str) and model else None
    request_model = model
    known_prefix = None
    if model:
        prefix, rest = _split_model_prefix(model)
        # Only a LiteLLM provider prefix is stripped: "BAAI/bge-large" names an organization, not a provider.
        if prefix in _EMBEDDER_MODEL_PREFIXES:
            known_prefix, request_model = prefix, rest

    class_name = type(node).__name__
    for suffix in _EMBEDDER_CLASS_SUFFIXES:
        if class_name.endswith(suffix) and class_name != suffix:
            provider = _PROVIDER_BY_EMBEDDER_CLASS.get(class_name[: -len(suffix)])
            if provider is not None:
                return provider, request_model
            break
    if known_prefix is not None:
        return _provider_for_prefix(known_prefix), request_model
    return class_name.lower(), request_model


def _provider_and_model(span_type: SpanType, node: Any) -> tuple[str | None, str | None]:
    if span_type == SpanType.EMBEDDINGS:
        return embedder_provider_and_model(node)
    return llm_provider_and_model(node)


def _data_source_id(node: Any) -> str | None:
    for field in _DATA_SOURCE_FIELDS:
        value = getattr(node, field, None)
        if isinstance(value, str) and value:
            return value
    return None


def _dependency_ids(depends: Any) -> list[str]:
    ids = []
    for depend in depends or []:
        node = _get(depend, "node")
        node_id = _get(node, "id") if node is not None else None
        if node_id is not None:
            ids.append(str(node_id))
    return ids


# Span specs (cheap: run before the sampling decision)


def scope_attributes() -> dict[str, Any]:
    """Attributes of the instrumentation scope."""
    return {attrs.DYNAMIQ_TRACE_SHAPE_VERSION: SHAPE_VERSION}


def parent_run_id_of(run_kwargs: dict[str, Any], run_id: Any) -> Any | None:
    """Id of the Dynamiq run that started the run ``run_id``, or None when no run did."""
    parent_run_id = run_kwargs.get("parent_run_id")
    # Flow.run_async names the flow run itself as its parent when it has none.
    if parent_run_id is None or str(parent_run_id) == str(run_id):
        return None
    return parent_run_id


def _root_marker(run_kwargs: dict[str, Any], run_id: Any) -> bool | None:
    """``True`` for a run that no Dynamiq run started. None, which leaves the attribute out, for any other: a
    backend that gets the spans of a trace children first tells a top-most run by the attribute being there."""
    return True if parent_run_id_of(run_kwargs, run_id) is None else None


def workflow_span_spec(
    workflow: Any | None, flow: Any | None, *, run_id: str, run_kind: RunKind, run_kwargs: dict[str, Any]
) -> SpanSpec:
    """Span of a workflow run, or of a flow that runs without a workflow (``workflow`` is None)."""
    entity = workflow if workflow is not None else flow
    if flow is None and workflow is not None:
        flow = getattr(workflow, "flow", None)
    name = getattr(entity, "name", None) or run_kind.value
    attributes = {
        attrs.DYNAMIQ_RUN_ID: str(run_id),
        attrs.DYNAMIQ_RUN_TYPE: run_kind.value,
        attrs.DYNAMIQ_RUN_IS_ROOT: _root_marker(run_kwargs, run_id),
        attrs.GEN_AI_OPERATION_NAME: attrs.OPERATION_INVOKE_WORKFLOW,
        attrs.GEN_AI_WORKFLOW_NAME: str(name),
        attrs.DYNAMIQ_FLOW_ID: _str_or_none(getattr(flow, "id", None)),
        attrs.DYNAMIQ_CONFIG_RUN_ID: _str_or_none(run_kwargs.get("wf_run_id")),
    }
    if workflow is not None:
        attributes[attrs.DYNAMIQ_WORKFLOW_ID] = _str_or_none(getattr(workflow, "id", None))
        attributes[attrs.DYNAMIQ_WORKFLOW_VERSION] = _str_or_none(getattr(workflow, "version", None))
    return SpanSpec(
        span_type=SpanType.WORKFLOW,
        name=f"{attrs.OPERATION_INVOKE_WORKFLOW} {name}",
        kind=SpanKind.INTERNAL,
        attributes=_clean(attributes),
    )


def node_span_spec(node: Any, *, run_id: str, run_kwargs: dict[str, Any]) -> SpanSpec:
    """Span of a node run."""
    span_type = span_type_of(node)
    node_name = getattr(node, "name", None) or type(node).__name__
    model = getattr(node, "model", None)
    attributes: dict[str, Any] = {
        attrs.DYNAMIQ_RUN_ID: str(run_id),
        attrs.DYNAMIQ_RUN_TYPE: RunKind.NODE.value,
        attrs.DYNAMIQ_RUN_IS_ROOT: _root_marker(run_kwargs, run_id),
        attrs.DYNAMIQ_NODE_ID: _str_or_none(getattr(node, "id", None)),
        attrs.DYNAMIQ_NODE_NAME: str(node_name),
        attrs.DYNAMIQ_NODE_TYPE: _str_or_none(getattr(node, "type", None)),
        attrs.DYNAMIQ_NODE_GROUP: _str_or_none(_value_of(getattr(node, "group", None))),
        attrs.DYNAMIQ_NODE_MODEL: model if isinstance(model, str) and model else None,
    }
    if depends := _dependency_ids(getattr(node, "depends", None)):
        attributes[attrs.DYNAMIQ_NODE_DEPENDS] = depends
    if run_depends := _dependency_ids(run_kwargs.get("run_depends")):
        attributes[attrs.DYNAMIQ_RUN_DEPENDS] = run_depends

    operation = _OPERATION_BY_SPAN_TYPE.get(span_type)
    if operation is not None:
        attributes[attrs.GEN_AI_OPERATION_NAME] = operation
    name = str(node_name)

    if span_type in _MODEL_SPAN_TYPES:
        provider, request_model = _provider_and_model(span_type, node)
        attributes[attrs.GEN_AI_PROVIDER_NAME] = provider
        attributes[attrs.GEN_AI_SYSTEM] = provider
        attributes[attrs.GEN_AI_REQUEST_MODEL] = request_model
        name = f"{operation} {request_model}" if request_model else operation
        if span_type == SpanType.CHAT and getattr(node, "_is_fallback_run", False) is True:
            attributes[attrs.DYNAMIQ_LLM_IS_FALLBACK] = True
    elif span_type == SpanType.RETRIEVAL:
        data_source = _data_source_id(node)
        attributes[attrs.GEN_AI_DATA_SOURCE_ID] = data_source
        name = f"{operation} {data_source}" if data_source else operation
    elif span_type == SpanType.TOOL:
        attributes[attrs.GEN_AI_TOOL_NAME] = str(node_name)
        attributes[attrs.GEN_AI_TOOL_TYPE] = attrs.TOOL_TYPE_FUNCTION
        name = f"{operation} {node_name}"
    elif span_type == SpanType.AGENT:
        attributes[attrs.GEN_AI_AGENT_NAME] = str(node_name)
        attributes[attrs.GEN_AI_AGENT_ID] = _str_or_none(getattr(node, "id", None))
        llm = _agent_llm(node)
        if llm is not None:
            provider, request_model = llm_provider_and_model(llm)
            attributes[attrs.GEN_AI_PROVIDER_NAME] = provider
            attributes[attrs.GEN_AI_REQUEST_MODEL] = request_model
        name = f"{operation} {node_name}"

    return SpanSpec(
        span_type=span_type,
        name=name,
        kind=SpanKind.CLIENT if span_type in _CLIENT_SPAN_TYPES else SpanKind.INTERNAL,
        attributes=_clean(attributes),
    )


def _agent_llm(node: Any) -> Any | None:
    llm = getattr(node, "llm", None)
    if llm is None:
        # SubAgentTool wraps the agent that has the LLM.
        llm = getattr(getattr(node, "agent", None), "llm", None)
    return llm if llm is not None and hasattr(llm, "model") else None


# Run-level attributes


def run_attributes(settings: CaptureSettings) -> dict[str, Any]:
    """Run-level attributes repeated on every span: session, user, source, tags, metadata."""
    attributes: dict[str, Any] = {}
    if settings.session_id:
        attributes[attrs.GEN_AI_CONVERSATION_ID] = settings.session_id
        attributes[attrs.SESSION_ID] = settings.session_id
    if settings.user_id:
        attributes[attrs.USER_ID] = settings.user_id
    if settings.source_id:
        attributes[attrs.DYNAMIQ_SOURCE_ID] = settings.source_id
    if settings.tags:
        attributes[attrs.DYNAMIQ_TAGS] = [str(tag) for tag in settings.tags]
    for key, value in settings.metadata.items():
        attributes[f"{attrs.DYNAMIQ_METADATA_PREFIX}{key}"] = value
    return _clean(attributes)


def conversation_of(span_type: SpanType, entity: Any, input_data: Any) -> tuple[str | None, str | None]:
    """``(session_id, user_id)`` a run declares in its input (agents), for spans below it to inherit."""
    if span_type != SpanType.AGENT or input_data is None:
        return None, None
    session_id = _get(input_data, "session_id")
    user_id = _get(input_data, "user_id")
    return (str(session_id) if session_id else None), (str(user_id) if user_id else None)


# Input


def _node_config(node: Any, settings: CaptureSettings) -> str | None:
    to_dict = getattr(node, "to_dict", None)
    if not callable(to_dict):
        return None
    data = to_dict(for_tracing=True)
    prompt = data.get("prompt") if isinstance(data, dict) else None
    if isinstance(prompt, dict):
        data["prompt"] = {key: value for key, value in prompt.items() if key != "messages"}
    return content.to_json(data, settings)


def _llm_request_attributes(node: Any, run_kwargs: dict[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for field, key, kind in (
        ("temperature", attrs.GEN_AI_REQUEST_TEMPERATURE, float),
        ("max_tokens", attrs.GEN_AI_REQUEST_MAX_TOKENS, int),
        ("top_p", attrs.GEN_AI_REQUEST_TOP_P, float),
        ("frequency_penalty", attrs.GEN_AI_REQUEST_FREQUENCY_PENALTY, float),
        ("presence_penalty", attrs.GEN_AI_REQUEST_PRESENCE_PENALTY, float),
        ("seed", attrs.GEN_AI_REQUEST_SEED, int),
    ):
        value = getattr(node, field, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attributes[key] = kind(value)
    stop = getattr(node, "stop", None)
    if isinstance(stop, str):
        stop = [stop]
    if stop:
        attributes[attrs.GEN_AI_REQUEST_STOP_SEQUENCES] = [str(item) for item in stop]
    streaming = getattr(node, "streaming", None)
    if getattr(streaming, "enabled", False) is True or getattr(node, "_force_stream", False) is True:
        attributes[attrs.GEN_AI_REQUEST_STREAM] = True
    if _requested_response_format(node, run_kwargs):
        attributes[attrs.GEN_AI_OUTPUT_TYPE] = attrs.OUTPUT_TYPE_JSON
    return attributes


def _prompt_of(node: Any, run_kwargs: dict[str, Any]) -> Any:
    return run_kwargs.get("prompt") or getattr(node, "prompt", None)


def _inference_mode(node: Any) -> str | None:
    # Pydantic warns on every read of a deprecated field.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return _value_of(getattr(node, "inference_mode", None))


def _requested_response_format(node: Any, run_kwargs: dict[str, Any]) -> bool:
    response_format = (
        run_kwargs.get("response_format")
        or getattr(node, "response_format", None)
        or _get(_prompt_of(node, run_kwargs), "response_format")
    )
    if isinstance(response_format, dict) and response_format.get("type") == "text":
        return False
    if response_format:
        return True
    return _inference_mode(node) == "STRUCTURED_OUTPUT" and getattr(node, "schema_", None) is not None


def _requested_tools(node: Any, run_kwargs: dict[str, Any]) -> Any:
    """Tools of the request, chosen as ``BaseLLM._get_response_format_and_tools`` chooses them."""
    tools = run_kwargs.get("tools") or getattr(node, "tools", None) or _get(_prompt_of(node, run_kwargs), "tools")
    if not tools and _inference_mode(node) == "FUNCTION_CALLING":
        schema = getattr(node, "schema_", None)
        if isinstance(schema, list):
            tools = schema
        elif isinstance(schema, dict):
            tools = [schema]
    return tools


def _io_key(span_type: SpanType, is_output: bool) -> str:
    if span_type == SpanType.TOOL:
        return attrs.GEN_AI_TOOL_CALL_RESULT if is_output else attrs.GEN_AI_TOOL_CALL_ARGUMENTS
    return attrs.DYNAMIQ_OUTPUT if is_output else attrs.DYNAMIQ_INPUT


def input_attributes(
    span_type: SpanType,
    entity: Any,
    input_data: Any,
    *,
    run_kwargs: dict[str, Any],
    settings: CaptureSettings,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Attributes known once the input is validated (node start, workflow start)."""
    attributes: dict[str, Any] = {}
    if settings.content_enabled:
        attributes[_io_key(span_type, is_output=False)] = content.to_json(input_data, settings)
    if span_type == SpanType.WORKFLOW:
        return _clean(attributes)

    if settings.node_config_enabled and settings.content_enabled:
        attributes[attrs.DYNAMIQ_NODE_CONFIG] = _node_config(entity, settings)

    if span_type == SpanType.CHAT:
        attributes.update(_llm_request_attributes(entity, run_kwargs))
        if settings.content_enabled and (tools := _requested_tools(entity, run_kwargs)):
            attributes[attrs.GEN_AI_TOOL_DEFINITIONS] = content.to_json(content.tool_definitions(tools), settings)
    elif span_type == SpanType.AGENT:
        description = getattr(entity, "description", None)
        if isinstance(description, str) and description:
            attributes[attrs.GEN_AI_AGENT_DESCRIPTION] = description
        if settings.content_enabled:
            text = _get(input_data, "input")
            message = content.text_message(attrs.ROLE_USER, "" if text is None else text, settings)
            attributes[attrs.GEN_AI_INPUT_MESSAGES] = content.to_json([message], settings)
    elif span_type == SpanType.RETRIEVAL:
        top_k = _get(input_data, "top_k") or getattr(entity, "top_k", None)
        if isinstance(top_k, (int, float)) and not isinstance(top_k, bool) and top_k:
            attributes[attrs.GEN_AI_REQUEST_TOP_K] = float(top_k)
        query = _get(input_data, "query")
        if settings.content_enabled and isinstance(query, str) and query:
            attributes[attrs.GEN_AI_RETRIEVAL_QUERY_TEXT] = content.prepare(query, settings)
    elif span_type == SpanType.EMBEDDINGS:
        dimensions = getattr(entity, "dimensions", None)
        if isinstance(dimensions, int) and not isinstance(dimensions, bool) and dimensions > 0:
            attributes[attrs.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] = dimensions
            state[_STATE_DIMENSIONS] = True
    return _clean(attributes)


# Execution reports


def _usage_attributes(span_type: SpanType, usage: Any) -> dict[str, Any]:
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    attributes: dict[str, Any] = {
        attrs.DYNAMIQ_USAGE_TOTAL_TOKENS: usage.get("total_tokens"),
        attrs.DYNAMIQ_USAGE_INPUT_COST_USD: usage.get("prompt_tokens_cost_usd"),
        attrs.DYNAMIQ_USAGE_OUTPUT_COST_USD: usage.get("completion_tokens_cost_usd"),
        attrs.DYNAMIQ_USAGE_TOTAL_COST_USD: usage.get("total_tokens_cost_usd"),
    }
    for key in (
        attrs.DYNAMIQ_USAGE_INPUT_COST_USD,
        attrs.DYNAMIQ_USAGE_OUTPUT_COST_USD,
        attrs.DYNAMIQ_USAGE_TOTAL_COST_USD,
    ):
        if isinstance(attributes[key], int) and not isinstance(attributes[key], bool):
            attributes[key] = float(attributes[key])
    if span_type in _MODEL_SPAN_TYPES:
        attributes[attrs.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens
        attributes[attrs.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens
        cache_read = usage.get("cache_read_input_tokens")
        # LLM nodes report 0 when the provider reports no cache read.
        if cache_read:
            attributes[attrs.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read
        cache_creation = usage.get("cache_creation_input_tokens")
        if cache_creation is not None:
            attributes[attrs.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] = cache_creation
            attributes[attrs.GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS] = cache_creation
    else:
        attributes[attrs.DYNAMIQ_USAGE_INPUT_TOKENS] = prompt_tokens
        attributes[attrs.DYNAMIQ_USAGE_OUTPUT_TOKENS] = completion_tokens
    return attributes


def _finish_reason(value: Any) -> str | None:
    if value is None:
        return None
    value = str(_value_of(value))
    return _FINISH_REASON_ALIASES.get(value, value)


def _shared_start(previous: list[Any], messages: list[Any]) -> int:
    """Number of messages at the start of ``messages`` that equal the ones at the start of ``previous``."""
    shared = 0
    for previous_message, message in zip(previous, messages):
        if previous_message != message:
            break
        shared += 1
    return shared


def prompt_attributes(
    messages: Any, *, settings: CaptureSettings, previous: PromptRecord | None = None
) -> dict[str, Any]:
    """Attributes for the prompt an LLM run reports (``prompt_messages``).

    ``previous`` is the prompt this one is compared with under prompt compaction. The messages the two share at
    their start are left out, and the span says where they are recorded: the span of ``previous``, and how many
    messages of that span's whole prompt. A prompt that shares none is recorded whole.
    """
    if not settings.content_enabled:
        return {}
    attributes: dict[str, Any] = {}
    shared = _shared_start(previous.messages, messages) if previous is not None else 0
    if shared:
        messages = messages[shared:]
        attributes[attrs.DYNAMIQ_PROMPT_MESSAGES_BASE_SPAN_ID] = previous.span_id
        attributes[attrs.DYNAMIQ_PROMPT_MESSAGES_BASE_COUNT] = shared
    attributes[attrs.GEN_AI_INPUT_MESSAGES] = content.to_json(content.input_messages(messages, settings), settings)
    return attributes


def execute_run_attributes(
    span_type: SpanType, entity: Any, *, run_kwargs: dict[str, Any], settings: CaptureSettings, state: dict[str, Any]
) -> dict[str, Any]:
    """Attributes for what ``run_on_node_execute_run`` reports in ``run_kwargs``: ``usage_data``,
    ``prompt_messages``, ``tool_data``, ``response_data``."""
    attributes: dict[str, Any] = {}
    if (usage := run_kwargs.get("usage_data")) is not None:
        attributes.update(_usage_attributes(span_type, usage))

    if (messages := run_kwargs.get("prompt_messages")) is not None:
        attributes.update(prompt_attributes(messages, settings=settings))

    if (tool_data := run_kwargs.get("tool_data")) is not None:
        attributes[attrs.DYNAMIQ_TOOL_DATA] = content.to_json(tool_data, settings)

    if span_type == SpanType.EMBEDDINGS and isinstance(response_data := run_kwargs.get("response_data"), dict):
        # Text embedders report the embedding component's meta here, as their output leaves it out.
        attributes.update(_embedding_meta_attributes(response_data))
    elif isinstance(response_data := run_kwargs.get("response_data"), dict):
        attributes[attrs.GEN_AI_RESPONSE_ID] = _str_or_none(response_data.get("id"))
        attributes[attrs.GEN_AI_RESPONSE_MODEL] = _str_or_none(response_data.get("model"))
        reasons = [_finish_reason(reason) for reason in response_data.get("finish_reasons") or []]
        reasons = [reason for reason in reasons if reason is not None]
        if reasons:
            attributes[attrs.GEN_AI_RESPONSE_FINISH_REASONS] = reasons
            state[_STATE_FINISH_REASONS] = reasons
    return _clean(attributes)


def hook_events(run_kwargs: dict[str, Any], *, settings: CaptureSettings) -> list[tuple[str, dict[str, Any]]]:
    """``(event name, attributes)`` for each entry of ``run_kwargs["hook_events"]``."""
    events = []
    for event in run_kwargs.get("hook_events") or []:
        if not isinstance(event, dict):
            continue
        attributes: dict[str, Any] = {key: event.get(field) for field, key in _HOOK_EVENT_FIELDS.items()}
        details = {field: value for field, value in event.items() if field not in _HOOK_EVENT_FIELDS}
        if settings.content_enabled and details:
            attributes[attrs.DYNAMIQ_HOOK_DETAILS] = content.to_json(details, settings)
        events.append((HOOK_EVENT_NAME, _clean(attributes)))
    return events


def _chat_output_attributes(output: Any, settings: CaptureSettings, state: dict[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if not isinstance(output, dict):
        return attributes
    reasons = state.get(_STATE_FINISH_REASONS)
    if reasons:
        finish_reason = reasons[0]
    elif output.get("finish_reason") is not None:
        finish_reason = _finish_reason(output["finish_reason"])
    else:
        finish_reason = attrs.FINISH_REASON_TOOL_CALL if output.get("tool_calls") else attrs.FINISH_REASON_STOP
    if not reasons:
        attributes[attrs.GEN_AI_RESPONSE_FINISH_REASONS] = [finish_reason]
    if settings.content_enabled:
        message = content.output_message(output, finish_reason, settings)
        attributes[attrs.GEN_AI_OUTPUT_MESSAGES] = content.to_json([message], settings)
    state[_STATE_OUTPUT_MESSAGES] = True
    return attributes


def _embedding_meta_attributes(meta: dict[str, Any]) -> dict[str, Any]:
    """Model and usage of an embedding component's ``meta`` (``{"model", "usage"}``)."""
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    return {
        attrs.GEN_AI_RESPONSE_MODEL: _str_or_none(meta.get("model")),
        attrs.GEN_AI_USAGE_INPUT_TOKENS: usage.get("prompt_tokens"),
        attrs.DYNAMIQ_USAGE_TOTAL_TOKENS: usage.get("total_tokens"),
    }


def _embedding_length(output: dict[str, Any]) -> int | None:
    embedding = output.get("embedding")
    if not isinstance(embedding, list):
        documents = output.get("documents")
        first = documents[0] if isinstance(documents, list) and documents else None
        embedding = _get(first, "embedding") if first is not None else None
    return len(embedding) if isinstance(embedding, list) and embedding else None


def execute_output_attributes(
    span_type: SpanType, entity: Any, output: Any, *, settings: CaptureSettings, state: dict[str, Any]
) -> dict[str, Any]:
    """Attributes read from the raw ``execute()`` result of a successful attempt (LLM output messages,
    embedding usage)."""
    attributes: dict[str, Any] = {}
    if span_type == SpanType.CHAT:
        attributes.update(_chat_output_attributes(output, settings, state))
    elif span_type == SpanType.EMBEDDINGS and isinstance(output, dict):
        if isinstance(meta := output.get("meta"), dict):
            attributes.update(_embedding_meta_attributes(meta))
        if not state.get(_STATE_DIMENSIONS):
            attributes[attrs.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] = _embedding_length(output)
    elif span_type == SpanType.RETRIEVAL and isinstance(output, dict) and settings.content_enabled:
        documents = output.get("documents")
        if isinstance(documents, list):
            attributes[attrs.GEN_AI_RETRIEVAL_DOCUMENTS] = content.to_json(
                content.retrieval_documents(documents), settings
            )
    return _clean(attributes)


# Outcome


def _terminal_output(flow: Any, output_data: Any) -> Any:
    """Output of the flow's terminal nodes: the output itself for one, else an object keyed by node id."""
    nodes = getattr(flow, "nodes", None) or []
    if not isinstance(output_data, dict) or not nodes:
        return output_data
    depended_on = {node_id for node in nodes for node_id in _dependency_ids(getattr(node, "depends", None))}
    terminal = [str(getattr(node, "id", "")) for node in nodes if str(getattr(node, "id", "")) not in depended_on]
    results = {node_id: output_data[node_id] for node_id in terminal if node_id in output_data}
    if not results:
        return output_data
    outputs = {node_id: _get(result, "output") for node_id, result in results.items()}
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    return outputs


def output_attributes(
    span_type: SpanType,
    entity: Any,
    output_data: Any,
    *,
    run_kwargs: dict[str, Any],
    settings: CaptureSettings,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Attributes known when the run succeeds (node end, workflow end)."""
    attributes: dict[str, Any] = {}
    if span_type == SpanType.WORKFLOW:
        if settings.content_enabled:
            flow = getattr(entity, "flow", None) or entity
            attributes[attrs.DYNAMIQ_OUTPUT] = content.to_json(_terminal_output(flow, output_data), settings)
        return _clean(attributes)

    if settings.content_enabled:
        attributes[_io_key(span_type, is_output=True)] = content.to_json(output_data, settings)
    if span_type == SpanType.AGENT and settings.content_enabled:
        answer = _get(output_data, "content") if isinstance(output_data, dict) else output_data
        message = content.text_message(
            attrs.ROLE_ASSISTANT, "" if answer is None else answer, settings, attrs.FINISH_REASON_STOP
        )
        attributes[attrs.GEN_AI_OUTPUT_MESSAGES] = content.to_json([message], settings)
    elif span_type == SpanType.CHAT and not state.get(_STATE_OUTPUT_MESSAGES):
        # A mocked or cached run has no execute() result; its output is the closest record of the response.
        attributes.update(_chat_output_attributes(output_data, settings, state))
    return _clean(attributes)


def outcome_attributes(outcome: RunOutcome, *, run_kwargs: dict[str, Any]) -> dict[str, Any]:
    """``dynamiq.run.status`` plus the mocked and cached flags found in ``run_kwargs``."""
    attributes: dict[str, Any] = {attrs.DYNAMIQ_RUN_STATUS: outcome.value}
    if run_kwargs.get("is_mocked") is True:
        attributes[attrs.DYNAMIQ_NODE_IS_MOCKED] = True
    if run_kwargs.get("is_output_from_cache") is True:
        attributes[attrs.DYNAMIQ_NODE_IS_OUTPUT_FROM_CACHE] = True
    return attributes


def _skip_identifiers(skip_data: Any) -> dict[str, Any] | None:
    """The failed dependency's node id, name and type and its option: the parts of ``skip_data`` that are not
    content. A dependency's condition holds values from the workflow definition, which can be the user's data."""
    dependency = _get(skip_data, _SKIP_FAILED_DEPENDENCY) if skip_data is not None else None
    if dependency is None:
        return None
    node = _get(dependency, "node")
    out: dict[str, Any] = {"node": {key: _get(node, key) for key in _SKIP_NODE_KEYS} if node is not None else None}
    out["option"] = _get(dependency, "option")
    return {_SKIP_FAILED_DEPENDENCY: out}


def skip_attributes(
    skip_data: Any,
    input_data: Any,
    human_feedback: str | None,
    *,
    span_type: SpanType,
    entity: Any,
    settings: CaptureSettings,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Attributes of a skipped run."""
    attributes: dict[str, Any] = {}
    if not settings.content_enabled:
        skip_data = _skip_identifiers(skip_data)
    if skip_data is not None:
        attributes[attrs.DYNAMIQ_SKIP] = content.to_json(skip_data, settings)
    if settings.content_enabled:
        if input_data is not None:
            attributes[_io_key(span_type, is_output=False)] = content.to_json(input_data, settings)
        if human_feedback:
            attributes[attrs.DYNAMIQ_HUMAN_FEEDBACK] = content.prepare(str(human_feedback), settings)
        # A node skipped before it starts never reaches input_attributes, where the configuration is recorded.
        if settings.node_config_enabled and span_type != SpanType.WORKFLOW:
            attributes[attrs.DYNAMIQ_NODE_CONFIG] = _node_config(entity, settings)
    return _clean(attributes)


def failure_attributes(run_kwargs: dict[str, Any], *, settings: CaptureSettings) -> dict[str, Any]:
    """Attributes a failed workflow or flow reports in ``run_kwargs`` (``failed_nodes``)."""
    failed_nodes = []
    for node in run_kwargs.get("failed_nodes") or []:
        if hasattr(node, "model_dump"):
            node = node.model_dump()
        if not isinstance(node, dict):
            continue
        item = {"id": node.get("id"), "name": node.get("name")}
        # An error message can hold user data, so it is content.
        if settings.content_enabled:
            item["error_message"] = node.get("error_message")
        failed_nodes.append(item)
    if not failed_nodes:
        return {}
    return {attrs.DYNAMIQ_FAILED_NODES: content.to_json(failed_nodes, settings)}


def resume_attributes(run_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Attributes of a run resumed from a checkpoint (``resumed_from_checkpoint``, ``original_run_id``)."""
    return _clean(
        {
            attrs.DYNAMIQ_RESUME_CHECKPOINT_ID: _str_or_none(run_kwargs.get("resumed_from_checkpoint")),
            attrs.DYNAMIQ_RESUME_ORIGINAL_RUN_ID: _str_or_none(run_kwargs.get("original_run_id")),
        }
    )


def flow_run_attributes(flow_run_id: Any) -> dict[str, Any]:
    """Attributes a workflow span gets from the flow run it represents, which has no span of its own."""
    return _clean({attrs.DYNAMIQ_FLOW_RUN_ID: _str_or_none(flow_run_id)})


def pause_attributes(entity: Any) -> dict[str, Any]:
    """Attributes of a paused workflow or flow run: the id of the checkpoint to resume from."""
    flow = getattr(entity, "flow", None) or entity
    checkpoint = getattr(flow, "_checkpoint", None)
    return _clean({attrs.DYNAMIQ_CHECKPOINT_ID: _str_or_none(getattr(checkpoint, "id", None))})


def error_spec(error: BaseException | None, outcome: RunOutcome, *, settings: CaptureSettings) -> ErrorSpec | None:
    """How to record ``error`` on a span that ends with ``outcome``; None when nothing is recorded (paused)."""
    if outcome == RunOutcome.CANCELED:
        return ErrorSpec(error_type=attrs.CANCELED_ERROR_TYPE)
    if outcome != RunOutcome.FAILED:
        return None
    if error is None:
        return ErrorSpec(
            error_type=attrs.ERROR_TYPE_OTHER, event_attributes={attrs.EXCEPTION_TYPE: attrs.ERROR_TYPE_OTHER}
        )
    error_type = type(error).__name__
    event_attributes: dict[str, Any] = {attrs.EXCEPTION_TYPE: error_type}
    description = None
    if settings.content_enabled:
        description = content.prepare(str(error), settings)
        event_attributes[attrs.EXCEPTION_MESSAGE] = description
        stacktrace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        event_attributes[attrs.EXCEPTION_STACKTRACE] = content.prepare(stacktrace, settings)
    return ErrorSpec(error_type=error_type, description=description or None, event_attributes=event_attributes)


def execution_event(
    *,
    execution_id: str,
    attempt: int,
    succeeded: bool,
    start_time_unix_nano: int,
    error: BaseException | None,
    settings: CaptureSettings,
) -> dict[str, Any]:
    """Attributes of one ``dynamiq.node.execution`` event."""
    attributes: dict[str, Any] = {
        attrs.DYNAMIQ_EXECUTION_ID: str(execution_id),
        attrs.DYNAMIQ_EXECUTION_ATTEMPT: int(attempt),
        attrs.DYNAMIQ_EXECUTION_STATUS: (
            attrs.EXECUTION_STATUS_SUCCEEDED if succeeded else attrs.EXECUTION_STATUS_FAILED
        ),
        attrs.DYNAMIQ_EXECUTION_START_TIME_UNIX_NANO: int(start_time_unix_nano),
    }
    if not succeeded and error is not None:
        attributes[attrs.ERROR_TYPE] = type(error).__name__
        if settings.content_enabled:
            attributes[attrs.EXCEPTION_MESSAGE] = content.prepare(str(error), settings)
    return attributes
