"""Resolves the tracer and the capture settings of a run tree from ``RunnableConfig.tracing`` and the environment.

Both are resolved once, by the run at the root of a tree; the runs below it reuse what the root resolved.
"""

import os
import threading
import weakref
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import NoOpTracer, NoOpTracerProvider, ProxyTracerProvider, Tracer

# The module, not its names: semconv can be the first telemetry module imported, and it imports dynamiq.nodes,
# which imports the emitter and this module while semconv is still being initialized.
from dynamiq.telemetry import semconv
from dynamiq.telemetry.types import CaptureSettings
from dynamiq.types.telemetry import ContentCaptureMode
from dynamiq.utils.env import get_env_var
from dynamiq.utils.logger import logger
from dynamiq.utils.user_agent import VERSION

TRACING_ENABLED_ENV_VAR = "DYNAMIQ_TRACING_ENABLED"
CAPTURE_CONTENT_ENV_VAR = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT"
CAPTURE_MESSAGE_CONTENT_ENV_VAR = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
BINARY_CONTENT_ENABLED_ENV_VAR = "DYNAMIQ_TRACING_BINARY_CONTENT_ENABLED"
NODE_CONFIG_ENABLED_ENV_VAR = "DYNAMIQ_TRACING_NODE_CONFIG_ENABLED"
CONTENT_MAX_LENGTH_ENV_VAR = "DYNAMIQ_TRACING_CONTENT_MAX_LENGTH"
PROMPT_COMPACTION_ENABLED_ENV_VAR = "DYNAMIQ_TRACING_PROMPT_COMPACTION_ENABLED"

_TRUE_VALUES = frozenset({"true", "1"})
_FALSE_VALUES = frozenset({"false", "0"})
_CONTENT_ON_VALUES = _TRUE_VALUES | {"span_only", "span_and_event"}
# Dynamiq emits no log-record events, so the event-only mode records nothing.
_CONTENT_OFF_VALUES = _FALSE_VALUES | {"no_content", "event_only"}

# One tracer per provider. The SDK's get_tracer() rewrites the process-wide warning filters on every call, so it
# is called once per provider rather than once per run. Weak keys let a discarded provider go.
_tracers: "weakref.WeakKeyDictionary[Any, Tracer]" = weakref.WeakKeyDictionary()
_warned_variables: set[str] = set()
_lock = threading.Lock()


def resolve_tracer(config: Any) -> Tracer | None:
    """Return the tracer for a run tree that starts with ``config``, or None when the tree is not traced.

    A tree is not traced when ``config.tracing.enabled`` is false, when ``DYNAMIQ_TRACING_ENABLED`` is false, or
    when no OpenTelemetry SDK provider is set (on the config, else globally) or the SDK is disabled.
    """
    tracing = getattr(config, "tracing", None)
    provider = None
    if tracing is not None:
        if not tracing.enabled:
            return None
        provider = tracing.tracer_provider
    if provider is None:
        provider = trace.get_tracer_provider()
    # The API's own providers only make non-recording spans: nothing would be exported.
    if isinstance(provider, (ProxyTracerProvider, NoOpTracerProvider)):
        return None
    if not _env_flag(TRACING_ENABLED_ENV_VAR, default=True):
        return None
    tracer = _tracer_of(provider)
    # What an SDK provider returns under OTEL_SDK_DISABLED=true.
    return None if isinstance(tracer, NoOpTracer) else tracer


def resolve_settings(config: Any) -> CaptureSettings:
    """Return what the run tree that starts with ``config`` records.

    A field set on ``config.tracing`` wins, then the environment variable, then the default. The environment is
    read on every call.
    """
    tracing = getattr(config, "tracing", None)
    content_enabled = _content_enabled(tracing)

    binary_content_enabled = getattr(tracing, "binary_content_enabled", None)
    if binary_content_enabled is None:
        binary_content_enabled = _env_flag(BINARY_CONTENT_ENABLED_ENV_VAR, default=False)

    node_config_enabled = getattr(tracing, "node_config_enabled", None)
    if node_config_enabled is None:
        node_config_enabled = _env_flag(NODE_CONFIG_ENABLED_ENV_VAR, default=False)

    content_max_length = getattr(tracing, "content_max_length", None)
    if content_max_length is None:
        content_max_length = _env_positive_int(CONTENT_MAX_LENGTH_ENV_VAR)

    prompt_compaction_enabled = getattr(tracing, "prompt_compaction_enabled", None)
    if prompt_compaction_enabled is None:
        prompt_compaction_enabled = _env_flag(PROMPT_COMPACTION_ENABLED_ENV_VAR, default=False)

    return CaptureSettings(
        content_enabled=content_enabled,
        binary_content_enabled=binary_content_enabled,
        # A node's configuration holds its prompt and tool descriptions, so it is content.
        node_config_enabled=node_config_enabled and content_enabled,
        content_max_length=content_max_length,
        prompt_compaction_enabled=prompt_compaction_enabled,
        session_id=getattr(tracing, "session_id", None),
        user_id=getattr(tracing, "user_id", None),
        source_id=getattr(tracing, "source_id", None),
        tags=tuple(getattr(tracing, "tags", ())),
        metadata=dict(getattr(tracing, "metadata", {})),
    )


def _tracer_of(provider: Any) -> Tracer:
    with _lock:
        try:
            tracer = _tracers.get(provider)
            if tracer is None:
                tracer = _tracers[provider] = _new_tracer(provider)
        except TypeError:
            # A provider that cannot be hashed or weakly referenced is not cached.
            tracer = _new_tracer(provider)
    return tracer


def _new_tracer(provider: Any) -> Tracer:
    return provider.get_tracer(
        semconv.INSTRUMENTATION_SCOPE_NAME,
        VERSION,
        schema_url=semconv.SEMCONV_SCHEMA_URL,
        attributes=semconv.scope_attributes(),
    )


def _content_enabled(tracing: Any) -> bool:
    mode = getattr(tracing, "content_capture", None)
    if mode is not None:
        return mode == ContentCaptureMode.SPAN_ONLY
    for variable in (CAPTURE_CONTENT_ENV_VAR, CAPTURE_MESSAGE_CONTENT_ENV_VAR):
        value = _env_value(variable)
        if not value:
            continue
        if value in _CONTENT_ON_VALUES:
            return True
        if value in _CONTENT_OFF_VALUES:
            return False
        _warn_once(variable, f"{variable}={value!r} is not a content capture mode; content is captured.")
        return True
    return True


def _env_value(variable: str) -> str:
    # The default keeps get_env_var from warning that an optional variable is not set.
    return str(get_env_var(variable, "")).strip().lower()


def _env_flag(variable: str, default: bool) -> bool:
    value = _env_value(variable)
    if not value:
        return default
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    _warn_once(variable, f"{variable}={value!r} is neither true nor false; using {str(default).lower()}.")
    return default


def _env_positive_int(variable: str) -> int | None:
    value = _env_value(variable)
    if not value:
        return None
    if value.isdecimal() and int(value) > 0:
        return int(value)
    _warn_once(variable, f"{variable}={value!r} is not a positive integer; content is not truncated.")
    return None


def _warn_once(variable: str, message: str) -> None:
    with _lock:
        if variable in _warned_variables:
            return
        _warned_variables.add(variable)
    logger.warning(message)


def _reset_after_fork() -> None:
    # The lock may have been held by a thread that does not exist in the child.
    global _lock
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
