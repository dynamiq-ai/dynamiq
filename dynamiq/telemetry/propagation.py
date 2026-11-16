"""Trace context that leaves the process: to worker processes, into checkpoints and to MCP servers.

Like the emitter, nothing here raises into a run: a carrier that cannot be built or read is logged and skipped.
"""

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import context as context_api
from opentelemetry import propagate, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.context import Context
from opentelemetry.trace import SpanContext
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from dynamiq.telemetry import registry
from dynamiq.utils.logger import logger

TRACEPARENT = "traceparent"

# Checkpoints and MCP requests carry W3C trace context whatever propagators the application configured: a
# checkpoint outlives the process that wrote it, and MCP names the W3C keys.
_TRACE_CONTEXT = TraceContextTextMapPropagator()
_BAGGAGE = W3CBaggagePropagator()


def carrier_for_run(run_id: Any) -> dict[str, str] | None:
    """Return a carrier of the span of a run, for a worker process to continue the trace from.

    It is written by the globally configured propagators, which the worker reads it with. None when the run is
    not traced in this process.
    """
    try:
        entry = registry.get(run_id)
        if entry is None:
            return None
        carrier: dict[str, str] = {}
        propagate.inject(carrier, context=entry.context)
        return carrier or None
    except Exception as e:
        logger.warning(f"Telemetry: trace context of run {run_id} was not passed on: {e!r}")
        return None


@contextmanager
def attached_carrier(carrier: dict[str, str] | None) -> Iterator[None]:
    """Make the trace context of ``carrier`` current for the block. Does nothing for None or an empty carrier."""
    token = None
    if carrier:
        try:
            token = context_api.attach(propagate.extract(carrier))
        except Exception as e:
            logger.warning(f"Telemetry: trace context of the parent process was not restored: {e!r}")
    try:
        yield
    finally:
        if token is not None:
            context_api.detach(token)


def traceparent_for_run(run_id: Any) -> str | None:
    """Return the W3C ``traceparent`` of the span of a run, or None when the run has no valid span context."""
    try:
        entry = registry.get(run_id)
        if entry is None:
            return None
        carrier: dict[str, str] = {}
        _TRACE_CONTEXT.inject(carrier, context=entry.context)
        return carrier.get(TRACEPARENT)
    except Exception as e:
        logger.warning(f"Telemetry: traceparent of run {run_id} was not built: {e!r}")
        return None


def span_context_from_traceparent(value: str) -> SpanContext | None:
    """Parse a W3C ``traceparent``. None when it is not one."""
    if not value or not isinstance(value, str):
        return None
    try:
        # An empty context, so that a value that does not parse cannot yield the span that is current here.
        context = _TRACE_CONTEXT.extract({TRACEPARENT: value}, context=Context())
        span_context = trace.get_current_span(context).get_span_context()
        return span_context if span_context.is_valid else None
    except Exception as e:
        logger.warning(f"Telemetry: traceparent {value!r} was not parsed: {e!r}")
        return None


def mcp_request_meta() -> dict[str, str] | None:
    """Return ``traceparent``, ``tracestate`` and ``baggage`` of the current context for the ``_meta`` of an MCP
    request, or None when the context holds none of them."""
    try:
        carrier: dict[str, str] = {}
        _TRACE_CONTEXT.inject(carrier)
        _BAGGAGE.inject(carrier)
        return carrier or None
    except Exception as e:
        logger.warning(f"Telemetry: trace context was not added to the MCP request: {e!r}")
        return None
