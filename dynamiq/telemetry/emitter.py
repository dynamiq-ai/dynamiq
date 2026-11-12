"""Span lifecycle of workflow, flow and node runs.

Run code calls these entry points where it already runs its callbacks, and passes the live objects. An entry
point finds the span of the run in the registry by run id. While no run is traced it returns after one check, and
it calls the mapping functions (``semconv``), which serialize, only for a span that records. Nothing raised here
reaches the run.
"""

import asyncio
import functools
import sys
import time
from typing import Any, Callable

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Tracer

from dynamiq.checkpoints.types import RunPausedException

# The modules, not their names: a mapping module that comes to import node code would import this module while it
# is still being initialized. Their names are read when a run calls in.
from dynamiq.telemetry import attributes as attrs
from dynamiq.telemetry import interop, propagation, registry, semconv, settings
from dynamiq.telemetry.registry import ENTRIES, RunEntry
from dynamiq.telemetry.types import CaptureSettings, PromptRecord, RunKind, RunOutcome, SpanSpec, SpanType
from dynamiq.types.cancellation import CanceledException
from dynamiq.utils.logger import logger

# Attributes the lifecycle sets itself, because their values are known only here: the mapping functions never
# see the flow run id of a workflow, the checkpoint a run paused on, or the ErrorSpec they returned.
ERROR_TYPE_ATTRIBUTE = attrs.ERROR_TYPE
FLOW_RUN_ID_ATTRIBUTE = attrs.DYNAMIQ_FLOW_RUN_ID
CHECKPOINT_ID_ATTRIBUTE = attrs.DYNAMIQ_CHECKPOINT_ID

# Exceptions that end a run because it was told to stop. GeneratorExit is a coroutine closed before it finished.
_CANCELLATIONS = (asyncio.CancelledError, CanceledException, GeneratorExit)
_ERROR_OUTCOMES = (RunOutcome.FAILED, RunOutcome.CANCELED)

# Sites already reported at warning level. Not locked: set.add is atomic, and a race costs a second warning.
_reported_failures: set[str] = set()


def _log_failure(site: str, error: BaseException) -> None:
    # A mapping that fails does so on every run, so only the first failure of a site is a warning.
    if site in _reported_failures:
        logger.debug(f"Telemetry: {site} failed again: {error!r}")
        return
    _reported_failures.add(site)
    logger.warning(f"Telemetry: {site} failed; the run is not affected. {error!r}", exc_info=error)


def _never_raises(entry_point: Callable[..., Any]) -> Callable[..., Any]:
    """Make an entry point log what it raises instead of raising it into the run."""

    @functools.wraps(entry_point)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            return entry_point(*args, **kwargs)
        except Exception as e:
            _log_failure(entry_point.__name__, e)
            return None

    return guarded


def _for_traced_runs(entry_point: Callable[..., Any]) -> Callable[..., Any]:
    """Like ``_never_raises``, for an entry point that acts only on a registered run: while the registry is empty
    it returns after that one check."""

    @functools.wraps(entry_point)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if not ENTRIES:
            return None
        try:
            return entry_point(*args, **kwargs)
        except Exception as e:
            _log_failure(entry_point.__name__, e)
            return None

    return guarded


# Node runs


@_never_raises
def start_node_run(node: Any, config: Any, run_kwargs: dict[str, Any]) -> RunEntry | None:
    """Start the span of a node run and make it current. Returns None when the run is not traced.

    The caller passes the result to ``finish_node_run`` in a ``finally`` that covers everything after this call.
    """
    run_id = run_kwargs.get("run_id")
    parent = _parent_of(run_kwargs, run_id) if ENTRIES else None
    tracing = _tracing_for(config, parent)
    if tracing is None:
        return None
    key = str(run_id)
    spec = semconv.node_span_spec(node, run_id=key, run_kwargs=run_kwargs)
    entry = _open(key, spec, RunKind.NODE, tracing, parent)
    if entry.recording and spec.span_type == SpanType.CHAT:
        _check_interop()
    return entry


def finish_node_run(scope: RunEntry | None) -> None:
    """Close what ``start_node_run`` opened: restore the context, end the span if no hook did, drop the entry."""
    if scope is None:
        return
    try:
        if scope.token is not None:
            context_api.detach(scope.token)
    except Exception as e:
        _log_failure("finish_node_run", e)
    try:
        if not scope.ended:
            # No hook reported an outcome, so the run is leaving through an exception that run code does not
            # turn into a result, such as asyncio.CancelledError. That exception is the outcome.
            outcome, error = _unreported_outcome(sys.exc_info()[1])
            _finish(scope, outcome, {}, error)
    except Exception as e:
        _log_failure("finish_node_run", e)
    finally:
        registry.pop(scope.run_id)


@_for_traced_runs
def on_node_start(node: Any, input_data: Any, run_kwargs: dict[str, Any]) -> None:
    entry = _recording_entry(run_kwargs)
    if entry is None:
        return
    _inherit_conversation(entry, node, input_data)
    _record_input(entry, node, input_data, run_kwargs)


@_for_traced_runs
def on_node_end(node: Any, output_data: Any, run_kwargs: dict[str, Any]) -> None:
    entry = ENTRIES.get(str(run_kwargs.get("run_id")))
    if entry is None:
        return
    _finish(
        entry,
        RunOutcome.SUCCEEDED,
        run_kwargs,
        record=lambda: _set(
            entry,
            semconv.output_attributes,
            entry.span_type,
            node,
            output_data,
            run_kwargs=run_kwargs,
            settings=entry.settings,
            state=entry.state,
        ),
    )


@_for_traced_runs
def on_node_error(node: Any, error: BaseException, run_kwargs: dict[str, Any]) -> None:
    entry = ENTRIES.get(str(run_kwargs.get("run_id")))
    if entry is None:
        return

    def record() -> None:
        # A run that fails before its input is validated never reports a start; the error hook brings the input.
        if not entry.input_recorded and "input_data" in run_kwargs:
            _record_input(entry, node, run_kwargs["input_data"], run_kwargs)

    _finish(entry, _failure_outcome(error), run_kwargs, error, record)


@_for_traced_runs
def on_node_skip(
    node: Any, skip_data: Any, input_data: Any, human_feedback: str | None, run_kwargs: dict[str, Any]
) -> None:
    entry = ENTRIES.get(str(run_kwargs.get("run_id")))
    if entry is None:
        return
    _finish(
        entry,
        RunOutcome.SKIPPED,
        run_kwargs,
        record=lambda: _set(
            entry,
            semconv.skip_attributes,
            skip_data,
            input_data,
            human_feedback,
            span_type=entry.span_type,
            entity=node,
            settings=entry.settings,
            state=entry.state,
        ),
    )


@_for_traced_runs
def on_node_canceled(node: Any, run_kwargs: dict[str, Any]) -> None:
    entry = ENTRIES.get(str(run_kwargs.get("run_id")))
    if entry is not None:
        _finish(entry, RunOutcome.CANCELED, run_kwargs)


@_for_traced_runs
def on_node_execute_start(node: Any, run_kwargs: dict[str, Any]) -> None:
    entry = _recording_entry(run_kwargs)
    if entry is None:
        return
    execution_id = run_kwargs.get("execution_run_id")
    entry.attempts += 1
    entry.attempt_started[str(execution_id)] = time.time_ns()
    entry.attempt_id = str(execution_id) if execution_id is not None else None
    entry.attempt_open = True


@_for_traced_runs
def on_node_execute_end(node: Any, output_data: Any, run_kwargs: dict[str, Any]) -> None:
    entry = _attempt_entry(run_kwargs)
    if entry is None:
        return
    entry.attempt_open = False
    _set(
        entry,
        semconv.execute_output_attributes,
        entry.span_type,
        node,
        output_data,
        settings=entry.settings,
        state=entry.state,
    )
    # A run that succeeds at once has no event: the span is the attempt. After a failed attempt every attempt
    # has one, the successful one included.
    if entry.attempts > 1:
        _add_execution_event(entry, run_kwargs, error=None)
    else:
        entry.attempt_started.pop(str(run_kwargs.get("execution_run_id")), None)


@_for_traced_runs
def on_node_execute_error(node: Any, error: BaseException, run_kwargs: dict[str, Any]) -> None:
    entry = _attempt_entry(run_kwargs)
    if entry is not None:
        entry.attempt_open = False
        _add_execution_event(entry, run_kwargs, error=error)


@_for_traced_runs
def on_node_execute_run(node: Any, run_kwargs: dict[str, Any]) -> None:
    entry = _attempt_entry(run_kwargs)
    if entry is None:
        return
    if _compacts_prompt(entry, run_kwargs.get("prompt_messages")):
        _record_compacted_prompt(entry, node, run_kwargs["prompt_messages"])
        # The prompt is dealt with; whatever else the report holds is mapped as usual.
        run_kwargs = {key: value for key, value in run_kwargs.items() if key != "prompt_messages"}
    _set(
        entry,
        semconv.execute_run_attributes,
        entry.span_type,
        node,
        run_kwargs=run_kwargs,
        settings=entry.settings,
        state=entry.state,
    )
    for name, attributes in semconv.hook_events(run_kwargs, settings=entry.settings):
        entry.span.add_event(name, attributes)


# Workflow and flow runs


@_never_raises
def on_workflow_start(workflow: Any, input_data: Any, config: Any, run_kwargs: dict[str, Any]) -> None:
    run_id = run_kwargs.get("run_id")
    parent = _parent_of(run_kwargs, run_id) if ENTRIES else None
    tracing = _tracing_for(config, parent)
    if tracing is None:
        return
    key = str(run_id)
    spec = semconv.workflow_span_spec(
        workflow, getattr(workflow, "flow", None), run_id=key, run_kind=RunKind.WORKFLOW, run_kwargs=run_kwargs
    )
    entry = _open(key, spec, RunKind.WORKFLOW, tracing, parent)
    if entry.recording:
        _record_input(entry, workflow, input_data, run_kwargs)


@_for_traced_runs
def on_workflow_end(workflow: Any, output_data: Any, run_kwargs: dict[str, Any]) -> None:
    _run_succeeded(workflow, output_data, run_kwargs)


@_for_traced_runs
def on_workflow_error(workflow: Any, error: BaseException, run_kwargs: dict[str, Any]) -> None:
    _run_failed(getattr(workflow, "flow", None), error, run_kwargs)


@_for_traced_runs
def on_workflow_canceled(workflow: Any, run_kwargs: dict[str, Any]) -> None:
    _run_canceled(run_kwargs)


@_never_raises
def on_flow_start(flow: Any, input_data: Any, config: Any, run_kwargs: dict[str, Any]) -> None:
    run_id = run_kwargs.get("run_id")
    parent = _parent_of(run_kwargs, run_id) if ENTRIES else None
    if parent is not None and parent.owner and parent.run_kind == RunKind.WORKFLOW:
        # The workflow span stands for the flow run too, so the flow gets no span of its own.
        entry = _alias(parent, str(run_id))
    else:
        tracing = _tracing_for(config, parent)
        if tracing is None:
            return
        key = str(run_id)
        spec = semconv.workflow_span_spec(None, flow, run_id=key, run_kind=RunKind.FLOW, run_kwargs=run_kwargs)
        entry = _open(key, spec, RunKind.FLOW, tracing, parent)
        if entry.recording:
            _record_input(entry, flow, input_data, run_kwargs)
    _link_resumed_run(entry, flow, run_kwargs)
    _store_traceparent(entry, flow)


@_for_traced_runs
def on_flow_end(flow: Any, output_data: Any, run_kwargs: dict[str, Any]) -> None:
    _run_succeeded(flow, output_data, run_kwargs)


@_for_traced_runs
def on_flow_error(flow: Any, error: BaseException, run_kwargs: dict[str, Any]) -> None:
    _run_failed(flow, error, run_kwargs)


@_for_traced_runs
def on_flow_canceled(flow: Any, run_kwargs: dict[str, Any]) -> None:
    _run_canceled(run_kwargs)


@_for_traced_runs
def ensure_run_ended(run_id: Any, error: BaseException | None = None) -> None:
    """End the span of a workflow or flow run that is leaving without having reported an outcome.

    Does nothing for a run that reported one. Without ``error`` the outcome is read from the exception being
    handled, so call it from the ``except`` or ``finally`` block the run leaves through.
    """
    entry = registry.pop(run_id)
    if entry is None or not entry.owner:
        return
    outcome, error = _unreported_outcome(error if error is not None else sys.exc_info()[1])
    _finish(entry, outcome, {}, error)


# Span start


def _parent_of(run_kwargs: dict[str, Any], run_id: Any) -> RunEntry | None:
    parent_run_id = semconv.parent_run_id_of(run_kwargs, run_id)
    return None if parent_run_id is None else ENTRIES.get(str(parent_run_id))


def _tracing_for(config: Any, parent: RunEntry | None) -> tuple[Tracer, CaptureSettings, CaptureSettings] | None:
    """Tracer, settings and root settings of a run: its parent's, so that a run tree has one provider and one set
    of settings, or the ones its config resolves to when it is the root. None when the tree is not traced."""
    if parent is not None:
        return parent.tracer, parent.settings, parent.root_settings
    tracer = settings.resolve_tracer(config)
    if tracer is None:
        return None
    capture = settings.resolve_settings(config)
    return tracer, capture, capture


def _open(
    run_id: str,
    spec: SpanSpec,
    run_kind: RunKind,
    tracing: tuple[Tracer, CaptureSettings, CaptureSettings],
    parent: RunEntry | None,
) -> RunEntry:
    """Start a span and register it. A node span is also made current.

    Workflow and flow spans are not: their run methods have no single exit to restore the context at, and a
    context left attached would make later, unrelated work on a pooled thread a child of a dead span.
    """
    tracer, capture, root_settings = tracing
    # Without a parent run in this process, the span joins the trace that is current: the caller's own span, or
    # the context a worker thread or process was given.
    parent_context = parent.context if parent is not None else None
    span = tracer.start_span(spec.name, context=parent_context, kind=spec.kind, attributes=spec.attributes)
    is_node = run_kind == RunKind.NODE
    try:
        entry = RunEntry(
            run_id=run_id,
            span=span,
            # A node span is added to the context of its own thread or task, which is the one it replaces
            # there. The others are never attached, so theirs only has to lead children to the right parent.
            context=trace.set_span_in_context(span, None if is_node else parent_context),
            tracer=tracer,
            settings=capture,
            root_settings=root_settings,
            span_type=spec.span_type,
            run_kind=run_kind,
            recording=span.is_recording(),
            parent=parent,
        )
        registry.put(entry)
        if is_node:
            entry.token = context_api.attach(entry.context)
    except BaseException:
        registry.pop(run_id)
        span.end()
        raise
    if entry.recording:
        _set(entry, semconv.run_attributes, capture)
    return entry


def _alias(owner: RunEntry, run_id: str) -> RunEntry:
    """Register a flow run under the span of the workflow that runs it, for the flow's nodes to find."""
    alias = RunEntry(
        run_id=run_id,
        span=owner.span,
        context=owner.context,
        tracer=owner.tracer,
        settings=owner.settings,
        root_settings=owner.root_settings,
        span_type=owner.span_type,
        run_kind=RunKind.FLOW,
        recording=owner.recording,
        owner=False,
    )
    # Listed on the owner first, so that the alias goes when the owner does, whichever hook is reached.
    owner.aliases.append(run_id)
    registry.put(alias)
    if owner.recording:
        owner.span.set_attribute(FLOW_RUN_ID_ATTRIBUTE, run_id)
    return alias


def _check_interop() -> None:
    try:
        interop.warn_if_litellm_otel_callback()
    except Exception as e:
        _log_failure("warn_if_litellm_otel_callback", e)


# Span data


def _recording_entry(run_kwargs: dict[str, Any]) -> RunEntry | None:
    """Entry of a run whose span still takes data: registered, recording and not ended."""
    entry = ENTRIES.get(str(run_kwargs.get("run_id")))
    if entry is None or not entry.recording or entry.ended:
        return None
    return entry


def _attempt_entry(run_kwargs: dict[str, Any]) -> RunEntry | None:
    """Entry of a run whose span takes a report of the execution attempt that ``run_kwargs`` names.

    Only the attempt in progress reports. A report that names no attempt, or comes before any attempt named
    itself, is taken as it is: the attempt it belongs to is unknown.
    """
    entry = _recording_entry(run_kwargs)
    if entry is None:
        return None
    execution_id = run_kwargs.get("execution_run_id")
    if execution_id is None or entry.attempt_id is None:
        return entry
    return entry if entry.attempt_open and str(execution_id) == entry.attempt_id else None


def _set(entry: RunEntry, mapping: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> None:
    """Set the attributes ``mapping`` returns. A mapping that fails costs the span its own attributes only."""
    try:
        attributes = mapping(*args, **kwargs)
        if attributes:
            entry.span.set_attributes(attributes)
    except Exception as e:
        _log_failure(getattr(mapping, "__name__", "mapping"), e)


def _compacts_prompt(entry: RunEntry, messages: Any) -> bool:
    """Whether the prompt a run reports is recorded under prompt compaction.

    As ``TracingCallbackHandler`` compacts: never for a run without a parent run. Here the parent run also needs
    a span in this process, which is what keeps the prompt of the previous run, and the span has to be exported:
    a later span must not point at one the backend never receives.
    """
    capture = entry.settings
    return (
        capture.prompt_compaction_enabled
        and capture.content_enabled
        and entry.parent is not None
        and entry.span_type == SpanType.CHAT
        and isinstance(messages, list)
        and bool(messages)
        and entry.span.get_span_context().trace_flags.sampled
    )


def _record_compacted_prompt(entry: RunEntry, node: Any, messages: list[Any]) -> None:
    """Record the prompt of an LLM run without the start it shares with the prompt of the previous run of the
    same LLM node under the same parent run, and keep it for the next run to be compared with.

    A run is compared with a prompt that was kept before it reported, and its own is kept only afterwards, so
    two runs never point at each other, also when runs of one parent report at once.
    """
    if entry.prompt_reported:
        # A retried run reports its prompt again. What it first recorded stays: a later run may already point at
        # it, and comparing it again, now with that later run, would make the two point at each other.
        return
    entry.prompt_reported = True
    parent, node_id = entry.parent, str(getattr(node, "id", None))
    record = PromptRecord(
        span_id=trace.format_span_id(entry.span.get_span_context().span_id),
        # Without None, as the messages are recorded, so that a number of messages counts recorded ones.
        messages=[message for message in messages if message is not None],
    )
    try:
        previous = registry.last_prompt(parent, node_id)
        attributes = semconv.prompt_attributes(record.messages, settings=entry.settings, previous=previous)
        entry.span.set_attributes(attributes)
    except Exception as e:
        _log_failure("prompt_attributes", e)
        return
    # Kept only now that the span holds the prompt, and not for a span that ended meanwhile and may have missed
    # it: a later run points at this span for the messages it leaves out.
    if not entry.ended:
        registry.keep_prompt(parent, node_id, record)


def _record_input(entry: RunEntry, entity: Any, input_data: Any, run_kwargs: dict[str, Any]) -> None:
    _set(
        entry,
        semconv.input_attributes,
        entry.span_type,
        entity,
        input_data,
        run_kwargs=run_kwargs,
        settings=entry.settings,
        state=entry.state,
    )
    entry.input_recorded = True


def _inherit_conversation(entry: RunEntry, node: Any, input_data: Any) -> None:
    """Take the session and user a run declares in its input, for its span and the runs below it.

    Ids set on ``TracingConfig`` win. Otherwise a run's own ids replace inherited ones, so that every span
    carries the ids of the nearest enclosing run that declares them.
    """
    try:
        session_id, user_id = semconv.conversation_of(entry.span_type, node, input_data)
    except Exception as e:
        _log_failure("conversation_of", e)
        return
    update = {}
    if session_id and not entry.root_settings.session_id and session_id != entry.settings.session_id:
        update["session_id"] = session_id
    if user_id and not entry.root_settings.user_id and user_id != entry.settings.user_id:
        update["user_id"] = user_id
    if update:
        entry.settings = entry.settings.model_copy(update=update)
        _set(entry, semconv.run_attributes, entry.settings)


def _add_execution_event(entry: RunEntry, run_kwargs: dict[str, Any], error: BaseException | None) -> None:
    execution_id = str(run_kwargs.get("execution_run_id"))
    started = entry.attempt_started.pop(execution_id, None)
    attributes = semconv.execution_event(
        execution_id=execution_id,
        attempt=max(entry.attempts, 1),
        succeeded=error is None,
        start_time_unix_nano=started if started is not None else time.time_ns(),
        error=error,
        settings=entry.settings,
    )
    entry.span.add_event(semconv.EXECUTION_EVENT_NAME, attributes)


def _link_resumed_run(entry: RunEntry, flow: Any, run_kwargs: dict[str, Any]) -> None:
    if not entry.recording or not run_kwargs.get("resumed_from_checkpoint"):
        return
    _set(entry, semconv.resume_attributes, run_kwargs)
    try:
        # Read before _store_traceparent replaces it: the loaded checkpoint still holds the paused run's value.
        metadata = getattr(getattr(flow, "_checkpoint", None), "metadata", None) or {}
        paused_span = propagation.span_context_from_traceparent(metadata.get(propagation.TRACEPARENT))
        if paused_span is not None:
            entry.span.add_link(paused_span, semconv.RESUME_LINK_ATTRIBUTES)
    except Exception as e:
        _log_failure("_link_resumed_run", e)


def _store_traceparent(entry: RunEntry, flow: Any) -> None:
    """Put the span of the run in the flow's checkpoint, where a pause saves it for the resumed run to link to."""
    try:
        checkpoint = getattr(flow, "_checkpoint", None)
        if checkpoint is None:
            return
        traceparent = propagation.traceparent_for_run(entry.run_id)
        if traceparent:
            # A new dict: checkpoint snapshots are shallow copies of one another and share this one.
            checkpoint.metadata = {**checkpoint.metadata, propagation.TRACEPARENT: traceparent}
    except Exception as e:
        _log_failure("_store_traceparent", e)


# Span end


def _failure_outcome(error: BaseException | None) -> RunOutcome:
    return RunOutcome.PAUSED if isinstance(error, RunPausedException) else RunOutcome.FAILED


def _unreported_outcome(error: BaseException | None) -> tuple[RunOutcome, BaseException | None]:
    """Outcome and error of a run that ended without reporting one, from the exception that ended it."""
    if isinstance(error, _CANCELLATIONS):
        return RunOutcome.CANCELED, None
    if error is None:
        error = RuntimeError("The run ended without reporting an outcome.")
    return _failure_outcome(error), error


def _finish(
    entry: RunEntry,
    outcome: RunOutcome,
    run_kwargs: dict[str, Any],
    error: BaseException | None = None,
    record: Callable[[], None] | None = None,
) -> None:
    """End the span of ``entry`` with ``outcome``, once.

    ``record`` sets the attributes known only at the end. It runs for a recording span. Whatever it does, the
    span gets its outcome and ends.
    """
    if not registry.claim_end(entry):
        return
    try:
        if entry.recording:
            if record is not None:
                try:
                    record()
                except Exception as e:
                    _log_failure("_finish", e)
            _set(entry, semconv.outcome_attributes, outcome, run_kwargs=run_kwargs)
            _record_error(entry, error, outcome)
    except Exception as e:
        _log_failure("_finish", e)
    finally:
        entry.span.end()


def _record_error(entry: RunEntry, error: BaseException | None, outcome: RunOutcome) -> None:
    span = entry.span
    try:
        spec = semconv.error_spec(error, outcome, settings=entry.settings)
    except Exception as e:
        _log_failure("error_spec", e)
        # The status still has to say that the run did not succeed.
        if outcome in _ERROR_OUTCOMES:
            span.set_status(Status(StatusCode.ERROR))
        return
    if spec is None:
        return
    span.set_attribute(ERROR_TYPE_ATTRIBUTE, spec.error_type)
    span.set_status(Status(StatusCode.ERROR, spec.description))
    if spec.event_attributes is not None:
        span.add_event(semconv.EXCEPTION_EVENT_NAME, spec.event_attributes)


def _run_succeeded(entity: Any, output_data: Any, run_kwargs: dict[str, Any]) -> None:
    entry = _pop_owner(run_kwargs)
    if entry is None:
        return
    _finish(
        entry,
        RunOutcome.SUCCEEDED,
        run_kwargs,
        record=lambda: _set(
            entry,
            semconv.output_attributes,
            entry.span_type,
            entity,
            output_data,
            run_kwargs=run_kwargs,
            settings=entry.settings,
            state=entry.state,
        ),
    )


def _run_failed(flow: Any, error: BaseException, run_kwargs: dict[str, Any]) -> None:
    entry = _pop_owner(run_kwargs)
    if entry is None:
        return
    outcome = _failure_outcome(error)

    def record() -> None:
        if outcome == RunOutcome.PAUSED:
            checkpoint_id = getattr(getattr(flow, "_checkpoint", None), "id", None)
            if checkpoint_id:
                entry.span.set_attribute(CHECKPOINT_ID_ATTRIBUTE, str(checkpoint_id))
        else:
            _set(entry, semconv.failure_attributes, run_kwargs, settings=entry.settings)

    _finish(entry, outcome, run_kwargs, error, record)


def _run_canceled(run_kwargs: dict[str, Any]) -> None:
    entry = _pop_owner(run_kwargs)
    if entry is not None:
        _finish(entry, RunOutcome.CANCELED, run_kwargs)


def _pop_owner(run_kwargs: dict[str, Any]) -> RunEntry | None:
    """Remove the entry of a workflow or flow run that reports its outcome. None for an alias: the workflow
    whose span it shares reports the outcome."""
    entry = registry.pop(run_kwargs.get("run_id"))
    return entry if entry is not None and entry.owner else None
