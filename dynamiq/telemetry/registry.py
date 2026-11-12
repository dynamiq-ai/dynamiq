"""Spans of the runs in progress in this process, keyed by run id.

Run code reports the start, the data and the outcome of a run from separate call sites that share only the run
id, and a child run names its parent by run id, so both find the span here. The framework generates run ids
(``uuid4``) and never takes them from a caller, which is what makes one registry safe for every run and tenant
of a process.
"""

import os
import threading
from typing import Any

from opentelemetry.context import Context
from opentelemetry.trace import Span, Tracer

from dynamiq.telemetry.types import CaptureSettings, PromptRecord, RunKind, SpanType


class RunEntry:
    """Runtime state of one traced run.

    Attributes:
        run_id: Registry key, the run id as a string.
        span: Span of the run. An alias shares the span of the run it stands for.
        context: OpenTelemetry context that holds ``span``; the parent context of child runs.
        tracer: Tracer of the run tree; child runs reuse it.
        settings: Capture settings child runs inherit. Replaced, never mutated.
        root_settings: Settings as the root of the run tree resolved them, before any run added to them.
        span_type: Span type the mapping functions are called with.
        run_kind: Whether the run is a workflow, a flow or a node.
        recording: Whether the span records. False for a span the sampler dropped.
        owner: False for an alias, which never ends the span.
        parent: Entry of the parent run, or None for a run whose parent run has no span in this process.
        ended: Whether the span has been ended.
        token: Token of the context attached for the run, or None. Only node runs attach.
        state: Scratch space of the mapping functions for this span.
        attempts: Execution attempts started so far.
        attempt_started: Start time in nanoseconds of each attempt, by execution id.
        attempt_id: Execution id of the latest attempt, or None while no attempt has named one.
        attempt_open: Whether that attempt is in progress: started, and neither its success nor its failure
            reported. An attempt that timed out is abandoned, not stopped, and keeps reporting from its thread.
        input_recorded: Whether the input attributes are on the span.
        aliases: Keys of the aliases of this entry, removed with it.
        prompts: Latest prompt among the runs below this one of each LLM node, by node id, for prompt compaction.
            None until such a run records one. It lives and goes with the entry, so that no prompt outlasts the
            run tree it was sent in.
        prompt_reported: Whether the run reported its prompt under prompt compaction.
    """

    __slots__ = (
        "run_id",
        "span",
        "context",
        "tracer",
        "settings",
        "root_settings",
        "span_type",
        "run_kind",
        "recording",
        "owner",
        "parent",
        "ended",
        "token",
        "state",
        "attempts",
        "attempt_started",
        "attempt_id",
        "attempt_open",
        "input_recorded",
        "aliases",
        "prompts",
        "prompt_reported",
    )

    def __init__(
        self,
        *,
        run_id: str,
        span: Span,
        context: Context,
        tracer: Tracer,
        settings: CaptureSettings,
        root_settings: CaptureSettings,
        span_type: SpanType,
        run_kind: RunKind,
        recording: bool,
        owner: bool = True,
        parent: "RunEntry | None" = None,
    ):
        self.run_id = run_id
        self.span = span
        self.context = context
        self.tracer = tracer
        self.settings = settings
        self.root_settings = root_settings
        self.span_type = span_type
        self.run_kind = run_kind
        self.recording = recording
        self.owner = owner
        self.parent = parent
        self.ended = False
        self.token: Any = None
        self.state: dict[str, Any] = {}
        self.attempts = 0
        self.attempt_started: dict[str, int] = {}
        self.attempt_id: str | None = None
        self.attempt_open = False
        self.input_recorded = False
        self.aliases: list[str] = []
        self.prompts: dict[str, PromptRecord] | None = None
        self.prompt_reported = False


# Written only by put() and pop(), under the lock. Read without it: a lookup is one dict.get, and the emitter
# tests the dict's truthiness on every call, which is all a run costs while nothing is traced.
ENTRIES: dict[str, RunEntry] = {}
_lock = threading.Lock()


def get(run_id: Any) -> RunEntry | None:
    """Return the entry of a run, or None when the run is not traced in this process."""
    if not ENTRIES or run_id is None:
        return None
    return ENTRIES.get(str(run_id))


def put(entry: RunEntry) -> None:
    """Register an entry under its run id."""
    with _lock:
        ENTRIES[entry.run_id] = entry


def pop(run_id: Any) -> RunEntry | None:
    """Remove and return the entry of a run, with the aliases registered for it."""
    if not ENTRIES or run_id is None:
        return None
    with _lock:
        entry = ENTRIES.pop(str(run_id), None)
        if entry is not None:
            for alias in entry.aliases:
                ENTRIES.pop(alias, None)
        return entry


def claim_end(entry: RunEntry) -> bool:
    """Mark the span of an entry as ended. False when it already was, so that it is ended once."""
    with _lock:
        if entry.ended:
            return False
        entry.ended = True
        return True


def last_prompt(parent: RunEntry, node_id: str) -> PromptRecord | None:
    """Return the latest prompt kept for an LLM node among the runs below ``parent``, or None."""
    # Read without the lock, as ENTRIES is: one dict.get.
    prompts = parent.prompts
    return prompts.get(node_id) if prompts else None


def keep_prompt(parent: RunEntry, node_id: str, record: PromptRecord) -> None:
    """Keep ``record`` as the latest prompt of an LLM node among the runs below ``parent``."""
    with _lock:
        if parent.prompts is None:
            parent.prompts = {}
        parent.prompts[node_id] = record


def size() -> int:
    """Number of registered entries."""
    return len(ENTRIES)


def _reset_after_fork() -> None:
    # The entries belong to runs of the parent process, which ends their spans, and the lock may have been held
    # by a thread that does not exist in the child.
    global _lock
    _lock = threading.Lock()
    ENTRIES.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
