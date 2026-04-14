"""Composite evaluator that runs multiple :class:`TraceEvaluator` members over the same trace.

The suite renders the trace once, dispatches the rendered view to every
member (bypassing re-rendering), and aggregates the per-detector results
into a single report.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field

from dynamiq.callbacks.tracing import Run
from dynamiq.evaluations import BaseEvaluator
from dynamiq.evaluations.trace.context import AgentContext
from dynamiq.evaluations.trace.rendering import DEFAULT_SHORT_ID_LENGTH, render_trace
from dynamiq.evaluations.trace.trace_evaluator import (
    DEFAULT_FIELD_CHAR_LIMIT,
    DEFAULT_MAX_TRACE_CHARS,
    Finding,
    TraceEvaluator,
    TraceRunResult,
)


class TraceSuiteResult(BaseModel):
    """Aggregated per-detector result for a single trace."""

    results: dict[str, TraceRunResult] = Field(default_factory=dict)

    @property
    def matched(self) -> list[str]:
        return [name for name, r in self.results.items() if r.matched]

    @property
    def findings(self) -> list[Finding]:
        merged: list[Finding] = []
        for r in self.results.values():
            merged.extend(r.findings)
        return merged

    @property
    def min_score(self) -> float:
        if not self.results:
            return 1.0
        return min(r.score for r in self.results.values())


class TraceSuiteOutput(BaseModel):
    """Batch output, one entry per input trace."""

    results: list[TraceSuiteResult]


class TraceEvaluatorSuite(BaseEvaluator):
    """Run several :class:`TraceEvaluator` detectors over the same trace."""

    name: str = "TraceEvaluatorSuite"
    evaluators: dict[str, TraceEvaluator]
    max_trace_chars: int = DEFAULT_MAX_TRACE_CHARS
    field_char_limit: int = DEFAULT_FIELD_CHAR_LIMIT
    short_id_length: int = DEFAULT_SHORT_ID_LENGTH

    def run_single(self, trace: Run | Iterable[Run] | None) -> TraceSuiteResult:
        rendered = render_trace(
            trace,
            max_chars=self.max_trace_chars,
            field_char_limit=self.field_char_limit,
            short_id_length=self.short_id_length,
        )
        results = {name: evaluator._run_on_rendered(rendered) for name, evaluator in self.evaluators.items()}
        return TraceSuiteResult(results=results)

    def run(self, traces: Iterable[Run | Iterable[Run] | None]) -> TraceSuiteOutput:
        return TraceSuiteOutput(results=[self.run_single(t) for t in traces])

    def set_context(self, context: AgentContext) -> None:
        """Share a single agent context across every member evaluator."""
        for evaluator in self.evaluators.values():
            evaluator.context = context
