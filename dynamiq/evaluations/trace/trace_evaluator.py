"""Generic LLM-judge over a captured workflow trace.

The :class:`TraceEvaluator` is the Raindrop-style building block for
post-hoc trace evaluation: the user describes in plain language what
behavior to detect, and the evaluator renders the captured ``Run`` tree,
asks an LLM judge to score it, and returns a structured result with
optional findings that point at offending steps.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.callbacks.tracing import Run
from dynamiq.evaluations import BaseEvaluator
from dynamiq.evaluations.llm_evaluator import LLMEvaluator
from dynamiq.evaluations.trace.context import AgentContext
from dynamiq.evaluations.trace.rendering import (
    DEFAULT_FIELD_CHAR_LIMIT,
    DEFAULT_MAX_TRACE_CHARS,
    DEFAULT_SHORT_ID_LENGTH,
    RenderedTrace,
    render_trace,
)
from dynamiq.nodes.llms import BaseLLM
from dynamiq.utils.logger import logger


class Severity(str, Enum):
    """Severity ranking for a finding."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Finding(BaseModel):
    """A single concrete issue surfaced by the judge."""

    run_id: str = Field(description="UUID of the offending Run.")
    run_name: str = Field(description="Human-readable node name.")
    severity: Severity = Field(description="Severity ranking.")
    message: str = Field(description="Short description of the issue.")
    evidence: str = Field(default="", description="Verbatim snippet quoted from the trace.")


class TraceRunResult(BaseModel):
    """Per-trace evaluation result."""

    score: float = Field(description="Health score in [0.0, 1.0]; 1.0 means no issues.")
    reasoning: str = Field(description="Narrative explanation from the judge.")
    findings: list[Finding] = Field(default_factory=list)
    suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Actionable suggestions for what to change in the agent prompt or tool "
            "wiring. Populated only when the evaluator is run in suggestion mode."
        ),
    )
    matched: bool = Field(
        default=False,
        description=(
            "True when this trace exhibits the behavior at or above the configured "
            "severity threshold (or when the score crosses the score threshold)."
        ),
    )


class TraceRunOutput(BaseModel):
    """Batch evaluation output, one entry per input trace."""

    results: list[TraceRunResult]


_DEFAULT_EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {
            "behavior": "Detect tool calls that errored.",
            "trace": (
                "[run_id=abc12345] node:HttpTool status=failed duration=120ms\n"
                '  input: {"url": "https://api.example.com/data"}\n'
                "  error: HTTP 500 Internal Server Error"
            ),
        },
        "outputs": {
            "score": 0.0,
            "reasoning": "The HttpTool node failed with a 500 error.",
            "findings": [
                {
                    "run_id": "abc12345",
                    "severity": "high",
                    "message": "HttpTool returned 500 Internal Server Error.",
                    "evidence": "error: HTTP 500 Internal Server Error",
                }
            ],
        },
    },
    {
        "inputs": {
            "behavior": "Detect tool calls that errored.",
            "trace": (
                "[run_id=def67890] node:Search status=succeeded duration=80ms\n"
                '  input: {"query": "weather"}\n'
                '  output: {"results": ["sunny"]}'
            ),
        },
        "outputs": {
            "score": 1.0,
            "reasoning": "The Search node completed successfully and produced output.",
            "findings": [],
        },
    },
]


_INSTRUCTIONS = """You are a trace auditor for an AI workflow. You will be given:
- BEHAVIOR: a plain-language description of the behavior to detect.
- TRACE: a textual rendering of a workflow execution. Each step is tagged
  with [run_id=<id>] so you can cite specific steps.
- AGENT_CONTEXT: a structured block describing the agent that produced
  the trace. It contains:
    - AGENT_PROMPT: the system prompt / role given to the agent.
    - TOOLS: the list of tools available to the agent (name + description).
    - CONVERSATION_HISTORY: the chronological messages between the user
      and the agent.

Your job:
1. Read the trace and decide whether the BEHAVIOR is exhibited.
2. Assign a score in [0.0, 1.0] where 1.0 means the behavior is NOT
   present (healthy) and 0.0 means it is severely present.
3. List concrete findings: for each offending step include the run_id
   (the short id from the [run_id=...] tag, exactly as written),
   a severity (info|low|medium|high|critical), a short message, and a
   verbatim evidence snippet from the trace.
4. Provide a brief overall reasoning. When relevant, cite specific tools
   by name or specific user turns from CONVERSATION_HISTORY.
5. Produce a list of concrete, actionable SUGGESTIONS. Each suggestion
   is a single short string proposing a specific fix to the AGENT_PROMPT
   or the TOOLS wiring (add/remove/rewrite a tool description) that would
   prevent the observed behavior. Do not suggest changes to user messages
   in CONVERSATION_HISTORY. If no issues are found, return an empty
   suggestions array.

If no issues are found, return an empty findings array, an empty
suggestions array, and score 1.0.
"""


class TraceEvaluator(BaseEvaluator):
    """Generic LLM judge over a captured workflow trace.

    Construct one instance per behavior you want to detect. The judge is
    backed by :class:`LLMEvaluator` and rendered traces are produced by
    :func:`dynamiq.evaluations.trace.rendering.render_trace`.

    Example:
        ```python
        evaluator = TraceEvaluator(
            llm=OpenAI(model="gpt-4o-mini"),
            instructions="Detect tool errors that were not surfaced to the user.",
        )
        result = evaluator.run_single(trace=list(tracing.runs.values()))
        for f in result.findings:
            print(f.severity, f.run_name, f.message)
        ```
    """

    name: str = "TraceEvaluator"
    instructions: str = Field(description="Plain-language description of the behavior to detect.")
    llm: BaseLLM
    severity_threshold: Severity = Severity.LOW
    score_threshold: float = Field(
        default=1.0,
        description="A trace with score strictly less than this is marked as `matched`.",
    )
    examples: list[dict[str, Any]] | None = None
    max_trace_chars: int = DEFAULT_MAX_TRACE_CHARS
    field_char_limit: int = DEFAULT_FIELD_CHAR_LIMIT
    short_id_length: int = DEFAULT_SHORT_ID_LENGTH
    context: AgentContext = Field(
        default_factory=AgentContext,
        description="Structured agent context (prompt, tools, conversation history).",
    )

    _judge: LLMEvaluator = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._initialize_judge()

    def _initialize_judge(self) -> None:
        self._judge = LLMEvaluator(
            instructions=_INSTRUCTIONS.strip(),
            inputs=[
                {"name": "behavior", "type": str},
                {"name": "trace", "type": str},
                {"name": "agent_context", "type": str},
            ],
            outputs=[
                {"name": "score", "type": float},
                {"name": "reasoning", "type": str},
                {"name": "findings", "type": list},
                {"name": "suggestions", "type": list},
            ],
            examples=self.examples if self.examples is not None else _DEFAULT_EXAMPLES,
            llm=self.llm,
        )

    def run_single(self, trace: Run | Iterable[Run] | None) -> TraceRunResult:
        """Evaluate a single captured trace.

        Args:
            trace: A single root :class:`Run`, an iterable of runs (e.g.
                ``list(tracing.runs.values())``), or ``None``. The renderer
                accepts both shapes and walks parent/child relationships.

        Returns:
            A :class:`TraceRunResult` containing score, reasoning and any
            structured findings the judge produced.
        """
        rendered = render_trace(
            trace,
            max_chars=self.max_trace_chars,
            field_char_limit=self.field_char_limit,
            short_id_length=self.short_id_length,
        )
        return self._run_on_rendered(rendered)

    def _run_on_rendered(self, rendered: RenderedTrace) -> TraceRunResult:
        """Evaluate a pre-rendered trace. Lets a suite render once and reuse."""
        if not rendered.text:
            return TraceRunResult(
                score=1.0,
                reasoning="Trace is empty; no behavior to evaluate.",
                findings=[],
                matched=False,
            )
        raw = self._judge.run(
            behavior=[self.instructions],
            trace=[rendered.text],
            agent_context=[self.context.render()],
        )
        return self._parse_judge_output(raw, rendered)

    def run(self, traces: Iterable[Run | Iterable[Run] | None]) -> TraceRunOutput:
        """Evaluate a batch of captured traces in order."""
        results: list[TraceRunResult] = []
        for trace in traces:
            results.append(self.run_single(trace))
        return TraceRunOutput(results=results)

    def _parse_judge_output(self, raw: dict[str, Any], rendered: RenderedTrace) -> TraceRunResult:
        result_list = raw.get("results") or []
        if not result_list or result_list[0] is None:
            logger.warning("TraceEvaluator: judge returned no results; defaulting to neutral score.")
            return TraceRunResult(
                score=1.0, reasoning="Judge returned no result.", findings=[], suggestions=[], matched=False
            )

        first = result_list[0]
        score = self._coerce_score(first.get("score"))
        reasoning = str(first.get("reasoning", "")).strip()
        raw_findings = first.get("findings") or []
        findings = self._build_findings(raw_findings, rendered)
        suggestions = [str(s).strip() for s in (first.get("suggestions") or []) if str(s).strip()]
        matched = self._compute_matched(score, findings)
        return TraceRunResult(
            score=score, reasoning=reasoning, findings=findings, suggestions=suggestions, matched=matched
        )

    @staticmethod
    def _coerce_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            logger.warning("TraceEvaluator: judge returned non-numeric score %r; defaulting to 1.0.", value)
            return 1.0
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score

    @staticmethod
    def _coerce_severity(value: Any) -> Severity:
        if isinstance(value, Severity):
            return value
        if isinstance(value, str):
            try:
                return Severity(value.lower())
            except ValueError:
                pass
        return Severity.MEDIUM

    def _build_findings(
        self,
        raw_findings: list[Any],
        rendered: RenderedTrace,
    ) -> list[Finding]:
        findings: list[Finding] = []
        for raw_finding in raw_findings:
            if not isinstance(raw_finding, dict):
                logger.warning("TraceEvaluator: skipping malformed finding %r.", raw_finding)
                continue
            short_id = str(raw_finding.get("run_id", "")).strip()
            run = rendered.short_id_to_run.get(short_id)
            if run is None:
                logger.warning(
                    "TraceEvaluator: dropping finding referencing unknown run_id %r.",
                    short_id,
                )
                continue
            findings.append(
                Finding(
                    run_id=str(run.id),
                    run_name=run.name or "?",
                    severity=self._coerce_severity(raw_finding.get("severity")),
                    message=str(raw_finding.get("message", "")).strip(),
                    evidence=str(raw_finding.get("evidence", "")).strip(),
                )
            )
        return findings

    def _compute_matched(self, score: float, findings: list[Finding]) -> bool:
        if score < self.score_threshold:
            return True
        threshold_rank = _SEVERITY_RANK[self.severity_threshold]
        for finding in findings:
            if _SEVERITY_RANK[finding.severity] >= threshold_rank:
                return True
        return False
