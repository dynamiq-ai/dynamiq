from __future__ import annotations

from typing import Any

from dynamiq.evaluations.trace.trace_evaluator import Severity, TraceEvaluator

_INSTRUCTIONS = (
    "Detect tool calls that errored or returned a failure status and "
    "were NOT recovered from. A recovery is a subsequent successful "
    "retry of the same tool or a clear fallback path. Flag failed steps "
    "whose error propagated to the final output or was silently ignored."
)

_EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {
            "behavior": _INSTRUCTIONS,
            "trace": (
                "[run_id=eeee1111] node:HttpTool status=failed duration=120ms\n"
                "  error: HTTP 500 Internal Server Error"
            ),
        },
        "outputs": {
            "score": 0.0,
            "reasoning": "HttpTool failed with a 500 and there is no retry or fallback.",
            "findings": [
                {
                    "run_id": "eeee1111",
                    "severity": "high",
                    "message": "HttpTool failed and the error was not recovered.",
                    "evidence": "error: HTTP 500 Internal Server Error",
                }
            ],
        },
    },
]


class UnrecoveredToolErrorDetector(TraceEvaluator):
    name: str = "tool_error"
    instructions: str = _INSTRUCTIONS
    examples: list[dict[str, Any]] | None = _EXAMPLES
    severity_threshold: Severity = Severity.MEDIUM
