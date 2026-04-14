from __future__ import annotations

from typing import Any

from dynamiq.evaluations.trace.trace_evaluator import Severity, TraceEvaluator

_INSTRUCTIONS = (
    "Detect hallucinations: claims in the agent's final output or "
    "intermediate reasoning that are NOT supported by the tool results, "
    "retrieved documents, or user-provided context visible in the trace. "
    "Flag fabricated facts, invented citations, and confident answers "
    "produced without a grounding tool call."
)

_EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {
            "behavior": _INSTRUCTIONS,
            "trace": (
                "[run_id=aaaa1111] node:Agent status=succeeded duration=200ms\n"
                '  output: {"answer": "The Eiffel Tower is 500 meters tall."}'
            ),
        },
        "outputs": {
            "score": 0.1,
            "reasoning": "Agent produced a numeric claim with no tool call to ground it.",
            "findings": [
                {
                    "run_id": "aaaa1111",
                    "severity": "high",
                    "message": "Unsupported numeric claim about Eiffel Tower height.",
                    "evidence": '"The Eiffel Tower is 500 meters tall."',
                }
            ],
        },
    },
    {
        "inputs": {
            "behavior": _INSTRUCTIONS,
            "trace": (
                "[run_id=bbbb2222] node:Search status=succeeded duration=90ms\n"
                '  output: {"results": ["Eiffel Tower is 330m tall"]}\n'
                "[run_id=cccc3333] node:Agent status=succeeded duration=50ms\n"
                '  output: {"answer": "The Eiffel Tower is 330 meters tall."}'
            ),
        },
        "outputs": {
            "score": 1.0,
            "reasoning": "Agent's answer is grounded in the Search tool result.",
            "findings": [],
        },
    },
]


class HallucinationDetector(TraceEvaluator):
    name: str = "hallucination"
    instructions: str = _INSTRUCTIONS
    examples: list[dict[str, Any]] | None = _EXAMPLES
    severity_threshold: Severity = Severity.MEDIUM
