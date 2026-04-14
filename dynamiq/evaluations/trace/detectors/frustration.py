from __future__ import annotations

from typing import Any

from dynamiq.evaluations.trace.trace_evaluator import Severity, TraceEvaluator

_INSTRUCTIONS = (
    "Detect user frustration or negative sentiment in the trace. Flag "
    "steps where the user expresses anger, repeats a request the agent "
    "failed to satisfy, escalates tone, or explicitly signals the agent "
    "is not helping."
)

_EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {
            "behavior": _INSTRUCTIONS,
            "trace": (
                "[run_id=ffff1111] node:Agent status=succeeded duration=80ms\n"
                '  input: {"message": "This is the third time I ask. Just cancel the order!"}'
            ),
        },
        "outputs": {
            "score": 0.3,
            "reasoning": "User signals frustration via repetition and imperative tone.",
            "findings": [
                {
                    "run_id": "ffff1111",
                    "severity": "medium",
                    "message": "User expresses repeated frustration about unresolved cancellation.",
                    "evidence": '"This is the third time I ask. Just cancel the order!"',
                }
            ],
        },
    },
]


class FrustrationDetector(TraceEvaluator):
    name: str = "frustration"
    instructions: str = _INSTRUCTIONS
    examples: list[dict[str, Any]] | None = _EXAMPLES
    severity_threshold: Severity = Severity.LOW
