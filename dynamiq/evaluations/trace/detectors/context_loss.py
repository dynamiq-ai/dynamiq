from __future__ import annotations

from typing import Any

from dynamiq.evaluations.trace.trace_evaluator import Severity, TraceEvaluator

_INSTRUCTIONS = (
    "Detect context loss: cases where the agent forgets, ignores, or "
    "contradicts information the user already provided earlier in the "
    "trace, or asks for information that was already supplied. Flag "
    "each step where established context was dropped."
)

_EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {
            "behavior": _INSTRUCTIONS,
            "trace": (
                "[run_id=1111aaaa] node:Agent status=succeeded duration=100ms\n"
                '  input: {"message": "My name is Ada. Book a flight to Paris."}\n'
                '  output: {"reply": "What city should I book the flight to?"}'
            ),
        },
        "outputs": {
            "score": 0.2,
            "reasoning": "Agent asked for the destination already given in the same turn.",
            "findings": [
                {
                    "run_id": "1111aaaa",
                    "severity": "high",
                    "message": "Agent lost the user-provided destination 'Paris'.",
                    "evidence": '"What city should I book the flight to?"',
                }
            ],
        },
    },
]


class ContextLossDetector(TraceEvaluator):
    name: str = "context_loss"
    instructions: str = _INSTRUCTIONS
    examples: list[dict[str, Any]] | None = _EXAMPLES
    severity_threshold: Severity = Severity.MEDIUM
