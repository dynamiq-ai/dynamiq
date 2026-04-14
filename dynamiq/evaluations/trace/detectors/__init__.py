"""Prebuilt :class:`TraceEvaluator` subclasses for common behaviors.

Each detector hard-codes a tuned instructions prompt, few-shot examples,
and a severity threshold. Use standalone or compose into a
:class:`TraceEvaluatorSuite`.
"""

from .context_loss import ContextLossDetector
from .frustration import FrustrationDetector
from .hallucination import HallucinationDetector
from .tool_error import UnrecoveredToolErrorDetector


def default_suite(llm):
    """Return a :class:`TraceEvaluatorSuite` with the four built-in detectors."""
    from dynamiq.evaluations.trace.suite import TraceEvaluatorSuite

    return TraceEvaluatorSuite(
        evaluators={
            "hallucination": HallucinationDetector(llm=llm),
            "context_loss": ContextLossDetector(llm=llm),
            "frustration": FrustrationDetector(llm=llm),
            "tool_error": UnrecoveredToolErrorDetector(llm=llm),
        }
    )


__all__ = [
    "ContextLossDetector",
    "FrustrationDetector",
    "HallucinationDetector",
    "UnrecoveredToolErrorDetector",
    "default_suite",
]
