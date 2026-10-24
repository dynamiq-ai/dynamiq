from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .answer_correctness import AnswerCorrectnessEvaluator
    from .bleu_score import BleuScoreEvaluator
    from .context_precision import ContextPrecisionEvaluator
    from .context_recall import ContextRecallEvaluator
    from .factual_correctness import FactualCorrectnessEvaluator
    from .faithfulness import FaithfulnessEvaluator
    from .rouge_score import RougeScoreEvaluator
    from .string_metrics import DistanceMeasure, ExactMatchEvaluator, StringPresenceEvaluator, StringSimilarityEvaluator

_LAZY_EXPORTS = {
    "AnswerCorrectnessEvaluator": ".answer_correctness",
    "BleuScoreEvaluator": ".bleu_score",
    "ContextPrecisionEvaluator": ".context_precision",
    "ContextRecallEvaluator": ".context_recall",
    "FactualCorrectnessEvaluator": ".factual_correctness",
    "FaithfulnessEvaluator": ".faithfulness",
    "RougeScoreEvaluator": ".rouge_score",
    "DistanceMeasure": ".string_metrics",
    "ExactMatchEvaluator": ".string_metrics",
    "StringPresenceEvaluator": ".string_metrics",
    "StringSimilarityEvaluator": ".string_metrics",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "AnswerCorrectnessEvaluator",
    "BleuScoreEvaluator",
    "ContextPrecisionEvaluator",
    "ContextRecallEvaluator",
    "FactualCorrectnessEvaluator",
    "FaithfulnessEvaluator",
    "RougeScoreEvaluator",
    "DistanceMeasure",
    "ExactMatchEvaluator",
    "StringPresenceEvaluator",
    "StringSimilarityEvaluator",
]
