from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .base_evaluator import BaseEvaluator
    from .llm_evaluator import LLMEvaluator
    from .memory import MemoryEvaluationReport, MemorySuite, evaluate_memory, replay_memory
    from .python_evaluator import PythonEvaluator

_LAZY_EXPORTS = {
    "BaseEvaluator": ".base_evaluator",
    "LLMEvaluator": ".llm_evaluator",
    "PythonEvaluator": ".python_evaluator",
    "MemoryEvaluationReport": ".memory",
    "MemorySuite": ".memory",
    "evaluate_memory": ".memory",
    "replay_memory": ".memory",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "BaseEvaluator",
    "LLMEvaluator",
    "PythonEvaluator",
    "MemoryEvaluationReport",
    "MemorySuite",
    "evaluate_memory",
    "replay_memory",
]
