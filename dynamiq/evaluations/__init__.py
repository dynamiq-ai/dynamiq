from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .base_evaluator import BaseEvaluator
    from .llm_evaluator import LLMEvaluator
    from .python_evaluator import PythonEvaluator

_LAZY_EXPORTS = {
    "BaseEvaluator": ".base_evaluator",
    "LLMEvaluator": ".llm_evaluator",
    "PythonEvaluator": ".python_evaluator",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "BaseEvaluator",
    "LLMEvaluator",
    "PythonEvaluator",
]
