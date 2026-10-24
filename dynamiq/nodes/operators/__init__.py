from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .decision_table import DecisionTable
    from .delay import Delay
    from .expression import Expression
    from .operators import Choice, ChoiceOption, Map, Pass
    from .rules import Rules
    from .sub_workflow import SubWorkflow

_LAZY_EXPORTS = {
    "DecisionTable": ".decision_table",
    "Delay": ".delay",
    "Expression": ".expression",
    "Choice": ".operators",
    "ChoiceOption": ".operators",
    "Map": ".operators",
    "Pass": ".operators",
    "Rules": ".rules",
    "SubWorkflow": ".sub_workflow",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "DecisionTable",
    "Delay",
    "Expression",
    "Choice",
    "ChoiceOption",
    "Map",
    "Pass",
    "Rules",
    "SubWorkflow",
]
