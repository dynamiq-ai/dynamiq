from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .base import BaseValidator
    from .regex_match import MatchType, RegexMatch
    from .valid_choices import ValidChoices
    from .valid_json import ValidJSON
    from .valid_python import ValidPython

_LAZY_EXPORTS = {
    "BaseValidator": ".base",
    "MatchType": ".regex_match",
    "RegexMatch": ".regex_match",
    "ValidChoices": ".valid_choices",
    "ValidJSON": ".valid_json",
    "ValidPython": ".valid_python",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "BaseValidator",
    "MatchType",
    "RegexMatch",
    "ValidChoices",
    "ValidJSON",
    "ValidPython",
]
