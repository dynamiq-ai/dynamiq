from enum import Enum
from typing import Any

from pydantic import BaseModel


class NodeGroup(str, Enum):
    """
    Enumeration of node groups that categorize different types of nodes.

    Each group represents a collection of related node types, providing a higher-level
    classification of the system's components.
    """

    LLMS = "llms"
    OPERATORS = "operators"
    EMBEDDERS = "embedders"
    RANKERS = "rankers"
    CONVERTERS = "converters"
    RETRIEVERS = "retrievers"
    SPLITTERS = "splitters"
    WRITERS = "writers"
    UTILS = "utils"
    TOOLS = "tools"
    AGENTS = "agents"
    AUDIO = "audio"
    VALIDATORS = "validators"


class InferenceMode(str, Enum):
    """
    Enumeration of inference types.
    """

    DEFAULT = "DEFAULT"
    XML = "XML"
    FUNCTION_CALLING = "FUNCTION_CALLING"
    STRUCTURED_OUTPUT = "STRUCTURED_OUTPUT"


class Behavior(str, Enum):
    RAISE = "raise"
    RETURN = "return"


class ConditionOperator(str, Enum):
    """Enum representing various condition operators."""

    OR = "or"
    AND = "and"
    BOOLEAN_EQUALS = "boolean-equals"
    BOOLEAN_EQUALS_PATH = "boolean-equals-path"
    NUMERIC_EQUALS = "numeric-equals"
    NUMERIC_EQUALS_PATH = "numeric-equals-path"
    NUMERIC_GREATER_THAN = "numeric-greater-than"
    NUMERIC_GREATER_THAN_PATH = "numeric-greater-than-path"
    NUMERIC_GREATER_THAN_OR_EQUALS = "numeric-greater-than-or-equals"
    NUMERIC_GREATER_THAN_OR_EQUALS_PATH = "numeric-greater-than-or-equals-path"
    NUMERIC_LESS_THAN = "numeric-less-than"
    NUMERIC_LESS_THAN_PATH = "numeric-less-than-path"
    NUMERIC_LESS_THAN_OR_EQUALS = "numeric-less-than-or-equals"
    NUMERIC_LESS_THAN_OR_EQUALS_PATH = "numeric-less-than-or-equals-path"
    STRING_EQUALS = "string-equals"
    STRING_EQUALS_PATH = "string-equals-path"
    STRING_GREATER_THAN = "string-greater-than"
    STRING_GREATER_THAN_PATH = "string-greater-than-path"
    STRING_GREATER_THAN_OR_EQUALS = "string-greater-than-or-equals"
    STRING_GREATER_THAN_OR_EQUALS_PATH = "string-greater-than-or-equals-path"
    STRING_LESS_THAN = "string-less-than"
    STRING_LESS_THAN_PATH = "string-less-than-path"
    STRING_LESS_THAN_OR_EQUALS = "string-less-than-or-equals"
    STRING_LESS_THAN_OR_EQUALS_PATH = "string-less-than-or-equals-path"


class ChoiceCondition(BaseModel):
    """Represents a condition."""

    variable: str | None = None
    operator: ConditionOperator | None = None
    value: Any = None
    is_not: bool = False
    operands: list["ChoiceCondition"] | None = None
