from enum import Enum


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
