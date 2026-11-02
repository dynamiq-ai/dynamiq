from typing import TYPE_CHECKING

from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .agent import Agent
    from .base import Agent as BaseAgent
    from .base import AgentManager

_LAZY_EXPORTS = {
    "Agent": ".agent",
    "BaseAgent": ".base:Agent",
    "AgentManager": ".base",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "InferenceMode",
    "Agent",
    "BaseAgent",
    "AgentManager",
]
