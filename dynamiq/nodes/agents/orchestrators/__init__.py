from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .graph import GraphOrchestrator
    from .graph_manager import GraphAgentManager
    from .graph_state import GraphState

_LAZY_EXPORTS = {
    "GraphOrchestrator": ".graph",
    "GraphAgentManager": ".graph_manager",
    "GraphState": ".graph_state",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "GraphOrchestrator",
    "GraphAgentManager",
    "GraphState",
]
