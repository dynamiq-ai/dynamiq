"""Deprecated module path.

``GraphRetriever`` has moved to :mod:`dynamiq.nodes.knowledge_graphs.retriever` and been renamed to
:class:`KnowledgeGraphRetriever`. This shim re-exports the public surface under the old names
so existing imports and serialized workflows keep working. Import from
``dynamiq.nodes.knowledge_graphs`` instead.
"""

import warnings

from dynamiq.nodes.knowledge_graphs.retriever import *  # noqa: F401,F403
from dynamiq.nodes.knowledge_graphs.retriever import KnowledgeGraphRetriever  # noqa: F401

# Back-compat alias for the renamed class.
GraphRetriever = KnowledgeGraphRetriever

warnings.warn(
    "dynamiq.nodes.retrievers.graph is deprecated; import from dynamiq.nodes.knowledge_graphs instead "
    "(GraphRetriever -> KnowledgeGraphRetriever).",
    DeprecationWarning,
    stacklevel=2,
)
