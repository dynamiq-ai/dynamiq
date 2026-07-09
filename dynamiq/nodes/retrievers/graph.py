"""Deprecated module path.

``GraphRetriever`` has moved to :mod:`dynamiq.nodes.graphs.retriever` and been renamed to
:class:`KnowledgeGraphRetriever`. This shim re-exports the public surface under the old names
so existing imports and serialized workflows keep working. Import from
``dynamiq.nodes.graphs`` instead.
"""

import warnings

from dynamiq.nodes.graphs.retriever import *  # noqa: F401,F403
from dynamiq.nodes.graphs.retriever import (  # noqa: F401
    GraphRetrieverInputSchema,
    KnowledgeGraphRetriever,
    _compile_edge_filters,
)

# Back-compat alias for the renamed class.
GraphRetriever = KnowledgeGraphRetriever

warnings.warn(
    "dynamiq.nodes.retrievers.graph is deprecated; import from dynamiq.nodes.graphs instead "
    "(GraphRetriever -> KnowledgeGraphRetriever).",
    DeprecationWarning,
    stacklevel=2,
)
