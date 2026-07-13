"""Deprecated module path.

``KnowledgeGraphWriter`` has moved to :mod:`dynamiq.nodes.knowledge_graph.writer`. This shim
re-exports it under the old path so existing imports and serialized workflows keep working.
Import from ``dynamiq.nodes.knowledge_graph`` instead.
"""

import warnings

from dynamiq.nodes.knowledge_graph.writer import *  # noqa: F401,F403
from dynamiq.nodes.knowledge_graph.writer import KnowledgeGraphWriter, _entity_ids_by_doc  # noqa: F401

warnings.warn(
    "dynamiq.nodes.extractors.knowledge_graph is deprecated; import KnowledgeGraphWriter "
    "from dynamiq.nodes.knowledge_graph instead.",
    DeprecationWarning,
    stacklevel=2,
)
