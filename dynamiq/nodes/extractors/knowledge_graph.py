"""Deprecated module path.

``KnowledgeGraphWriter`` has moved to :mod:`dynamiq.nodes.graphs.writer`. This shim
re-exports it under the old path so existing imports and serialized workflows keep working.
Import from ``dynamiq.nodes.graphs`` instead.
"""

import warnings

from dynamiq.nodes.graphs.writer import *  # noqa: F401,F403
from dynamiq.nodes.graphs.writer import (  # noqa: F401
    KnowledgeGraphWriter,
    _entity_ids_by_doc,
)

warnings.warn(
    "dynamiq.nodes.extractors.knowledge_graph is deprecated; import KnowledgeGraphWriter "
    "from dynamiq.nodes.graphs instead.",
    DeprecationWarning,
    stacklevel=2,
)
