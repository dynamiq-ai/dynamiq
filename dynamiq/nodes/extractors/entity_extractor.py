"""Deprecated module path.

``EntityExtractor`` has moved to :mod:`dynamiq.nodes.knowledge_graphs.entity_extractor` and been
renamed to :class:`KnowledgeGraphEntityExtractor`. This shim re-exports the public surface
under the old names so existing imports and serialized workflows keep working. Import from
``dynamiq.nodes.knowledge_graphs`` instead.
"""

import warnings

from dynamiq.nodes.knowledge_graphs.entity_extractor import *  # noqa: F401,F403
from dynamiq.nodes.knowledge_graphs.entity_extractor import KnowledgeGraphEntityExtractor  # noqa: F401

# Back-compat alias for the renamed class.
EntityExtractor = KnowledgeGraphEntityExtractor

warnings.warn(
    "dynamiq.nodes.extractors.entity_extractor is deprecated; import from "
    "dynamiq.nodes.knowledge_graphs instead (EntityExtractor -> KnowledgeGraphEntityExtractor).",
    DeprecationWarning,
    stacklevel=2,
)
