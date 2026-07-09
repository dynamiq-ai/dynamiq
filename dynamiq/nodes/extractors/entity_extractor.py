"""Deprecated module path.

``EntityExtractor`` has moved to :mod:`dynamiq.nodes.graphs.entity_extractor` and been
renamed to :class:`KnowledgeGraphEntityExtractor`. This shim re-exports the public surface
under the old names so existing imports and serialized workflows keep working. Import from
``dynamiq.nodes.graphs`` instead.
"""

import warnings

from dynamiq.nodes.graphs.entity_extractor import *  # noqa: F401,F403
from dynamiq.nodes.graphs.entity_extractor import (  # noqa: F401
    KnowledgeGraphEntityExtractor,
)

# Back-compat alias for the renamed class.
EntityExtractor = KnowledgeGraphEntityExtractor

warnings.warn(
    "dynamiq.nodes.extractors.entity_extractor is deprecated; import from "
    "dynamiq.nodes.graphs instead (EntityExtractor -> KnowledgeGraphEntityExtractor).",
    DeprecationWarning,
    stacklevel=2,
)
