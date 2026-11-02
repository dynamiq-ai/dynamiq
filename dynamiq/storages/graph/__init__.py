from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .age import ApacheAgeGraphStore
    from .base import BaseGraphStore
    from .neo4j import Neo4jGraphStore
    from .neptune import NeptuneGraphStore

_LAZY_EXPORTS = {
    "ApacheAgeGraphStore": ".age",
    "BaseGraphStore": ".base",
    "Neo4jGraphStore": ".neo4j",
    "NeptuneGraphStore": ".neptune",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "ApacheAgeGraphStore",
    "BaseGraphStore",
    "Neo4jGraphStore",
    "NeptuneGraphStore",
]
