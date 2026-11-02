from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from dynamiq.checkpoints.backends.base import CheckpointBackend
    from dynamiq.checkpoints.backends.filesystem import FileSystem
    from dynamiq.checkpoints.backends.in_memory import InMemory
    from dynamiq.checkpoints.backends.postgresql import PostgreSQL

_LAZY_EXPORTS = {
    "CheckpointBackend": ".base",
    "FileSystem": ".filesystem",
    "InMemory": ".in_memory",
    "PostgreSQL": ".postgresql",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "CheckpointBackend",
    "FileSystem",
    "InMemory",
    "PostgreSQL",
]
