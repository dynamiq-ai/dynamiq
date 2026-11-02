from typing import TYPE_CHECKING

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .base import (
        MemoryEntry,
        MemoryNotFoundError,
        MemoryPermissionError,
        MemoryStore,
        MemoryStoreConfig,
        MemoryStoreError,
        render_namespaces,
    )
    from .composite import CompositeMemoryStore
    from .dynamiq import DynamiqMemoryStore

_LAZY_EXPORTS = {
    "MemoryEntry": ".base",
    "MemoryNotFoundError": ".base",
    "MemoryPermissionError": ".base",
    "MemoryStore": ".base",
    "MemoryStoreConfig": ".base",
    "MemoryStoreError": ".base",
    "render_namespaces": ".base",
    "CompositeMemoryStore": ".composite",
    "DynamiqMemoryStore": ".dynamiq",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "MemoryEntry",
    "MemoryNotFoundError",
    "MemoryPermissionError",
    "MemoryStore",
    "MemoryStoreConfig",
    "MemoryStoreError",
    "render_namespaces",
    "CompositeMemoryStore",
    "DynamiqMemoryStore",
]
