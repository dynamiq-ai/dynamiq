import pytest

from dynamiq.storages.memory import MemoryEntry, MemoryNotFoundError, MemoryStore


class FakeMemoryStore(MemoryStore):
    """A local stand-in for the API-backed store.

    The shipped implementation is API-backed only, so the tests carry their own double rather than
    the package shipping one nobody would run in production.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._memories: dict[str, str] = {}

    def list(self, prefix: str = "", user_id: str | None = None) -> list[MemoryEntry]:
        return [
            MemoryEntry(path=path, size=len(content))
            for path, content in self._memories.items()
            if path.startswith(prefix)
        ]

    def read(self, path: str, user_id: str | None = None) -> str:
        if path not in self._memories:
            raise MemoryNotFoundError(f"Memory '{path}' not found", operation="read", path=path)
        return self._memories[path]

    def write(self, path: str, content: str, user_id: str | None = None) -> MemoryEntry:
        self._memories[path] = content
        return MemoryEntry(path=path, size=len(content))

    def delete(self, path: str, user_id: str | None = None) -> bool:
        return self._memories.pop(path, None) is not None


@pytest.fixture
def fake_store():
    return FakeMemoryStore()
