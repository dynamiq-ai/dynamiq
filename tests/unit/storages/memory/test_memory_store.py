from dynamiq.storages.memory import MemoryEntry, MemoryNotFoundError
from tests.unit.storages.memory.conftest import FakeMemoryStore


def test_an_entry_from_a_store_without_descriptions_is_undescribed():
    assert MemoryEntry(path="a.md", description=None).description == ""


def test_default_search_skips_a_memory_deleted_while_it_runs():
    class Vanishing(FakeMemoryStore):
        def read(self, path, user_id=None):
            if path == "ghost.md":
                raise MemoryNotFoundError("Memory 'ghost.md' not found", operation="read", path=path)
            return super().read(path, user_id)

    store = Vanishing()
    store.write("ghost.md", "billing rota")
    store.write("oncall.md", "billing rota")

    assert [hit.path for hit in store.search("billing rota")] == ["oncall.md"]
