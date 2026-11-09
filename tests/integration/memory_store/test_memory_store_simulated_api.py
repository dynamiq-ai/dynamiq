"""The memory store's HTTP contract, against a simulated server. No model is called."""

import pytest

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.storages.memory import CompositeMemoryStore, DynamiqMemoryStore, MemoryNotFoundError, MemoryStoreError
from tests.helpers.memory_store_api import MemoryStoreAPISimulator

MEMORY_STORE_ID = "ms-user"
USER_ID = "u-test"


def _store(store_id: str, description: str) -> DynamiqMemoryStore:
    return DynamiqMemoryStore(
        connection=DynamiqConnection(url="https://memory-store.simulated", api_key="simulated-key"),
        memory_store_id=store_id,
        description=description,
    )


@pytest.fixture
def memory_store(monkeypatch):
    """The agent's memory, served by the simulator for the duration of one test."""
    simulator = MemoryStoreAPISimulator()
    monkeypatch.setattr(DynamiqConnection, "connect", lambda self: simulator)
    return _store(MEMORY_STORE_ID, "What you learn about this user.")


@pytest.mark.integration
def test_store_crud_against_the_api(memory_store):
    """Every endpoint of the contract, without a model in the loop.

    Run this first when bringing a server up: it isolates the HTTP contract from agent behaviour,
    so a failure here is the server's, not the model's.
    """
    store = memory_store
    path = "probe.md"

    entry = store.write(path, "probe body", user_id=USER_ID)
    assert entry.path == path, f"Server returned a different path: {entry.path}"

    assert store.read(path, user_id=USER_ID) == "probe body"
    assert path in [e.path for e in store.list(user_id=USER_ID)]

    store.write(path, "updated body", user_id=USER_ID)  # upsert: no overwrite flag, no 409
    assert store.read(path, user_id=USER_ID) == "updated body"

    assert store.delete(path, user_id=USER_ID) is True
    assert store.delete(path, user_id=USER_ID) is False, "Deleting a missing memory must return False, not raise."


@pytest.mark.integration
def test_store_error_branches(memory_store):
    """The parts of the contract a server is most likely to leave out."""
    store = memory_store

    with pytest.raises(MemoryNotFoundError):
        store.read("never-written.md", user_id=USER_ID)

    assert store.list("empty/", user_id=USER_ID) == []

    with pytest.raises(MemoryStoreError, match="8192|8,192"):
        store.write("digest.md", "x" * 8193, user_id=USER_ID)

    # Mounted behind a prefix, a path outside every route is refused by name.
    with pytest.raises(MemoryStoreError) as excinfo:
        CompositeMemoryStore(routes={"user/": store}).write("outside.md", "x", user_id=USER_ID)
    assert "user/" in str(excinfo.value)
