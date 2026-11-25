"""The memory store's HTTP contract, against a simulated server. No model is called."""

import pytest

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.storages.memory import (
    CompositeMemoryStore,
    DynamiqMemoryStore,
    MemoryChangedError,
    MemoryNotFoundError,
    MemoryStoreError,
)
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


@pytest.mark.integration
def test_search_against_the_api(memory_store):
    """The search route and the description every entry carries, as the client calls them."""
    store = memory_store
    store.write("team/oncall.md", "Billing pages go to Marco at weekends.", user_id=USER_ID)
    store.write("preferences.md", "Prefers British English.", user_id=USER_ID)

    hits = store.search("weekend billing", user_id=USER_ID)
    assert [hit.path for hit in hits] == ["team/oncall.md"]
    assert hits[0].snippet and hits[0].score > 0
    assert hits[0].description == "Billing pages go to Marco at weekends."


@pytest.mark.integration
def test_search_on_a_platform_without_it(monkeypatch):
    """A platform older than search answers its route with 404; the agent is told what to use instead."""
    simulator = MemoryStoreAPISimulator(search_enabled=False)
    monkeypatch.setattr(DynamiqConnection, "connect", lambda self: simulator)
    store = _store(MEMORY_STORE_ID, "What you learn about this user.")

    with pytest.raises(MemoryStoreError, match="cannot be searched yet"):
        store.search("anything", user_id=USER_ID)


@pytest.mark.integration
def test_write_if_unchanged_against_the_api(memory_store):
    """Two writers over one memory: the platform refuses the write made from a stale read."""
    store = memory_store
    store.write("team/oncall.md", "Marco covers weekends.", user_id=USER_ID)
    note = store.read_note("team/oncall.md", user_id=USER_ID)
    store.write("team/oncall.md", "Priya covers weekends.", user_id=USER_ID)

    with pytest.raises(MemoryChangedError):
        store.write_if_unchanged("team/oncall.md", "Ana covers weekends.", note.content_hash, user_id=USER_ID)
    assert store.read("team/oncall.md", user_id=USER_ID) == "Priya covers weekends."

    fresh = store.read_note("team/oncall.md", user_id=USER_ID)
    entry = store.write_if_unchanged("team/oncall.md", "Ana covers weekends.", fresh.content_hash, user_id=USER_ID)
    assert entry.content_hash and entry.content_hash != fresh.content_hash

    store.write_if_unchanged("team/escalation.md", "Escalate after 15 minutes.", None, user_id=USER_ID)
    with pytest.raises(MemoryChangedError):
        store.write_if_unchanged("team/escalation.md", "Escalate to Priya.", None, user_id=USER_ID)
    assert store.read("team/escalation.md", user_id=USER_ID) == "Escalate after 15 minutes."


@pytest.mark.integration
def test_delete_if_unchanged_against_the_api(memory_store):
    """A delete made from a stale read is refused, so it cannot remove a correction it never saw."""
    store = memory_store
    store.write("team/oncall.md", "Marco covers weekends.", user_id=USER_ID)
    stale = store.read_note("team/oncall.md", user_id=USER_ID).content_hash
    store.write("team/oncall.md", "Priya covers weekends.", user_id=USER_ID)

    with pytest.raises(MemoryChangedError):
        store.delete_if_unchanged("team/oncall.md", stale, user_id=USER_ID)
    assert store.read("team/oncall.md", user_id=USER_ID) == "Priya covers weekends."

    fresh = store.read_note("team/oncall.md", user_id=USER_ID).content_hash
    assert store.delete_if_unchanged("team/oncall.md", fresh, user_id=USER_ID)
    assert not store.delete_if_unchanged("team/oncall.md", fresh, user_id=USER_ID)
