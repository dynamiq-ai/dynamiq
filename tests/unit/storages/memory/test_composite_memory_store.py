import json

import pytest

from dynamiq.connections import Dynamiq
from dynamiq.storages.memory import CompositeMemoryStore, DynamiqMemoryStore, MemoryStoreError
from dynamiq.storages.memory.composite import normalize_path
from tests.unit.storages.memory.conftest import FakeMemoryStore


@pytest.fixture
def user():
    return FakeMemoryStore(description="What you learn about this user.")


@pytest.fixture
def team():
    return FakeMemoryStore(description="Conventions the whole team follows.")


@pytest.fixture
def store(user, team):
    return CompositeMemoryStore(routes={"user/": user, "team/": team})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("user/prefs.md", "user/prefs.md"),
        ("/user/prefs.md", "user/prefs.md"),
        ("./user/prefs.md", "user/prefs.md"),
        ("user//prefs.md", "user/prefs.md"),
        ("", ""),
        ("/", ""),
    ],
)
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected


@pytest.mark.parametrize("prefix", ["team", "team/", "/team/"])
def test_route_prefixes_are_normalized(user, team, prefix):
    store = CompositeMemoryStore(routes={"user/": user, prefix: team})
    assert set(store.routes) == {"user/", "team/"}


def test_routes_are_required(user):
    with pytest.raises(ValueError):
        CompositeMemoryStore(routes={})


def test_an_empty_prefix_is_rejected(user):
    """There is no catch-all: every memory is addressed by a prefix."""
    with pytest.raises(ValueError):
        CompositeMemoryStore(routes={"": user})


def test_writes_land_in_the_routed_memory_without_the_prefix(store, user, team):
    """The prefix is a mount point, so the store holds the path *within* it."""
    store.write("user/prefs.md", "likes docstrings")
    store.write("team/naming.md", "zx_ prefix")

    assert user.read("prefs.md") == "likes docstrings"
    assert team.read("naming.md") == "zx_ prefix"


def test_paths_come_back_mounted(store):
    """A path from `list` or `write` must be readable as-is: the mount point is restored."""
    written = store.write("user/prefs.md", "likes docstrings")

    assert written.path == "user/prefs.md"
    assert [entry.path for entry in store.list()] == ["user/prefs.md"]
    assert store.read(written.path) == "likes docstrings"


def test_a_route_can_be_remounted_without_moving_anything(user, team):
    """Because the prefix is not part of the key, renaming a route keeps its memories reachable."""
    CompositeMemoryStore(routes={"user/": user, "team/": team}).write("user/prefs.md", "remembered")

    remounted = CompositeMemoryStore(routes={"personal/": user, "team/": team})

    assert remounted.read("personal/prefs.md") == "remembered"


def test_leading_slash_addresses_the_same_memory(store, user):
    store.write("/user/prefs.md", "remembered")

    assert user.read("prefs.md") == "remembered"
    assert store.read("user/prefs.md") == "remembered"


def test_longest_prefix_wins(user, team):
    deep = FakeMemoryStore()
    store = CompositeMemoryStore(routes={"team/": team, "team/private/": deep})

    store.write("team/naming.md", "shallow")
    store.write("team/private/pay.md", "deep")

    assert team.read("naming.md") == "shallow"
    assert deep.read("pay.md") == "deep"
    assert {entry.path for entry in store.list()} == {"team/naming.md", "team/private/pay.md"}


def test_an_unrouted_path_raises_and_names_the_valid_prefixes(store):
    """A catch-all would file a mistyped path somewhere plausible; this tells the agent where to go."""
    with pytest.raises(MemoryStoreError) as excinfo:
        store.write("notes.md", "x")

    message = str(excinfo.value)
    assert "team/" in message and "user/" in message


def test_delete_follows_the_route(store, user):
    store.write("user/prefs.md", "remembered")

    assert store.delete("user/prefs.md") is True
    assert store.delete("user/prefs.md") is False
    assert not user.list()


def test_list_without_a_prefix_merges_every_memory(store):
    store.write("user/prefs.md", "a")
    store.write("team/naming.md", "b")

    assert {entry.path for entry in store.list()} == {"user/prefs.md", "team/naming.md"}


def test_list_with_a_prefix_delegates(store):
    store.write("user/prefs.md", "a")
    store.write("team/naming.md", "b")

    assert [entry.path for entry in store.list("team/")] == ["team/naming.md"]


def test_describe_namespaces_composes_the_routes(store):
    assert store.describe_namespaces() == {
        "user/": "What you learn about this user.",
        "team/": "Conventions the whole team follows.",
    }


def test_describe_namespaces_nests(user):
    """A composite inside a composite composes prefixes, because each store answers the same question."""
    inner = CompositeMemoryStore(routes={"private/": FakeMemoryStore(description="Sensitive team notes.")})
    store = CompositeMemoryStore(routes={"user/": user, "team/": inner})

    assert store.describe_namespaces() == {
        "user/": "What you learn about this user.",
        "team/private/": "Sensitive team notes.",
    }


def test_an_undescribed_memory_is_omitted(user):
    store = CompositeMemoryStore(routes={"user/": user, "scratch/": FakeMemoryStore()})

    assert store.describe_namespaces() == {"user/": "What you learn about this user."}


def test_type_is_package_path(store):
    assert store.type == "dynamiq.storages.memory.CompositeMemoryStore"


def test_to_dict_serializes_routes_and_hides_credentials(user):
    remote = DynamiqMemoryStore(
        connection=Dynamiq(url="https://api.example.ai/", api_key="secret-token"),
        memory_store_id="ms-123",
        user_id="u-42",
    )
    store = CompositeMemoryStore(routes={"user/": user, "team/": remote})

    data = store.to_dict()

    assert data["type"] == "dynamiq.storages.memory.CompositeMemoryStore"
    assert data["routes"]["team/"]["type"] == "dynamiq.storages.memory.DynamiqMemoryStore"
    assert "secret-token" not in json.dumps(data)


def test_two_routes_over_one_store_are_refused(user):
    """After stripping they would map different agent paths onto the same key."""
    shared = FakeMemoryStore()

    with pytest.raises(ValueError) as excinfo:
        CompositeMemoryStore(routes={"team/": shared, "me/": shared})

    assert "same memory store" in str(excinfo.value)
