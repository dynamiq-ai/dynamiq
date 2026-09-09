import json

import pytest

from dynamiq.connections import Dynamiq
from dynamiq.storages.file import CompositeFileStore, DynamiqFileStore, InMemoryFileStore
from dynamiq.storages.file.composite import normalize_path


@pytest.fixture
def persistent():
    return InMemoryFileStore()


@pytest.fixture
def workspace():
    return InMemoryFileStore()


@pytest.fixture
def store(workspace, persistent):
    return CompositeFileStore(default=workspace, routes={"memories/": persistent})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("memories/notes.md", "memories/notes.md"),
        ("/memories/notes.md", "memories/notes.md"),
        ("./memories/notes.md", "memories/notes.md"),
        ("memories//notes.md", "memories/notes.md"),
        ("", ""),
        ("/", ""),
    ],
)
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected


def test_route_prefixes_are_normalized(workspace, persistent):
    store = CompositeFileStore(default=workspace, routes={"/memories/": persistent})
    assert list(store.routes) == ["memories/"]


def test_empty_route_prefix_is_rejected(workspace, persistent):
    with pytest.raises(ValueError):
        CompositeFileStore(default=workspace, routes={"/": persistent})


def test_writes_land_in_the_routed_store(store, workspace, persistent):
    store.store("memories/notes.md", "remembered")
    store.store("scratch.md", "ephemeral")

    assert persistent.retrieve("memories/notes.md") == b"remembered"
    assert workspace.retrieve("scratch.md") == b"ephemeral"
    assert not workspace.exists("memories/notes.md")


def test_leading_slash_addresses_the_same_file(store, persistent):
    store.store("/memories/notes.md", "remembered")

    assert persistent.exists("memories/notes.md")
    assert store.retrieve("memories/notes.md") == b"remembered"
    assert store.exists("/memories/notes.md")


def test_longest_prefix_wins(workspace):
    shallow, deep = InMemoryFileStore(), InMemoryFileStore()
    store = CompositeFileStore(default=workspace, routes={"memories/": shallow, "memories/team/": deep})

    store.store("memories/mine.md", "a")
    store.store("memories/team/ours.md", "b")

    assert shallow.exists("memories/mine.md")
    assert deep.exists("memories/team/ours.md")
    assert not shallow.exists("memories/team/ours.md")


def test_delete_and_exists_follow_the_route(store, persistent):
    store.store("memories/notes.md", "remembered")

    assert store.exists("memories/notes.md") is True
    assert store.delete("memories/notes.md") is True
    assert store.exists("memories/notes.md") is False
    assert not persistent.exists("memories/notes.md")


def test_list_files_at_root_merges_every_backend(store):
    store.store("scratch.md", "a")
    store.store("memories/notes.md", "b")

    paths = {info.path for info in store.list_files(recursive=True)}

    assert paths == {"scratch.md", "memories/notes.md"}


def test_list_files_in_a_routed_directory_delegates(store):
    store.store("scratch.md", "a")
    store.store("memories/notes.md", "b")

    paths = {info.path for info in store.list_files(directory="memories/", recursive=True)}

    assert paths == {"memories/notes.md"}


def test_list_files_above_several_routes_merges_them(workspace):
    """A directory holding more than one route belongs to none of them, so all of them answer."""
    handbook, personal = InMemoryFileStore(), InMemoryFileStore()
    store = CompositeFileStore(
        default=workspace,
        routes={"memories/handbook/": handbook, "memories/me/": personal},
    )
    store.store("scratch.md", "a")
    store.store("memories/handbook/deploys.md", "b")
    store.store("memories/me/style.md", "c")

    paths = {info.path for info in store.list_files(directory="memories/", recursive=True)}

    assert paths == {"memories/handbook/deploys.md", "memories/me/style.md"}
    assert handbook.exists("memories/handbook/deploys.md")
    assert personal.exists("memories/me/style.md")


def test_list_files_tolerates_a_backend_without_pattern_support(store):
    """InMemoryFileStore drops the `pattern` argument the base class declares."""
    store.store("memories/notes.md", "b")

    assert [info.path for info in store.list_files(recursive=True, pattern="*.md")] == ["memories/notes.md"]


def test_list_files_bytes_without_paths_skips_routed_stores(store, persistent):
    store.store("scratch.md", "a")
    store.store("memories/notes.md", "b")

    names = [f.name for f in store.list_files_bytes()]

    assert names == ["scratch.md"]


def test_list_files_bytes_with_paths_spans_backends(store):
    store.store("scratch.md", "a")
    store.store("memories/notes.md", "b")

    files = store.list_files_bytes(["scratch.md", "memories/notes.md"])

    assert {f.name for f in files} == {"scratch.md", "memories/notes.md"}


def test_type_is_package_path(store):
    assert store.type == "dynamiq.storages.file.CompositeFileStore"


def test_to_dict_serializes_sub_stores_and_hides_credentials(workspace):
    remote = DynamiqFileStore(
        connection=Dynamiq(url="https://api.example.ai/", api_key="secret-token"),
        memory_store_id="ms-123",
    )
    store = CompositeFileStore(default=workspace, routes={"memories/": remote})

    data = store.to_dict()

    assert data["default"]["type"] == "dynamiq.storages.file.InMemoryFileStore"
    assert data["routes"]["memories/"]["type"] == "dynamiq.storages.file.DynamiqFileStore"
    assert "secret-token" not in json.dumps(data)


def test_extracted_text_cache_is_resolved_per_path(workspace):
    """A workspace file may cache extracted text; one routed to a declining store may not."""
    remote = DynamiqFileStore(
        connection=Dynamiq(url="https://api.example.ai/", api_key="k"),
        memory_store_id="ms-123",
    )
    store = CompositeFileStore(default=workspace, routes={"memories/": remote})

    assert store.supports_extracted_text_cache("scratch.pdf") is True
    assert store.supports_extracted_text_cache("memories/report.pdf") is False
