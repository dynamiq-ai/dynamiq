from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from dynamiq.connections import Dynamiq
from dynamiq.nodes.tools.memory_store_tool import MemoryStoreTool, MemoryStoreToolInputSchema
from dynamiq.runnables import RunnableStatus
from dynamiq.storages.memory import (
    CompositeMemoryStore,
    DynamiqMemoryStore,
    MemoryEntry,
    MemoryLink,
    MemoryNote,
    MemoryStoreError,
)
from tests.unit.storages.memory.conftest import FakeMemoryStore


@pytest.fixture
def backend():
    return FakeMemoryStore(description="What you learn about this user.")


@pytest.fixture
def tool(backend):
    return MemoryStoreTool(backend=backend)


def run(tool, **kwargs):
    return tool.run(input_data=kwargs)


def content(tool, **kwargs):
    result = run(tool, **kwargs)
    assert result.status == RunnableStatus.SUCCESS, result.error
    return result.output["content"]


def test_list_when_empty(tool):
    assert content(tool, action="list") == "No memories yet."


def test_write_then_read(tool, backend):
    assert "preferences.md" in content(tool, action="write", path="preferences.md", content="Prefers British English.")
    assert backend.read("preferences.md") == "Prefers British English."
    assert content(tool, action="read", path="preferences.md") == "Prefers British English."


def test_list_shows_paths(tool):
    content(tool, action="write", path="preferences.md", content="abc")

    result = run(tool, action="list")

    assert "preferences.md" in result.output["content"]
    assert result.output["paths"] == ["preferences.md"]


def test_list_narrows_to_a_prefix(tool):
    content(tool, action="write", path="user/a.md", content="a")
    content(tool, action="write", path="team/b.md", content="b")

    assert run(tool, action="list", path="team/").output["paths"] == ["team/b.md"]


def test_edit_replaces_in_place(tool, backend):
    content(tool, action="write", path="preferences.md", content="Prefers British English.")

    content(tool, action="edit", path="preferences.md", find="British", replace="Australian")

    assert backend.read("preferences.md") == "Prefers Australian English."


def test_edit_requires_a_unique_match(tool):
    content(tool, action="write", path="notes.md", content="tabs. tabs.")

    result = run(tool, action="edit", path="notes.md", find="tabs", replace="spaces")

    assert result.status == RunnableStatus.FAILURE
    assert "appears 2 times" in str(result.error)


def test_edit_replace_all(tool, backend):
    content(tool, action="write", path="notes.md", content="tabs. tabs.")

    content(tool, action="edit", path="notes.md", find="tabs", replace="spaces", replace_all=True)

    assert backend.read("notes.md") == "spaces. spaces."


def test_edit_reports_missing_text(tool):
    content(tool, action="write", path="notes.md", content="abc")

    result = run(tool, action="edit", path="notes.md", find="zzz", replace="y")

    assert result.status == RunnableStatus.FAILURE
    assert "does not appear" in str(result.error)


@pytest.mark.parametrize("replace_all", [True, False])
def test_edit_refuses_an_empty_find(tool, backend, replace_all):
    """It matches between every character, so replace_all would interleave the replacement."""
    content(tool, action="write", path="preferences.md", content="Prefers British English.")

    result = run(tool, action="edit", path="preferences.md", find="", replace="X", replace_all=replace_all)

    assert result.status == RunnableStatus.FAILURE
    assert backend.read("preferences.md") == "Prefers British English."


def test_edit_accepts_an_empty_replace(tool, backend):
    """Emptying a match is a real edit: only 'find' is constrained."""
    content(tool, action="write", path="notes.md", content="tabs and spaces")

    content(tool, action="edit", path="notes.md", find=" and spaces", replace="")

    assert backend.read("notes.md") == "tabs"


def test_edit_on_a_missing_memory_points_at_write(tool):
    result = run(tool, action="edit", path="nope.md", find="a", replace="b")

    assert result.status == RunnableStatus.FAILURE
    assert "'write'" in str(result.error)


def test_user_id_is_hidden_from_agent_tool_schema():
    from dynamiq.nodes.agents.components.schema_generator import generate_function_calling_schemas

    tool_name = "memory-store"
    tool = MagicMock()
    tool.name = tool_name
    tool.description = "test"
    tool.input_schema = MemoryStoreToolInputSchema
    tool.resolved_input_schema = MemoryStoreToolInputSchema

    schema = next(
        s
        for s in generate_function_calling_schemas(
            tools=[tool], delegation_allowed=False, sanitize_tool_name=lambda name: name
        )
        if s["function"]["name"] == tool_name
    )
    properties = schema["function"]["parameters"]["properties"]

    assert "user_id" not in properties
    assert "action" in properties
    assert "path" in properties


def test_a_caller_supplied_user_id_reaches_the_backend():
    """The tenant is bound on the tool - by the agent from the run, or by a direct caller."""
    seen = {}

    class Recording(FakeMemoryStore):
        def write(self, path, content, user_id=None):
            seen["user_id"] = user_id
            return super().write(path, content, user_id)

    content(MemoryStoreTool(backend=Recording(), user_id="u-99"), action="write", path="p.md", content="x")

    assert seen["user_id"] == "u-99"


def test_delete(tool):
    content(tool, action="write", path="stale.md", content="x")

    assert "Deleted" in content(tool, action="delete", path="stale.md")
    assert "no memory" in content(tool, action="delete", path="stale.md")


def test_read_missing_points_at_list(tool):
    result = run(tool, action="read", path="nope.md")

    assert result.status == RunnableStatus.FAILURE
    assert "list" in str(result.error)


@pytest.mark.parametrize(
    "call",
    [
        {"action": "read"},
        {"action": "write", "path": "a.md"},
        {"action": "edit", "path": "a.md", "find": "x"},
        {"action": "delete"},
    ],
)
def test_missing_arguments_are_rejected(tool, call):
    assert run(tool, **call).status == RunnableStatus.FAILURE


@pytest.mark.parametrize(
    "call",
    [
        {"action": "write", "path": "a.md", "content": "x"},
        {"action": "edit", "path": "a.md", "find": "x", "replace": "y"},
        {"action": "delete", "path": "a.md"},
    ],
)
def test_read_only_rejects_every_mutation(backend, call):
    tool = MemoryStoreTool(backend=backend, write_enabled=False)

    result = run(tool, **call)

    assert result.status == RunnableStatus.FAILURE
    assert "read-only" in str(result.error)


def test_read_only_still_lists_and_reads(backend):
    backend.write("preferences.md", "remembered")
    tool = MemoryStoreTool(backend=backend, write_enabled=False)

    assert content(tool, action="read", path="preferences.md") == "remembered"
    assert "preferences.md" in content(tool, action="list")


def test_read_only_is_stated_in_the_description(backend):
    assert "READ-ONLY" in MemoryStoreTool(backend=backend, write_enabled=False).description
    assert "READ-ONLY" not in MemoryStoreTool(backend=backend).description


def test_the_description_names_each_memory(backend):
    """Straight from the stores, so there is no parallel structure to keep in sync."""
    tool = MemoryStoreTool(
        backend=CompositeMemoryStore(
            routes={
                "user/": FakeMemoryStore(description="What you learn about this user."),
                "team/": FakeMemoryStore(description="Conventions the whole team follows."),
            }
        )
    )

    assert "- user/ - What you learn about this user." in tool.description
    assert "- team/ - Conventions the whole team follows." in tool.description


def test_the_description_states_the_size_limit_of_a_writable_memory(backend):
    limited = DynamiqMemoryStore(connection=Dynamiq(url="https://api.example.ai", api_key="key"), memory_store_id="ms")
    composite = CompositeMemoryStore(routes={"user/": backend, "team/": limited})

    assert "Each memory holds at most 8,192 characters" in MemoryStoreTool(backend=composite).description
    assert "Each memory holds" not in MemoryStoreTool(backend=composite, write_enabled=False).description
    assert "Each memory holds" not in MemoryStoreTool(backend=backend).description


def test_an_unrouted_path_is_relayed_with_the_valid_prefixes(backend):
    tool = MemoryStoreTool(backend=CompositeMemoryStore(routes={"user/": backend}))

    result = run(tool, action="write", path="notes.md", content="x")

    assert result.status == RunnableStatus.FAILURE
    assert "user/" in str(result.error)


def test_routes_through_a_composite(backend):
    team = FakeMemoryStore()
    tool = MemoryStoreTool(backend=CompositeMemoryStore(routes={"user/": backend, "team/": team}))

    content(tool, action="write", path="team/naming.md", content="zx_ prefix")

    assert team.read("naming.md") == "zx_ prefix", "the mount point is stripped before the store sees it"
    assert content(tool, action="read", path="team/naming.md") == "zx_ prefix"
    assert not backend.list()


def test_backend_is_not_serialized(tool):
    assert "backend" not in tool.to_dict()


def test_the_listing_says_what_each_memory_holds():
    """A bare path gives the model nothing to judge relevance by, so it lists and reads nothing."""
    tool = MemoryStoreTool(
        backend=CompositeMemoryStore(
            routes={
                "team/": FakeMemoryStore(description="Conventions the whole team follows."),
                "me/": FakeMemoryStore(description="What you learn about this specific user."),
            }
        )
    )
    content(tool, action="write", path="team/naming.md", content="prefix vars with zx_")
    content(tool, action="write", path="me/style.md", content="docstrings")

    listing = content(tool, action="list")

    assert "team/ — Conventions the whole team follows." in listing
    assert "me/ — What you learn about this specific user." in listing
    assert "- team/naming.md" in listing and "- me/style.md" in listing


def test_the_listing_of_an_undescribed_memory_still_shows_paths(backend):
    tool = MemoryStoreTool(backend=FakeMemoryStore())
    content(tool, action="write", path="notes.md", content="x")

    assert "notes.md" in content(tool, action="list")


def test_a_large_listing_describes_the_most_recent_memories_that_fit():
    start = datetime(2026, 1, 1)
    entries = [
        MemoryEntry(
            path=f"notes/n{i:03}.md",
            size=10,
            description="Note on one topic. " * 6,
            updated_at=start + timedelta(days=i),
        )
        for i in range(250)
    ]
    entries[0] = entries[0].model_copy(update={"updated_at": None})
    entries[1] = entries[1].model_copy(update={"updated_at": entries[1].updated_at.replace(tzinfo=timezone.utc)})

    class ManyMemories(FakeMemoryStore):
        def list(self, prefix: str = "", user_id: str | None = None):
            return entries

    result = run(MemoryStoreTool(backend=ManyMemories()), action="list")
    text = result.output["content"]
    described = text.count(" — Note on one topic.")

    assert 0 < described < 250
    assert "notes/n249.md — Note on one topic." in text
    assert "  - notes/n000.md (10 chars)" in text
    assert f"250 memories; the {described} changed most recently are described." in text
    assert len(result.output["paths"]) == 250


def test_search_finds_a_memory_by_its_content(tool):
    content(tool, action="write", path="team/oncall.md", content="Billing pages go to Marco at weekends.")
    content(tool, action="write", path="preferences.md", content="Prefers British English.")

    result = run(tool, action="search", query="who is paged at weekends")

    assert result.output["paths"] == ["team/oncall.md"]
    assert "No memories" in content(tool, action="search", query="kubernetes")


def test_search_needs_a_query(tool):
    result = run(tool, action="search", query="  ")

    assert result.status == RunnableStatus.FAILURE


def test_write_names_memories_on_the_same_topic(tool):
    content(tool, action="write", path="playbook/refund-policy.md", content="Refunds over 500 EUR need approval.")

    text = content(tool, action="write", path="playbook/refund-exceptions.md", content="Exceptions for refunds.")

    assert "Memories on nearby topics: playbook/refund-policy.md" in text


def test_write_names_no_memory_that_shares_only_a_common_word(tool):
    content(tool, action="write", path="customers/larkspur.md", content="Larkspur renews in March.")

    assert "nearby topics" not in content(tool, action="write", path="team/march-offsite-agenda.md", content="Agenda.")


def test_write_names_related_memories_only_inside_its_own_memory():
    user, team = FakeMemoryStore(), FakeMemoryStore()
    team.write("timezone-policy.md", "Everyone schedules in the user's timezone.")
    tool = MemoryStoreTool(backend=CompositeMemoryStore(routes={"user/": user, "team/": team}))

    assert content(tool, action="write", path="user/timezone.md", content="Works from Kyiv.") == (
        "Remembered in 'user/timezone.md'."
    )


def test_write_survives_a_failing_search(tool, backend, monkeypatch):
    monkeypatch.setattr(type(backend), "search", MagicMock(side_effect=MemoryStoreError("index down")))

    assert content(tool, action="write", path="a.md", content="x") == "Remembered in 'a.md'."


def test_read_names_the_memories_around_it():
    class LinkedMemoryStore(FakeMemoryStore):
        def read_note(self, path: str, user_id: str | None = None) -> MemoryNote:
            return MemoryNote(
                path=path,
                content=self.read(path, user_id),
                links=[MemoryLink(target="dana-okafor", path="people/dana-okafor.md"), MemoryLink(target="msa")],
                backlinks=["projects/q3-pricing.md"],
            )

    tool = MemoryStoreTool(backend=LinkedMemoryStore())
    content(tool, action="write", path="customers/larkspur.md", content="12% discount.")

    text = content(tool, action="read", path="customers/larkspur.md")

    assert text == (
        "12% discount.\n\n[Links, not part of the memory]\nLinks to: people/dana-okafor.md\n"
        "Unresolved links, no memory has these names: msa\nLinked from: projects/q3-pricing.md"
    )


def test_write_drops_the_links_a_read_appended():
    class LinkedMemoryStore(FakeMemoryStore):
        def read_note(self, path: str, user_id: str | None = None) -> MemoryNote:
            return MemoryNote(
                path=path,
                content=self.read(path, user_id),
                links=[MemoryLink(target="dana-okafor", path="people/dana-okafor.md")],
            )

    backend = LinkedMemoryStore()
    tool = MemoryStoreTool(backend=backend)
    content(tool, action="write", path="customers/larkspur.md", content="12% discount.")
    read_back = content(tool, action="read", path="customers/larkspur.md")

    content(tool, action="write", path="customers/larkspur.md", content=read_back + "\nRenews in March.")

    assert backend.read("customers/larkspur.md") == "12% discount.\nRenews in March."


def test_write_names_no_memory_that_shares_only_its_folder():
    backend = FakeMemoryStore()
    backend.write("team/rota.md", "Holiday rota for December.")
    tool = MemoryStoreTool(backend=backend)

    assert content(tool, action="write", path="team/oncall.md", content="Pager rotation.") == (
        "Remembered in 'team/oncall.md'."
    )


@pytest.mark.parametrize("change", ["rewritten", "deleted"])
def test_write_refuses_to_replace_a_change_made_since_the_read(tool, backend, change):
    content(tool, action="write", path="team/oncall.md", content="Marco covers weekends.")
    content(tool, action="read", path="team/oncall.md")
    if change == "rewritten":
        backend.write("team/oncall.md", "Priya covers weekends from March.")
    else:
        backend.delete("team/oncall.md")
    current = backend._memories.get("team/oncall.md")

    for _ in range(2):
        result = run(tool, action="write", path="team/oncall.md", content="Marco covers weekends and holidays.")
        assert result.status == RunnableStatus.FAILURE
        assert "changed since you read it" in result.error.message
    assert backend._memories.get("team/oncall.md") == current

    run(tool, action="read", path="team/oncall.md")
    assert "Remembered" in content(tool, action="write", path="team/oncall.md", content="Both cover holidays.")


def test_edit_applies_only_to_the_text_it_read():
    class RacedMemoryStore(FakeMemoryStore):
        def read_note(self, path: str, user_id: str | None = None) -> MemoryNote:
            note = super().read_note(path, user_id)
            self.write(path, "Priya covers weekends.")
            return note

    store = RacedMemoryStore()
    store.write("team/oncall.md", "Marco covers weekends.")

    result = run(MemoryStoreTool(backend=store), action="edit", path="team/oncall.md", find="Marco", replace="Ana")

    assert result.status == RunnableStatus.FAILURE
    assert store.read("team/oncall.md") == "Priya covers weekends."


def test_a_write_over_a_memory_the_run_never_read_is_refused(tool, backend):
    backend.write("notes.md", "Marco covers weekends.")
    content(tool, action="read", path="notes.md")
    other_run = tool.clone()

    assert (
        "already exists and you haven't read it"
        in run(other_run, action="write", path="notes.md", content="Ana covers weekends.").error.message
    )
    assert backend.read("notes.md") == "Marco covers weekends."

    content(other_run, action="read", path="notes.md")
    assert "Remembered" in content(other_run, action="write", path="notes.md", content="Ana covers weekends.")
    assert "Remembered" in content(other_run, action="write", path="new.md", content="Created without a read.")


def test_delete_refuses_a_change_made_since_the_read(tool, backend):
    content(tool, action="write", path="team/oncall.md", content="Marco covers weekends.")
    content(tool, action="read", path="team/oncall.md")
    backend.write("team/oncall.md", "Priya covers weekends from March.")

    result = run(tool, action="delete", path="team/oncall.md")

    assert result.status == RunnableStatus.FAILURE
    assert "changed since you read it, so it was not deleted" in result.error.message
    assert backend.read("team/oncall.md") == "Priya covers weekends from March."

    content(tool, action="read", path="team/oncall.md")
    assert "Deleted" in content(tool, action="delete", path="team/oncall.md")


def test_a_store_that_reports_no_hash_on_write_keeps_later_writes_conditional():
    class UnhashedWritesMemoryStore(FakeMemoryStore):
        def write(self, path: str, content: str, user_id: str | None = None) -> MemoryEntry:
            return super().write(path, content, user_id).model_copy(update={"content_hash": ""})

    store = UnhashedWritesMemoryStore()
    tool = MemoryStoreTool(backend=store)
    content(tool, action="write", path="team/oncall.md", content="Marco covers weekends.")
    store.write("team/oncall.md", "Priya covers weekends.")

    result = run(tool, action="write", path="team/oncall.md", content="Marco covers holidays.")

    assert result.status == RunnableStatus.FAILURE
    assert store.read("team/oncall.md") == "Priya covers weekends."


def test_a_store_without_content_hashes_writes_unconditionally():
    class UnhashedMemoryStore(FakeMemoryStore):
        def read_note(self, path: str, user_id: str | None = None) -> MemoryNote:
            return MemoryNote(path=path, content=self.read(path, user_id))

    store = UnhashedMemoryStore()
    tool = MemoryStoreTool(backend=store)
    store.write("a.md", "one")
    content(tool, action="read", path="a.md")
    store.write("a.md", "two")

    assert "Remembered" in content(tool, action="write", path="a.md", content="three")
    assert store.read("a.md") == "three"
