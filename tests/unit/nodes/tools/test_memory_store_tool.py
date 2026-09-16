import json
from unittest.mock import MagicMock, patch

import pytest

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.tools.memory_store_tool import MemoryStoreTool, MemoryStoreToolInputSchema
from dynamiq.nodes.types import InferenceMode
from dynamiq.nodes.utils import Input
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.storages.memory import CompositeMemoryStore
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
    """The tenant arrives as tool input - from tool_params or a workflow mapping, not the model."""
    seen = {}

    class Recording(FakeMemoryStore):
        def write(self, path, content, user_id=None):
            seen["user_id"] = user_id
            return super().write(path, content, user_id)

    content(MemoryStoreTool(backend=Recording()), action="write", path="p.md", content="x", user_id="u-99")

    assert seen["user_id"] == "u-99"


def test_user_id_arrives_through_a_workflow_input_transformer():
    """The real delivery path: a selector maps the flow's user_id onto the tool's hidden field.

    The model supplies only the memory arguments; the tenant comes from the run, so one deployed
    agent serves many users.
    """
    seen = {}

    class Recording(FakeMemoryStore):
        def write(self, path, content, user_id=None):
            seen["user_id"] = user_id
            return super().write(path, content, user_id)

    start = Input(id="start")
    tool = MemoryStoreTool(
        backend=Recording(),
        input_transformer=InputTransformer(selector={"user_id": "$.start.output.user_id"}),
    )
    agent = Agent(
        id="agent",
        name="assistant",
        llm=OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o"),
        role="remember things",
        tools=[tool],
        depends=[NodeDependency(start)],
        inference_mode=InferenceMode.STRUCTURED_OUTPUT,
    )
    stream = iter(
        [
            json.dumps(
                {
                    "thought": "record it",
                    "action": "memory-store",
                    "action_input": {"action": "write", "path": "prefs.md", "content": "Prefers tabs."},
                }
            ),
            json.dumps({"thought": "done", "action": "finish", "action_input": "Noted."}),
        ]
    )

    def run(**kwargs):
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": next(stream)}
        return result

    workflow = Workflow(flow=Flow(nodes=[start, agent]))
    with patch.object(agent.llm, "run", side_effect=run):
        result = workflow.run(input_data={"input": "I like tabs.", "user_id": "u-42"})

    assert result.status == RunnableStatus.SUCCESS
    assert seen["user_id"] == "u-42"


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
