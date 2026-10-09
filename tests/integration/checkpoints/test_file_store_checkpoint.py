"""Integration tests for in-memory file store contents in agent checkpoints.

Verifies that the files in an agent's in-memory file store survive a checkpoint
round-trip, as a sandbox's files do by reconnecting:

- the files are captured in ``AgentCheckpointState.file_store_state``
- the state survives JSON serialization and the InMemory / FileSystem backends
- an agent restored from that checkpoint, with a new empty store, has the files
  again, and its file tools read them
- an agent that writes a file, then waits on the user past the input timeout,
  returns the file after a new process resumes the run with the answer, with
  and without a configured store
- files the agent's tools returned are carried whether or not it may write files
- an empty store and files over the budget are left out
- files the agent got from its input are left out, so a resumed run that stores
  them again keeps one copy
- an agent whose store is created only when input files arrive gets it back on
  restore, with its file tools
"""

import io
import json
from queue import Queue
from typing import Any, ClassVar, Literal

import pytest
from litellm import ModelResponse
from pydantic import BaseModel

from dynamiq import connections, flows
from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.checkpoint import CheckpointStatus
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.nodes import NodeGroup, llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents import checkpoint as agent_checkpoint
from dynamiq.nodes.agents.checkpoint import AgentCheckpointState
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.file_tools import FileListTool
from dynamiq.nodes.tools.human_feedback import (
    HFStreamingInputEventMessage,
    HFStreamingInputEventMessageData,
    HumanFeedbackAction,
    HumanFeedbackTool,
)
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.storages.file.base import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.types.streaming import StreamingConfig

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"
FLOW_ID = "file-store-checkpoint-flow"
AGENT_ID = "file-store-agent"
REPORT_PATH = "report.md"
REPORT_CONTENT = b"# Report\nwritten before checkpoint"
IMAGE_PATH = "charts/chart.png"
IMAGE_CONTENT = bytes(range(256))
INPUT_PATH = "data.csv"
INPUT_CONTENT = b"a,b\n1,2\n"


def make_llm(node_id: str) -> llms.OpenAI:
    return llms.OpenAI(
        id=node_id,
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        is_postponed_component_init=True,
    )


def make_file_store_agent(agent_id: str = AGENT_ID, writable: bool = True, tools: list | None = None) -> Agent:
    return Agent(
        id=agent_id,
        name="File Store Agent",
        llm=make_llm(f"{agent_id}-llm"),
        tools=tools or [],
        role="Test",
        max_loops=5,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=writable),
    )


def write_files(agent: Agent) -> None:
    agent.file_store_backend.store(REPORT_PATH, REPORT_CONTENT, metadata={"description": "Quarterly report"})
    agent.file_store_backend.store(IMAGE_PATH, IMAGE_CONTENT, content_type="image/png")


FILE_TOOL_NAMES = {"file-read", "file-search", "file-list"}


class ChartSchema(BaseModel):
    model_config = {"extra": "allow"}


class ChartTool(Node):
    """Returns a chart as a file, as tools such as the code interpreter do."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "chart"
    description: str = "Draws a chart."
    input_schema: ClassVar[type[ChartSchema]] = ChartSchema

    def execute(self, input_data: ChartSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        return {"content": "Chart drawn.", "files": [chart_file()]}


def chart_file() -> io.BytesIO:
    file = io.BytesIO(IMAGE_CONTENT)
    file.name = IMAGE_PATH
    return file


def store_chart_from_tool(agent: Agent) -> None:
    agent._handle_tool_generated_files(
        ChartTool(), RunnableResult(status=RunnableStatus.SUCCESS, output={"content": "", "files": [chart_file()]})
    )


def make_agent_without_store(agent_id: str = AGENT_ID, tools: list | None = None) -> Agent:
    return Agent(id=agent_id, name="Agent", llm=make_llm(f"{agent_id}-llm"), tools=tools or [], role="Test")


def input_file() -> io.BytesIO:
    file = io.BytesIO(INPUT_CONTENT)
    file.name = INPUT_PATH
    return file


def stored_paths(agent: Agent) -> list[str]:
    return sorted(info.path for info in agent.file_store_backend.list_files(recursive=True))


def make_flow(backend, agent: Agent) -> flows.Flow:
    return flows.Flow(
        id=FLOW_ID,
        nodes=[agent],
        checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_mid_agent_loop_enabled=True),
    )


def mock_final_answer(mocker, answer: str = "Done."):
    """Mock the LLM so the agent answers immediately without calling any tool."""

    def side_effect(stream: bool, *args, **kwargs):
        r = ModelResponse()
        r["choices"][0]["message"]["content"] = f"Thought: Done.\nFinal Answer: {answer}"
        return r

    return mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)


@pytest.fixture
def backend_factory(tmp_path):
    def _create(backend_type: str):
        if backend_type == "in_memory":
            return InMemory()
        return FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))

    return _create


class TestFileStoreCheckpointRoundTrip:
    """In-memory file store contents survive checkpoint save/restore."""

    def test_written_files_are_captured_and_restored(self):
        agent_a = make_file_store_agent("agent-a")
        write_files(agent_a)

        state_dict = json.loads(json.dumps(agent_a.to_checkpoint_state().model_dump()))
        assert set(state_dict["file_store_state"]["files"]) == {REPORT_PATH, IMAGE_PATH}

        agent_b = make_file_store_agent("agent-b")
        assert agent_b.file_store_backend.is_empty()

        agent_b.from_checkpoint_state(state_dict)

        store = agent_b.file_store_backend
        assert store.retrieve(REPORT_PATH) == REPORT_CONTENT
        assert store.retrieve(IMAGE_PATH) == IMAGE_CONTENT
        report, image = store.list_files_bytes([REPORT_PATH, IMAGE_PATH])
        assert report.description == "Quarterly report"
        assert image.content_type == "image/png"
        list_tool = next(tool for tool in agent_b.tools if isinstance(tool, FileListTool))
        assert list_tool.file_store is store

    def test_empty_store_is_not_captured(self):
        assert make_file_store_agent().to_checkpoint_state().file_store_state is None

    def test_files_from_tools_are_captured_without_the_write_flags(self):
        agent = make_file_store_agent(writable=False)
        store_chart_from_tool(agent)
        assert set(agent.to_checkpoint_state().file_store_state["files"]) == {IMAGE_PATH}

    def test_input_files_are_left_out_and_stored_once_after_restore(self):
        agent_a = make_file_store_agent("agent-a")
        assert agent_a._upload_files_to_file_store([input_file()]) == [INPUT_PATH]
        agent_a.file_store_backend.store(REPORT_PATH, REPORT_CONTENT)

        state_dict = json.loads(json.dumps(agent_a.to_checkpoint_state().model_dump()))
        assert set(state_dict["file_store_state"]["files"]) == {REPORT_PATH}

        agent_b = make_file_store_agent("agent-b")
        agent_b.from_checkpoint_state(state_dict)
        assert agent_b._upload_files_to_file_store([input_file()]) == [INPUT_PATH]
        assert stored_paths(agent_b) == [INPUT_PATH, REPORT_PATH]

    def test_store_with_only_input_files_is_not_captured(self):
        agent = make_file_store_agent()
        agent._upload_files_to_file_store([input_file()])
        assert agent.to_checkpoint_state().file_store_state is None

    def test_files_over_the_budget_are_left_out(self, monkeypatch):
        monkeypatch.setattr(agent_checkpoint, "MAX_CHECKPOINT_FILE_STORE_BYTES", 300)
        agent = make_file_store_agent()
        agent.file_store_backend.store("big.bin", b"x" * 301)
        write_files(agent)  # 33 + 256 bytes, which fit together.

        files = agent.to_checkpoint_state().file_store_state["files"]
        assert set(files) == {REPORT_PATH, IMAGE_PATH}

    def test_nothing_is_captured_when_no_file_fits(self, monkeypatch):
        monkeypatch.setattr(agent_checkpoint, "MAX_CHECKPOINT_FILE_STORE_BYTES", 10)
        agent = make_file_store_agent()
        write_files(agent)
        assert agent.to_checkpoint_state().file_store_state is None

    def test_restore_creates_the_store_an_agent_makes_for_input_files(self):
        agent_a = make_agent_without_store("agent-a")
        agent_a._setup_in_memory_file_store_and_tools()
        agent_a._upload_files_to_file_store([input_file()])
        store_chart_from_tool(agent_a)
        state_dict = json.loads(json.dumps(agent_a.to_checkpoint_state().model_dump()))
        assert set(state_dict["file_store_state"]["files"]) == {IMAGE_PATH}

        agent_b = make_agent_without_store("agent-b")
        assert agent_b.file_store_backend is None
        agent_b.from_checkpoint_state(state_dict)

        assert agent_b.file_store_backend.retrieve(IMAGE_PATH) == IMAGE_CONTENT
        tool_names = [tool.name for tool in agent_b.tools]
        assert FILE_TOOL_NAMES <= set(tool_names)
        assert len(tool_names) == len(set(tool_names))

    def test_restore_is_skipped_for_an_agent_with_a_sandbox(self):
        agent_a = make_file_store_agent("agent-a")
        write_files(agent_a)
        state_dict = agent_a.to_checkpoint_state().model_dump()

        agent_b = Agent(
            id="agent-b",
            name="Sandbox Agent",
            llm=make_llm("agent-b-llm"),
            role="Test",
            sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=connections.E2B(api_key=TEST_API_KEY))),
        )
        agent_b.from_checkpoint_state(state_dict)
        assert agent_b.file_store_backend is None

    def test_old_checkpoint_without_file_store_state_is_backward_compatible(self):
        agent = make_file_store_agent()
        agent.from_checkpoint_state({"history_offset": 3})
        assert agent._history_offset == 3
        assert agent.file_store_backend.is_empty()


class TestFileStoreCheckpointInFlow:
    """In-memory file store contents flow through a real Flow checkpoint and resume."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_resumed_flow_has_the_files_written_before_the_checkpoint(self, mocker, backend_factory, backend_type):
        backend = backend_factory(backend_type)

        agent1 = make_file_store_agent()
        write_files(agent1)
        flow1 = make_flow(backend, agent1)
        mock_final_answer(mocker, "First run")
        assert flow1.run_sync(input_data={"input": "hello"}).status == RunnableStatus.SUCCESS
        mocker.stopall()

        cp = backend.get_latest_by_flow(flow1.id)
        assert set(cp.node_states[AGENT_ID].internal_state["file_store_state"]["files"]) == {REPORT_PATH, IMAGE_PATH}
        cp.node_states[AGENT_ID].status = CheckpointStatus.ACTIVE.value
        cp.node_states[AGENT_ID].output_data = None
        cp.completed_node_ids = [nid for nid in cp.completed_node_ids if nid != AGENT_ID]
        cp.status = CheckpointStatus.ACTIVE
        backend.save(cp)

        agent2 = make_file_store_agent()
        assert agent2.file_store_backend.is_empty()
        flow2 = make_flow(backend, agent2)
        mock_final_answer(mocker, "Resumed")

        assert flow2.run_sync(input_data=None, resume_from=cp.id).status == RunnableStatus.SUCCESS

        assert agent2.file_store_backend.retrieve(REPORT_PATH) == REPORT_CONTENT
        assert agent2.file_store_backend.retrieve(IMAGE_PATH) == IMAGE_CONTENT


class TestFileStoreSurvivesInputTimeout:
    """A question to the user outlives the input timeout, where the Dynamiq runtime checkpoints a run.

    Checkpointing is configured as the runtime does, on input timeout only, and the resume runs
    in a fresh agent and flow, as a new process would.
    """

    ASK_TOOL_ID = "ask-user"
    STREAMING_TIMEOUT = 0.3

    def _ask_user(self, answers: Queue) -> HumanFeedbackTool:
        return HumanFeedbackTool(
            id=self.ASK_TOOL_ID,
            name="ask-user",
            action=HumanFeedbackAction.ASK,
            input_method=FeedbackMethod.STREAM,
            output_method=FeedbackMethod.STREAM,
            streaming=StreamingConfig(enabled=True, input_queue=answers, timeout=self.STREAMING_TIMEOUT),
        )

    def _make_agent(self, answers: Queue) -> Agent:
        return make_file_store_agent(tools=[self._ask_user(answers)])

    def _answered(self) -> Queue:
        answers = Queue()
        answers.put(
            HFStreamingInputEventMessage(
                entity_id=self.ASK_TOOL_ID, data=HFStreamingInputEventMessageData(content="Yes")
            ).model_dump_json()
        )
        return answers

    def _make_flow(self, backend, agent: Agent) -> flows.Flow:
        return flows.Flow(
            id=FLOW_ID,
            nodes=[agent],
            checkpoint=CheckpointConfig(
                enabled=True,
                backend=backend,
                checkpoint_after_node_enabled=False,
                checkpoint_on_failure_enabled=False,
                checkpoint_on_cancel_enabled=False,
                checkpoint_mid_agent_loop_enabled=False,
                checkpoint_on_input_timeout_enabled=True,
            ),
        )

    def _mock_llm(self, mocker, first_action: str, output_file: str):
        """The LLM takes the first action, asks the user, then answers with the given output file.

        The question is replayed from the checkpoint on resume, without an LLM call.
        """
        replies = iter(
            [
                first_action,
                'Thought: Check with the user.\nAction: ask-user\nAction Input: {"input": "Send it as is?"}',
                f"Thought: The user agreed.\nOutput Files: {output_file}\nAnswer: The file is attached.",
            ]
        )

        def side_effect(stream: bool, *args, **kwargs):
            r = ModelResponse()
            r["choices"][0]["message"]["content"] = next(replies)
            return r

        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)

    def test_file_written_before_the_question_is_returned_after_resume(self, mocker, tmp_path):
        backend = FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))
        write_report = (
            "Thought: Write the report first.\nAction: file-write\n"
            f"Action Input: {json.dumps({'file_path': REPORT_PATH, 'content': REPORT_CONTENT.decode()})}"
        )
        self._mock_llm(mocker, write_report, REPORT_PATH)

        first = self._make_flow(backend, self._make_agent(Queue())).run_sync(
            input_data={"input": "Draft the report", "files": [input_file()]}
        )
        assert first.status == RunnableStatus.FAILURE
        saved = backend.get_latest_by_flow(FLOW_ID)
        assert set(saved.node_states[AGENT_ID].internal_state["file_store_state"]["files"]) == {REPORT_PATH}

        resumed_agent = self._make_agent(self._answered())
        second = self._make_flow(backend, resumed_agent).run_sync(
            input_data={"input": "Draft the report", "files": [input_file()]}, resume_from=saved.id
        )

        assert second.status == RunnableStatus.SUCCESS
        files = second.output[AGENT_ID]["output"]["files"]
        assert [(f.name, f.getvalue()) for f in files] == [(REPORT_PATH, REPORT_CONTENT)]
        assert stored_paths(resumed_agent) == [INPUT_PATH, REPORT_PATH]

    def test_file_from_a_tool_is_returned_after_resume_without_a_configured_store(self, mocker, tmp_path):
        """The agent creates its store when the input file arrives, and only then can keep a tool's file."""
        backend = FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))
        self._mock_llm(mocker, 'Thought: Draw it.\nAction: chart\nAction Input: {"data": "data.csv"}', IMAGE_PATH)

        def agent(answers: Queue) -> Agent:
            return make_agent_without_store(tools=[ChartTool(id="chart"), self._ask_user(answers)])

        first = self._make_flow(backend, agent(Queue())).run_sync(
            input_data={"input": "Chart the data", "files": [input_file()]}
        )
        assert first.status == RunnableStatus.FAILURE
        saved = backend.get_latest_by_flow(FLOW_ID)
        assert set(saved.node_states[AGENT_ID].internal_state["file_store_state"]["files"]) == {IMAGE_PATH}

        resumed_agent = agent(self._answered())
        second = self._make_flow(backend, resumed_agent).run_sync(
            input_data={"input": "Chart the data", "files": [input_file()]}, resume_from=saved.id
        )

        assert second.status == RunnableStatus.SUCCESS
        files = second.output[AGENT_ID]["output"]["files"]
        assert [(f.name, f.getvalue()) for f in files] == [(IMAGE_PATH, IMAGE_CONTENT)]
        assert stored_paths(resumed_agent) == [IMAGE_PATH, INPUT_PATH]
        tool_names = [tool.name for tool in resumed_agent.tools]
        assert len(tool_names) == len(set(tool_names))


class TestFileStoreCheckpointStateModel:
    def test_state_model_defaults_file_store_state_to_none(self):
        assert AgentCheckpointState().file_store_state is None
