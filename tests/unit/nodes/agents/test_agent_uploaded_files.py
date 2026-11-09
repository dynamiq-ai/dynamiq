"""Files uploaded to an agent that has neither a sandbox nor a file store of its own."""

import io
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.storages.file.in_memory import InMemoryFileStore
from tests.helpers.agent_hooks import ALL_MODES, SearchTool, build_agent, encode, final_step, run_scripted, tool_step


def upload(name: str, data: bytes) -> io.BytesIO:
    file = io.BytesIO(data)
    file.name = name
    return file


def run_capturing_request_kwargs(agent, mode, steps, files):
    """Run once; return the result and the kwargs of every LLM call (tools, response_format...)."""
    calls: list[dict] = []

    def run(**kwargs):
        calls.append(kwargs)
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = encode(mode, steps[min(len(calls) - 1, len(steps) - 1)], len(calls) - 1)
        return result

    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": "Summarize the CSV.", "files": files})
    return result, calls


def test_function_calling_offers_the_file_tools_added_for_an_upload():
    agent = build_agent(tools=[SearchTool()], inference_mode=InferenceMode.FUNCTION_CALLING)
    steps = [tool_step("file-read", file_path="data.csv"), final_step("done")]

    result, calls = run_capturing_request_kwargs(
        agent, InferenceMode.FUNCTION_CALLING, steps, [upload("data.csv", b"a,b\n1,2\n")]
    )

    assert result.status == RunnableStatus.SUCCESS
    offered = {schema["function"]["name"] for schema in calls[0]["tools"]}
    assert {"file-read", "file-list", "file-search", "search", "provide_final_answer"} <= offered


def test_structured_output_offers_the_file_tools_added_for_an_upload():
    agent = build_agent(tools=[SearchTool()], inference_mode=InferenceMode.STRUCTURED_OUTPUT)
    steps = [tool_step("file-read", file_path="data.csv"), final_step("done")]

    result, calls = run_capturing_request_kwargs(
        agent, InferenceMode.STRUCTURED_OUTPUT, steps, [upload("data.csv", b"a,b\n1,2\n")]
    )

    assert result.status == RunnableStatus.SUCCESS
    action = calls[0]["response_format"]["json_schema"]["schema"]["properties"]["action"]
    assert "file-read" in action["description"]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_a_reused_agent_does_not_show_another_user_the_files_uploaded_earlier(mode):
    # One agent instance can serve many conversations; the second caller must not see the first one's files.
    agent = build_agent(tools=[SearchTool()])
    alice = {"user_id": "alice", "session_id": "s1"}
    bob = {"user_id": "bob", "session_id": "s2"}
    run_scripted(agent, mode, [final_step("ok")], input_extra={**alice, "files": [upload("salaries.csv", b"a,1\n")]})

    result, recorder = run_scripted(
        agent, mode, [tool_step("file-list"), final_step("ok")], input_extra={**bob, "files": [upload("b.csv", b"b\n")]}
    )

    assert result.status == RunnableStatus.SUCCESS
    listing = recorder.sent_text(1).rsplit("file-list", 1)[-1]
    assert "b.csv" in listing
    assert "salaries.csv" not in listing


@pytest.mark.parametrize("mode", ALL_MODES)
def test_a_later_turn_of_the_same_conversation_still_reads_its_uploads(mode):
    # The conversation's memory says the file is attached, so the next turn must still find it.
    agent = build_agent(tools=[SearchTool()])
    conversation = {"user_id": "alice", "session_id": "s1"}
    run_scripted(
        agent, mode, [final_step("ok")], input_extra={**conversation, "files": [upload("sales.csv", b"q3,42\n")]}
    )

    result, recorder = run_scripted(
        agent, mode, [tool_step("file-read", file_path="sales.csv"), final_step("ok")], input_extra=conversation
    )

    assert result.status == RunnableStatus.SUCCESS
    assert "q3,42" in recorder.sent_text(1)


def test_restoring_a_checkpoint_replaces_the_files_an_earlier_run_uploaded():
    agent = build_agent(tools=[SearchTool()])
    run_scripted(
        agent,
        InferenceMode.FUNCTION_CALLING,
        [final_step("ok")],
        input_extra={"user_id": "alice", "files": [upload("salaries.csv", b"a,1\n")]},
    )
    paused_run_files = InMemoryFileStore()
    paused_run_files.store("b.csv", b"b\n")

    agent._restore_file_store_state(paused_run_files.to_checkpoint_state(max_bytes=1_000_000))

    assert [info.path for info in agent.file_store_backend.list_files(recursive=True)] == ["b.csv"]
