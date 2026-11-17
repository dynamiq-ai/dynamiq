"""Files the agent names in its final answer, returned with the run output."""

import json
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.storages.file.base import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore
from tests.helpers.agent_hooks import SearchTool, build_agent


def final_answer(mode: InferenceMode, output_files) -> dict:
    if mode == InferenceMode.FUNCTION_CALLING:
        arguments = {"thought": "done", "answer": "Report attached.", "output_files": output_files}
        call = {"name": "provide_final_answer", "arguments": json.dumps(arguments)}
        return {"content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": call}]}
    if mode == InferenceMode.XML:
        return {
            "content": "<output><thought>done</thought><answer>Report attached.</answer>"
            f"<output_files>{output_files}</output_files></output>"
        }
    return {
        "content": json.dumps(
            {"thought": "done", "action": "finish", "action_input": "Report attached.", "output_files": output_files}
        )
    }


def run_with_final_answer(mode: InferenceMode, output_files):
    store = InMemoryFileStore()
    store.store("out/report.md", b"# Report\n")
    store.store("out/data.csv", b"a,b\n")
    agent = build_agent(tools=[SearchTool()], file_store=FileStoreConfig(enabled=True, backend=store))
    agent.inference_mode = mode
    agent._init_prompt_blocks()

    def run(**kwargs):
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = final_answer(mode, output_files)
        return result

    with patch.object(agent.llm, "run", side_effect=run):
        return agent.run(input_data={"input": "Write the report."})


@pytest.mark.parametrize("mode", [InferenceMode.XML, InferenceMode.FUNCTION_CALLING])
@pytest.mark.parametrize("path", ["/out/report.md", "./out/report.md", "out/report.md"])
def test_a_file_store_file_is_returned_whether_named_absolute_or_relative(mode, path):
    # The prompt asks for absolute paths, while file store keys are relative.
    result = run_with_final_answer(mode, path)

    assert result.status == RunnableStatus.SUCCESS
    assert [getattr(f, "name", "") for f in result.output.get("files", [])] == ["out/report.md"]


@pytest.mark.parametrize("mode", [InferenceMode.FUNCTION_CALLING, InferenceMode.STRUCTURED_OUTPUT])
def test_output_files_sent_as_a_list_are_returned(mode):
    result = run_with_final_answer(mode, ["out/report.md", "out/data.csv"])

    assert result.status == RunnableStatus.SUCCESS
    assert sorted(getattr(f, "name", "") for f in result.output.get("files", [])) == ["out/data.csv", "out/report.md"]
