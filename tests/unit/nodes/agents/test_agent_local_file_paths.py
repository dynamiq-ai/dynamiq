"""An agent and its tools read files by local path only where the environment allows it."""

import base64

import pytest

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.converters import TextFileConverter
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.local_files import LOCAL_FILE_PATHS_ALLOWED_ENV_VAR
from tests.helpers.agent_hooks import build_agent, final_step, run_scripted, tool_step

CONTENT = "Quarterly figures: 4f1c9e"
PNG = b"\x89PNG\r\n\x1a\n" + CONTENT.encode()


@pytest.fixture
def local_file_paths_off(monkeypatch):
    monkeypatch.setenv(LOCAL_FILE_PATHS_ALLOWED_ENV_VAR, "false")


@pytest.fixture
def local_file_paths_on(monkeypatch):
    monkeypatch.delenv(LOCAL_FILE_PATHS_ALLOWED_ENV_VAR, raising=False)


@pytest.fixture
def notes_path(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(CONTENT)
    return str(path)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "chart.png"
    path.write_bytes(PNG)
    return str(path)


def run_with_converter_tool(path):
    """Run an agent whose model asks a converter tool for ``path``. Returns everything the run let out: its output,
    what the model was sent and the traced runs, where the converter's documents are recorded."""
    agent = build_agent(tools=[TextFileConverter(name="read-notes")])
    tracing = TracingCallbackHandler()
    steps = [tool_step("read-notes", file_paths=[path]), final_step("done")]

    result, recorder = run_scripted(
        agent, InferenceMode.XML, steps, question="Read the notes.", config=RunnableConfig(callbacks=[tracing])
    )

    traced = str([run.to_dict() for run in tracing.runs.values()])
    return "\n".join([str(result.output), recorder.sent_text(), traced])


def test_converter_tool_refuses_a_path_the_model_gives_when_local_paths_are_turned_off(
    local_file_paths_off, notes_path
):
    let_out = run_with_converter_tool(notes_path)

    assert LOCAL_FILE_PATHS_ALLOWED_ENV_VAR in let_out
    assert CONTENT not in let_out


def test_converter_tool_reads_a_path_the_model_gives_when_the_environment_does_not_turn_them_off(
    local_file_paths_on, notes_path
):
    assert CONTENT in run_with_converter_tool(notes_path)


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-3.5-turbo"], ids=["vision", "no-vision"])
def test_agent_does_not_read_an_image_path_when_local_paths_are_turned_off(local_file_paths_off, image_path, model):
    agent = build_agent(tools=[])
    agent.llm = OpenAI(connection=agent.llm.connection, model=model, max_tokens=100, temperature=0)

    _, recorder = run_scripted(
        agent,
        InferenceMode.XML,
        [final_step("done")],
        question="Describe the chart.",
        input_extra={"images": [image_path]},
    )

    sent = "\n".join(str(message) for prompt in recorder.prompts for message in prompt)
    assert base64.b64encode(PNG).decode() not in sent
    assert CONTENT not in sent
    assert agent.file_store_backend is None or not agent.file_store_backend.list_files()


def test_agent_reads_an_image_path_when_the_environment_does_not_turn_them_off(local_file_paths_on, image_path):
    agent = build_agent(tools=[])

    _, recorder = run_scripted(
        agent,
        InferenceMode.XML,
        [final_step("done")],
        question="Describe the chart.",
        input_extra={"images": [image_path]},
    )

    sent = "\n".join(str(message) for prompt in recorder.prompts for message in prompt)
    assert base64.b64encode(PNG).decode() in sent
