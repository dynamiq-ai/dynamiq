from pathlib import Path

import pytest
from litellm import ModelResponse

from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import CancellationConfig, CancellationToken

DEFAULT_MAX_CHECKPOINTS = 10


@pytest.fixture
def cancellation_token():
    """A fresh CancellationToken for each test."""
    return CancellationToken()


@pytest.fixture
def cancellation_config(cancellation_token):
    return CancellationConfig(token=cancellation_token)


@pytest.fixture
def runnable_config(cancellation_config):
    """A RunnableConfig wired with a shared CancellationToken."""
    return RunnableConfig(cancellation=cancellation_config)


@pytest.fixture
def memory_backend():
    return InMemory()


@pytest.fixture
def filesystem_backend(tmp_path: Path):
    return FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))


@pytest.fixture
def backend_factory(tmp_path):
    def _create(backend_type: str):
        if backend_type == "in_memory":
            return InMemory()
        return FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))

    return _create


@pytest.fixture
def checkpoint_config(memory_backend):
    return CheckpointConfig(enabled=True, backend=memory_backend, max_checkpoints=DEFAULT_MAX_CHECKPOINTS)


def mock_llm_success(mocker, response: str = "mocked-response"):
    def side_effect(stream: bool, *args, **kwargs):
        r = ModelResponse()
        r["choices"][0]["message"]["content"] = response
        return r

    return mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)


def mock_llm_react_loop(mocker, *, tool_calls: int = 5, tool_name: str = "calculator", final_answer: str = "Done"):
    """Mock an agent's ReAct LLM. Returns ``call_count`` dict so test can assert call counts."""
    call_count = {"value": 0}

    def side_effect(stream: bool, *args, **kwargs):
        call_count["value"] += 1
        r = ModelResponse()
        if call_count["value"] <= tool_calls:
            r["choices"][0]["message"]["content"] = (
                f"Thought: Step {call_count['value']}.\n"
                f"Action: {tool_name}\n"
                f"Action Input: {{\"step\": {call_count['value']}}}"
            )
        else:
            r["choices"][0]["message"]["content"] = f"Thought: Done.\nAnswer: {final_answer}"
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count
