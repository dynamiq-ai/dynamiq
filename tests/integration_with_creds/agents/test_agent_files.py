import os
from io import BytesIO

import pytest

from dynamiq import ROOT_PATH, Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Gemini, OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.file import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore
from dynamiq.utils.logger import logger


# Path fixtures
@pytest.fixture(scope="module")
def data_folder_path():
    return os.path.join(os.path.dirname(ROOT_PATH), "examples", "components", "data")


@pytest.fixture(scope="module")
def image_file_path(data_folder_path):
    return os.path.join(data_folder_path, "img.jpeg")


@pytest.fixture(scope="function")
def image_bytes(image_file_path):
    """Load image and return as BytesIO file object.

    Uses function scope to ensure each test gets a fresh BytesIO with stream position at 0.
    This prevents issues when multiple LLM fixtures are parametrized and the stream is consumed.
    """
    with open(image_file_path, "rb") as f:
        image_data = f.read()

    image_file = BytesIO(image_data)
    image_file.name = "img.jpeg"
    image_file.seek(0)
    return image_file


@pytest.fixture(scope="function")
def image_bytes_2(image_file_path):
    """Load the same image as a second file with a different name."""
    with open(image_file_path, "rb") as f:
        image_data = f.read()

    image_file = BytesIO(image_data)
    image_file.name = "img2.jpeg"
    image_file.seek(0)
    return image_file


# Test configuration fixtures
@pytest.fixture(scope="module")
def agent_role():
    return "helpful assistant that describes images and saves the description to a file when asked."


@pytest.fixture(scope="module")
def camera_query():
    return (
        "I gave you two photos. Save a short description of the first photo to summary1.txt"
        " and a short description of the second photo to summary2.txt. Return both files."
    )


@pytest.fixture(scope="module")
def expected_camera_keywords():
    return ["image", "photo", "camera", "description", "summary"]


@pytest.fixture(scope="module")
def run_config():
    return RunnableConfig(request_timeout=120)


# Connection fixtures
@pytest.fixture(scope="module")
def openai_connection():
    return OpenAIConnection()


@pytest.fixture(scope="module")
def gemini_connection():
    return GeminiConnection()


# LLM fixtures
@pytest.fixture(scope="module")
def openai_llm(openai_connection):
    return OpenAI(
        connection=openai_connection,
        model="gpt-4o-mini",
        max_tokens=3000,
        temperature=0.1,
    )


@pytest.fixture(scope="module")
def gemini_llm(gemini_connection):
    return Gemini(
        connection=gemini_connection,
        model="gemini-3.0-flash-exp",
        max_tokens=3000,
        temperature=0.1,
    )


def _run_and_assert_agent(agent, input_data, expected_keywords, run_config, expected_file_names=None):
    """Generic helper function to run an agent and check results.

    Args:
        agent: The agent to run
        input_data: Input data for the agent
        expected_keywords: Keywords expected in the agent output
        run_config: Runnable configuration
        expected_file_names: Optional list of file names expected in agent output files
    """
    logger.info(f"\n--- Running Agent: {agent.name} ---")

    workflow = Workflow(flow=Flow(nodes=[agent]))
    tracing = TracingCallbackHandler()

    config = run_config.model_copy(update={"callbacks": [tracing]})

    try:
        result = workflow.run(input_data=input_data, config=config)
        logger.info(f"Agent run completed with status: {result.status}")

        if result.status != RunnableStatus.SUCCESS:
            pytest.fail(f"Agent run failed with status '{result.status}'. Output: {result.output}.")

        logger.info(f"Agent output: {result.output[agent.id]['output']}")
        agent_output = result.output[agent.id]["output"]["content"]
        agent_output_files = result.output[agent.id]["output"].get("files", [])
        logger.info(f"files: {len(agent.file_store.backend._files)}")
        logger.info(f"Agent output: {agent_output}")
        logger.info(f"Agent output files: {len(agent_output_files) if agent_output_files else 0} file(s)")

    except Exception as e:
        pytest.fail(f"Agent run failed with exception: {e}")

    assert agent_output is not None, "Agent output should not be None"
    assert isinstance(agent_output, str), f"Agent output should be a string, got {type(agent_output)}"

    assert agent_output_files is not None, "Agent output files should not be None"
    assert isinstance(agent_output_files, list), f"Agent output files should be a list, got {type(agent_output_files)}"

    if expected_file_names:
        assert len(agent_output_files) >= len(expected_file_names), (
            f"Expected at least {len(expected_file_names)} file(s), got {len(agent_output_files)}"
        )
        returned_names = {f.name for f in agent_output_files}
        for name in expected_file_names:
            assert name in returned_names, (
                f"Expected file '{name}' not found in returned files: {returned_names}"
            )

        for returned_file in agent_output_files:
            assert isinstance(returned_file, BytesIO), (
                f"Agent should return file as BytesIO, got {type(returned_file)}"
            )
            file_content = returned_file.read()
            assert file_content is not None, "File content should not be None"
            assert len(file_content) > 0, "File content should not be empty"
            returned_file.seek(0)
            logger.info(f"Validated file: {returned_file.name} ({len(file_content)} bytes)")

    if expected_keywords:
        matches = [keyword for keyword in expected_keywords if keyword.lower() in agent_output.lower()]
        assert len(matches) > 0, f"Expected at least one of {expected_keywords} to be mentioned in the output"
        logger.info(f"Found keywords in response: {matches}")

    assert tracing.runs, "Expected tracing data to be collected"

    logger.info(f"--- Test Passed for Agent: {agent.name} ---")

    return agent_output


@pytest.mark.flaky(reruns=3)
@pytest.mark.integration
@pytest.mark.parametrize(
    "llm_fixture, inference_mode",
    [
        ("openai_llm", InferenceMode.XML),
        ("openai_llm", InferenceMode.DEFAULT),
        # ("openai_llm", InferenceMode.FUNCTION_CALLING),
        ("openai_llm", InferenceMode.STRUCTURED_OUTPUT),
    ],
)
def test_agent_filestore_multiple_files(
    llm_fixture,
    inference_mode,
    image_bytes,
    image_bytes_2,
    agent_role,
    run_config,
    camera_query,
    expected_camera_keywords,
    request,
):
    """Test Agent can create multiple files in FileStore and return them in output."""
    llm = request.getfixturevalue(llm_fixture)

    file_store_backend = InMemoryFileStore()
    file_store_config = FileStoreConfig(
        enabled=True,
        backend=file_store_backend,
        agent_file_write_enabled=True,
    )

    agent = Agent(
        name=f"MultiFileAgent_{llm_fixture}_{inference_mode.value}",
        id=f"multi_file_agent_{llm_fixture}_{inference_mode.value}",
        llm=llm,
        role=agent_role,
        inference_mode=inference_mode,
        file_store=file_store_config,
        tools=[],
        verbose=True,
    )

    input_data = {
        "input": camera_query,
        "files": [image_bytes, image_bytes_2],
    }

    _run_and_assert_agent(
        agent, input_data, expected_camera_keywords, run_config,
        expected_file_names=["summary1.txt", "summary2.txt"],
    )

    logger.info(f"--- Test Passed for Multiple File Creation with {llm_fixture} ({inference_mode.value}) ---")
