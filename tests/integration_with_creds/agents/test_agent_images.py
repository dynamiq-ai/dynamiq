import os

import pytest

from dynamiq import ROOT_PATH, Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import Gemini, OpenAI
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


# Path fixtures
@pytest.fixture(scope="module")
def data_folder_path():
    return os.path.join(os.path.dirname(ROOT_PATH), "examples", "components", "data")


@pytest.fixture(scope="module")
def image_file_path(data_folder_path):
    return os.path.join(data_folder_path, "img.jpeg")


@pytest.fixture(scope="module")
def image_bytes(image_file_path):
    with open(image_file_path, "rb") as f:
        return f.read()


# Test configuration fixtures
@pytest.fixture(scope="module")
def agent_role():
    return "helpful assistant that accurately analyzes images and provides technical details"


@pytest.fixture(scope="module")
def camera_query():
    return "Look at this image and tell me what camera manufacturer is likely used to take this photo."


@pytest.fixture(scope="module")
def expected_camera_keywords():
    return ["canon", "camera", "manufacturer", "dslr", "photography", "equipment"]


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
        model="gpt-4o",
        max_tokens=1000,
        temperature=0,
    )


@pytest.fixture(scope="module")
def gemini_llm(gemini_connection):
    return Gemini(
        connection=gemini_connection,
        model="gemini-2.0-flash",
        max_tokens=1000,
        temperature=0,
    )


# Helper functions
def _run_and_assert_agent(agent, input_data, expected_keywords, run_config):
    """Generic helper function to run an agent and check results."""
    logger.info(f"\n--- Running Agent: {agent.name} ---")

    workflow = Workflow(flow=Flow(nodes=[agent]))
    tracing = TracingCallbackHandler()
    config = run_config.model_copy(update={"callbacks": [tracing]})

    try:
        result = workflow.run(input_data=input_data, config=config)

        logger.info(f"Agent run completed with status: {result.status}")

        if result.status != RunnableStatus.SUCCESS:
            intermediate_steps = (
                result.output.get("intermediate_steps", "N/A") if isinstance(result.output, dict) else "N/A"
            )
            logger.info(f"Intermediate Steps on Failure: {intermediate_steps}")
            pytest.fail(f"Agent run failed with status '{result.status}'. Output: {result.output}.")

        agent_output = result.output[agent.id]["output"]["content"]
        logger.info(f"Agent output: {agent_output}")

    except Exception as e:
        pytest.fail(f"Agent run failed with exception: {e}")

    assert agent_output is not None, "Agent output should not be None"
    assert isinstance(agent_output, str), f"Agent output should be a string, got {type(agent_output)}"

    if expected_keywords:
        matches = [keyword for keyword in expected_keywords if keyword.lower() in agent_output.lower()]
        assert len(matches) > 0, f"Expected at least one of {expected_keywords} to be mentioned in the output"
        logger.info(f"Found keywords in response: {matches}")

    assert tracing.runs, "Expected tracing data to be collected"

    logger.info(f"--- Test Passed for Agent: {agent.name} ---")

    return agent_output


@pytest.mark.integration
@pytest.mark.parametrize(
    "inference_mode",
    [
        InferenceMode.DEFAULT,
        InferenceMode.XML,
    ],
)
def test_react_agent_inference_modes(
    openai_llm, image_bytes, camera_query, expected_camera_keywords, agent_role, run_config, inference_mode
):
    """Test ReActAgent with all inference modes."""
    with get_connection_manager():
        agent = ReActAgent(
            name=f"CameraDetection_{inference_mode.value}",
            id=f"camera_detection_{inference_mode.value.lower()}",
            llm=openai_llm,
            role=agent_role,
            inference_mode=inference_mode,
            tools=[],
            verbose=True,
        )

        input_data = {"input": camera_query, "images": [image_bytes]}
        _run_and_assert_agent(agent, input_data, expected_camera_keywords, run_config)


@pytest.mark.integration
def test_simple_agent_with_text_and_image(
    gemini_llm, image_bytes, camera_query, expected_camera_keywords, agent_role, run_config
):
    """Test SimpleAgent with both a text query and image."""
    with get_connection_manager():
        agent = SimpleAgent(
            name="SimpleAgent_TextAndImage",
            id="simple_agent_text_and_image",
            llm=gemini_llm,
            role=agent_role,
        )

        input_data = {"input": camera_query, "images": [image_bytes]}
        _run_and_assert_agent(agent, input_data, expected_camera_keywords, run_config)


@pytest.mark.integration
def test_simple_agent_with_image_only(gemini_llm, image_bytes, agent_role, run_config):
    """Test SimpleAgent with just an image and no specific query."""
    with get_connection_manager():
        agent = SimpleAgent(
            name="SimpleAgent_ImageOnly",
            id="simple_agent_image_only",
            llm=gemini_llm,
            role=agent_role,
        )

        input_data = {"input": "", "images": [image_bytes]}

        _run_and_assert_agent(agent, input_data, None, run_config)
