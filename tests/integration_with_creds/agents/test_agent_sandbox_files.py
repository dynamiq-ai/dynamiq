"""Integration tests for agent with sandbox file return (E2B_API_KEY + OPENAI_API_KEY required)."""

import os
from io import BytesIO

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.utils.logger import logger


@pytest.fixture(scope="module")
def openai_llm():
    return OpenAI(
        connection=OpenAIConnection(),
        model="gpt-4o-mini",
        max_tokens=3000,
        temperature=0.1,
    )


@pytest.fixture(scope="module")
def e2b_connection():
    return E2BConnection()


@pytest.fixture(scope="module")
def run_config():
    return RunnableConfig(request_timeout=120)


def _run_and_assert_sandbox_agent(
    agent, input_data, expected_keywords, run_config, expected_file_name: list[str] | None = None
):
    """Helper to run an agent with sandbox and validate results and returned files.

    Args:
        agent: The agent to run.
        input_data: Input data for the agent.
        expected_keywords: Keywords expected in the agent text output.
        run_config: Runnable configuration.
        expected_file_name: Optional list of file names to validate in output files.
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

        agent_output = result.output[agent.id]["output"]["content"]
        agent_output_files = result.output[agent.id]["output"].get("files", [])
        logger.info(f"Agent output: {agent_output}")
        logger.info(f"Agent output files: {len(agent_output_files) if agent_output_files else 0} file(s)")

    except Exception as e:
        pytest.fail(f"Agent run failed with exception: {e}")

    assert agent_output is not None, "Agent output should not be None"
    assert isinstance(agent_output, str), f"Agent output should be a string, got {type(agent_output)}"

    assert agent_output_files is not None, "Agent output files should not be None"
    assert isinstance(agent_output_files, list), f"Agent output files should be a list, got {type(agent_output_files)}"

    if expected_file_name:
        assert len(agent_output_files) >= len(expected_file_name), (
            f"Expected at least {len(expected_file_name)} file(s), got {len(agent_output_files)}"
        )

        returned_names = {f.name for f in agent_output_files}
        for name in expected_file_name:
            assert name in returned_names, (
                f"Expected file '{name}' not found in returned files: {returned_names}"
            )

        for returned_file in agent_output_files:
            assert isinstance(returned_file, BytesIO), (
                f"Agent should return file as BytesIO, got {type(returned_file)}"
            )
            file_content = returned_file.read()
            assert file_content is not None, "File content should not be None"
            assert len(file_content) > 0, f"File content for '{returned_file.name}' should not be empty"
            returned_file.seek(0)
            logger.info(f"Validated file: {returned_file.name} ({len(file_content)} bytes)")

    if expected_keywords:
        matches = [keyword for keyword in expected_keywords if keyword.lower() in agent_output.lower()]
        assert len(matches) > 0, f"Expected at least one of {expected_keywords} to be mentioned in the output"
        logger.info(f"Found keywords in response: {matches}")

    assert tracing.runs, "Expected tracing data to be collected"

    logger.info(f"--- Test Passed for Agent: {agent.name} ---")

    return agent_output


@pytest.mark.integration
def test_agent_sandbox_creates_and_returns_file(openai_llm, e2b_connection, run_config):
    """Test Agent with E2B sandbox can create a file and return it in output."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping credentials-required test.")
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY is not set; skipping credentials-required test.")

    sandbox_backend = E2BSandbox(connection=e2b_connection)
    try:
        sandbox_config = SandboxConfig(enabled=True, backend=sandbox_backend)

        agent = Agent(
            name="SandboxFileAgent",
            id="sandbox_file_agent",
            llm=openai_llm,
            role=(
                "You are a helpful assistant that can execute commands in the sandbox. "
                "When asked to create files, save them to /home/user/output so they are returned."
            ),
            inference_mode=InferenceMode.XML,
            sandbox=sandbox_config,
            max_loops=10,
            verbose=True,
        )

        input_data = {
            "input": (
                "Create a file called summary.txt in the /home/user/output directory "
                "with the text 'Hello from sandbox'. Then confirm the file was created."
            ),
        }

        _run_and_assert_sandbox_agent(
            agent,
            input_data,
            expected_keywords=["summary", "created", "file"],
            run_config=run_config,
            expected_file_name=["summary.txt"],
        )

        logger.info("--- Test Passed for Sandbox File Creation ---")
    finally:
        sandbox_backend.close()


@pytest.mark.integration
def test_agent_sandbox_with_input_files_returns_output(openai_llm, e2b_connection, run_config):
    """Test Agent with E2B sandbox receives input files and returns generated output files."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping credentials-required test.")
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY is not set; skipping credentials-required test.")

    sandbox_backend = E2BSandbox(connection=e2b_connection)
    try:
        sandbox_config = SandboxConfig(enabled=True, backend=sandbox_backend)

        agent = Agent(
            name="SandboxIOAgent",
            id="sandbox_io_agent",
            llm=openai_llm,
            role=(
                "You are a helpful assistant that can execute commands in the sandbox. "
                "When asked to create files, save them to /home/user/output so they are returned. "
                "For Python tasks, write a .py script file first, then run it with 'python script.py'. "
                "Never use Python one-liners with semicolons for multi-step logic."
            ),
            inference_mode=InferenceMode.XML,
            sandbox=sandbox_config,
            max_loops=10,
            verbose=True,
        )

        # Provide an input file
        input_file = BytesIO(b"name,score\nAlice,95\nBob,87\nCharlie,92\n")
        input_file.name = "scores.csv"
        input_file.seek(0)

        input_data = {
            "input": (
                "I've uploaded scores.csv. Read it from /home/user/scores.csv, "
                "calculate the average score using Python, "
                "and save the result to /home/user/output/average.txt. "
                "Write a Python script file and run it rather than using a one-liner."
            ),
            "files": [input_file],
        }

        _run_and_assert_sandbox_agent(
            agent,
            input_data,
            expected_keywords=["average", "score"],
            run_config=run_config,
            expected_file_name=["average.txt"],
        )

        logger.info("--- Test Passed for Sandbox Input/Output File Handling ---")
    finally:
        sandbox_backend.close()
