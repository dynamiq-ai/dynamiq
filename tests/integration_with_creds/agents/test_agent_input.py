import pytest

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus


@pytest.mark.unit
def test_react_agent_without_tools_or_memory():
    connection = OpenAIConnection()
    llm = OpenAI(
        connection=connection,
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0,
    )

    agent = Agent(
        name="TestReactAgent",
        id="test_react_agent",
        llm=llm,
        role="You are a helpful assistant.",
        inference_mode=InferenceMode.DEFAULT,
        tools=[],
        memory=None,
        verbose=False,
        max_loops=3,
    )

    input_data = {
        "input": "What is 2 + 2?",
        "user_id": None,
        "session_id": None,
    }

    config = RunnableConfig(request_timeout=30)
    result = agent.run(input_data=input_data, config=config)

    assert result.status == RunnableStatus.SUCCESS

    content = result.output["content"]
    assert isinstance(content, str)
    assert "4" in content, f"Expected '4' in the output, got: {content!r}"


@pytest.fixture(scope="module")
def openai_connection():
    """Provides a reusable OpenAI connection."""
    return OpenAIConnection()


@pytest.fixture(scope="module")
def test_llm(openai_connection):
    """Provides a lightweight LLM instance for testing."""
    return OpenAI(
        connection=openai_connection,
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0,
    )


@pytest.fixture
def test_react_agent(test_llm):
    """Provides a configured Agent instance."""
    return Agent(
        name="TestReactAgentInWorkflow",
        id="test_react_agent_workflow_node",
        llm=test_llm,
        role="You are a helpful assistant.",
        inference_mode=InferenceMode.DEFAULT,
        tools=[],
        memory=None,
        max_loops=3,
    )


@pytest.fixture
def test_workflow(test_react_agent):
    """Provides a Workflow containing the test Agent."""
    return Workflow(flow=Flow(nodes=[test_react_agent]))


@pytest.fixture
def agent_input_data():
    """Provides standard input data for the agent."""
    return {
        "input": "What is 2 + 2?",
        "user_id": None,
        "session_id": None,
    }


@pytest.fixture
def run_config():
    """Provides a standard RunnableConfig."""
    return RunnableConfig(request_timeout=30)


@pytest.mark.unit
@pytest.mark.parametrize(
    "test_id,agent_name,role",
    [
        (
            "code_examples_with_braces",
            "TestReactAgentRoleCodeExamples",
            (
                "You are a helpful coding assistant.\n\n"
                "When providing examples, use proper formatting:\n"
                "- For JSON responses, use: ```json\n{'status':'success', 'data':'value'}```\n"
                "- For React components with props, show them like:\n"
                "```jsx\n"
                "{% raw %}<Button onClick={handleClick} data={{userId: 123}} />{% endraw %}\n"
                "```\n"
                "- Always include code context in your explanations."
                " Additional instructions: {{additional_instructions}}"
            ),
        ),
    ],
    ids=["code_examples_with_braces"],
)
def test_react_agent_role_with_special_characters(test_llm, test_id, agent_name, role):
    """
    Test agent roles with various special character patterns (curly braces, double braces, etc.)
    to ensure they don't break prompt rendering.

    Scenarios:
    - code_examples_with_braces: Role with single/double braces in code blocks and Jinja raw blocks
    """
    agent = Agent(
        name=agent_name,
        id=f"test_react_agent_{test_id}",
        llm=test_llm,
        role=role,
        inference_mode=InferenceMode.DEFAULT,
        tools=[],
        memory=None,
        verbose=False,
        max_loops=3,
    )

    input_data = {
        "input": "What is 2 + 2? Provide result in JSON format.",
        "additional_instructions": "In the very end of the response, say 'have a nice day!'.",
    }
    config = RunnableConfig(request_timeout=30)
    result = agent.run(input_data=input_data, config=config)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert isinstance(content, str)
    assert "4" in content, f"Expected '4' in the output, got: {content!r}"

    assert "have a nice day!" in content.lower(), f"Expected 'have a nice day!' in the output, got: {content!r}"
