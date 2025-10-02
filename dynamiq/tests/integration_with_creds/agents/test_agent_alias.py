import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus


@pytest.mark.integration
def test_agent_alias_with_openai_default_mode():
    connection = OpenAIConnection()
    # Use a broadly supported chat model to avoid empty responses with reasoning models
    llm = OpenAI(connection=connection, model="gpt-4o-mini-2024-07-18", max_tokens=256, temperature=0)

    agent = Agent(
        name="Unified Agent Alias (OPENAI)",
        llm=llm,
        tools=[],
        role="is to help the user and include emojis",
        inference_mode=InferenceMode.DEFAULT,
        verbose=True,
    )

    result = agent.run(
        input_data={"input": "What is the capital of the UK?"},
        config=RunnableConfig(request_timeout=60),
    )
    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"] if isinstance(result.output, dict) else result.output
    assert isinstance(content, str) and len(content) > 0
    assert "London" in content
