import uuid

import pytest

from dynamiq import Workflow, connections, flows, prompts
from dynamiq.memory import Memory
from dynamiq.memory.backend import InMemory
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableStatus

# Constants
AGENT_ROLE = "helpful assistant, goal is to provide useful information and answer questions"


@pytest.fixture
def openai_connection():
    return connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api-key",
    )


@pytest.fixture
def openai_node(openai_connection):
    return OpenAI(
        name="OpenAI",
        model="gpt-3.5-turbo",
        connection=openai_connection,
        prompt=prompts.Prompt(
            messages=[
                prompts.Message(
                    role="user",
                    content="{{input}}",
                ),
            ],
        ),
    )


def test_workflow_with_agent_and_in_memory_memory(openai_node):
    memory = Memory(backend=InMemory())
    agent = SimpleAgent(
        name="Agent",
        llm=openai_node,
        role=AGENT_ROLE,
        id="agent",
        memory=memory,
    )
    wf = Workflow(flow=flows.Flow(nodes=[agent]))

    user_input_1 = "Hi, what's the weather like today?"
    result_1 = wf.run(input_data={"input": user_input_1})
    assert result_1.status == RunnableStatus.SUCCESS

    user_input_2 = "And what about tomorrow?"
    result_2 = wf.run(input_data={"input": user_input_2})
    assert result_2.status == RunnableStatus.SUCCESS

    all_messages = memory.get_all()
    assert len(all_messages) == 4
    assert all_messages[0].role == MessageRole.USER
    assert all_messages[0].content == user_input_1

    assert all_messages[1].role == MessageRole.ASSISTANT
    assert all_messages[1].content == result_1.output[agent.id]["output"]["content"]

    assert all_messages[2].role == MessageRole.USER
    assert all_messages[2].content == user_input_2

    assert all_messages[3].role == MessageRole.ASSISTANT
    assert all_messages[3].content == result_2.output[agent.id]["output"]["content"]
