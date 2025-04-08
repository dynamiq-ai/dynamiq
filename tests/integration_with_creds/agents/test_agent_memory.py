import uuid
from time import sleep

import pytest

from dynamiq import Workflow, connections, flows
from dynamiq.memory import Memory
from dynamiq.memory.backends import Qdrant
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.llms import OpenAI
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger

USER_NAME = "Alex"
USER_COMPANY = "TechCorp"


@pytest.fixture
def personal_info_input():
    return f"Hi, my name is {USER_NAME} and I work as a software engineer at {USER_COMPANY}."


@pytest.fixture
def memory_test_input():
    return "What's my name and where do I work?"


@pytest.fixture
def general_question_input():
    return "What's the weather forecast for tomorrow?"


@pytest.fixture
def run_config():
    return RunnableConfig(request_timeout=120)


@pytest.fixture
def agent_role():
    return (
        "You are a helpful assistant that answers user queries and "
        "remembers personal information shared during the conversation."
    )


def verify_memory(memory, memory_response, user_id, session_id):
    """
    Verifies that the memory contains the expected conversation details for a given user and session.

    This helper function performs the following checks:
      1. Confirms that the agent's memory response includes the user's name ("Alex") and workplace ("TechCorp").
      2. Retrieves the conversation messages filtered by the provided user_id and session_id.
      3. Asserts that exactly six messages have been recorded in the conversation.

    Parameters:
        memory: An instance of the Memory class managing conversation storage.
        memory_response: The string output from the agent containing recalled information.
        user_id: A unique identifier for the user, used to filter stored messages.
        session_id: A unique identifier for the session, used to filter stored messages.

    Raises:
        AssertionError: If any of the expected conditions are not met.
    """
    assert USER_NAME in memory_response, f"Expected user name '{USER_NAME}' in memory response"
    assert USER_COMPANY in memory_response, f"Expected user company '{USER_COMPANY}' in memory response"

    filters = {"user_id": user_id, "session_id": session_id}
    conversation_messages = memory.get_agent_conversation(filters=filters)
    assert len(conversation_messages) == 6, f"Expected 6 messages, found {len(conversation_messages)}"


@pytest.fixture
def openai_connection():
    return connections.OpenAI()


@pytest.fixture
def pinecone_connection():
    return connections.Pinecone()


@pytest.fixture
def qdrant_connection():
    return connections.Qdrant()


@pytest.fixture
def openai_llm(openai_connection):
    return OpenAI(
        name="OpenAI",
        model="gpt-4o-mini",
        connection=openai_connection,
        max_tokens=1000,
        temperature=0,
    )


@pytest.fixture
def openai_embedder(openai_connection):
    return OpenAIDocumentEmbedder(connection=openai_connection)


@pytest.mark.integration
def test_react_agent_with_qdrant_memory(
    openai_llm,
    qdrant_connection,
    openai_embedder,
    agent_role,
    personal_info_input,
    general_question_input,
    memory_test_input,
    run_config,
    monkeypatch,
):
    """Test ReActAgent with Qdrant memory backend."""
    memory_backend = Qdrant(
        connection=qdrant_connection,
        embedder=openai_embedder,
        index_name="test-memory-qdrant",
    )

    memory = Memory(backend=memory_backend)

    agent = ReActAgent(
        name="QdrantMemoryAgent",
        llm=openai_llm,
        tools=[],
        role=agent_role,
        inference_mode=InferenceMode.DEFAULT,
        memory=memory,
        verbose=True,
    )

    wf = Workflow(flow=flows.Flow(nodes=[agent]))

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    logger.info(f"\nUsing user_id: {user_id} and session_id: {session_id}")

    logger.info("\n--- Testing Qdrant Memory: Step 1 - Personal Info ---")
    result_1 = wf.run(
        input_data={"input": personal_info_input, "user_id": user_id, "session_id": session_id},
        config=run_config,
    )

    assert result_1.status == RunnableStatus.SUCCESS
    logger.info(f"Agent response: {result_1.output[agent.id]['output']['content']}")
    sleep(3)

    logger.info("--- Testing Qdrant Memory: Step 2 - General Question ---")
    result_2 = wf.run(
        input_data={"input": general_question_input, "user_id": user_id, "session_id": session_id},
        config=run_config,
    )

    assert result_2.status == RunnableStatus.SUCCESS
    logger.info(f"Agent response: {result_2.output[agent.id]['output']['content']}")
    sleep(3)

    logger.info("--- Testing Qdrant Memory: Step 3 - Memory Test ---")
    result_3 = wf.run(
        input_data={"input": memory_test_input, "user_id": user_id, "session_id": session_id},
        config=run_config,
    )

    assert result_3.status == RunnableStatus.SUCCESS
    memory_response = result_3.output[agent.id]["output"]["content"]
    logger.info(f"Memory test response: {memory_response}")
    sleep(3)

    verify_memory(memory, memory_response, user_id, session_id)

    logger.info("--- Qdrant Memory Test Passed ---")
