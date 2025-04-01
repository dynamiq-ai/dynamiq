import uuid
from time import sleep

import pytest

from dynamiq import Workflow, connections, flows
from dynamiq.memory import Memory
from dynamiq.memory.backends import Pinecone, Qdrant
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.llms import OpenAI
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType

AGENT_ROLE = "helpful assistant, goal is to provide useful information and answer questions"
PERSONAL_INFO_INPUT = "Hi, my name is Alex and I work as a software engineer at TechCorp."
GENERAL_QUESTION_INPUT = "What's the weather forecast for tomorrow?"
MEMORY_TEST_INPUT = "What's my name and where do I work?"
RUN_CONFIG = RunnableConfig(request_timeout=120)


def verify_memory_and_cleanup(memory, memory_response, user_id, session_id):
    """Helper function to verify memory contents and clean up afterward."""
    assert "Alex" in memory_response, "Agent failed to remember the user's name"
    assert "TechCorp" in memory_response, "Agent failed to remember the user's workplace"

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
def test_react_agent_with_pinecone_memory(openai_llm, pinecone_connection, openai_embedder, monkeypatch):
    """Test ReActAgent with Pinecone memory backend."""
    memory_backend = Pinecone(
        index_name="test-memory-pinecone",
        connection=pinecone_connection,
        embedder=openai_embedder,
        index_type=PineconeIndexType.SERVERLESS,
        cloud="aws",
        region="us-east-1",
    )

    memory = Memory(backend=memory_backend)

    agent = ReActAgent(
        name="PineconeMemoryAgent",
        llm=openai_llm,
        tools=[],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.DEFAULT,
        memory=memory,
        verbose=True,
    )

    wf = Workflow(flow=flows.Flow(nodes=[agent]))

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    print(f"\nUsing user_id: {user_id} and session_id: {session_id}")

    print("\n--- Testing Pinecone Memory: Step 1 - Personal Info ---")
    result_1 = wf.run(
        input_data={"input": PERSONAL_INFO_INPUT, "user_id": user_id, "session_id": session_id},
        config=RUN_CONFIG,
    )

    assert result_1.status == RunnableStatus.SUCCESS
    print(f"Agent response: {result_1.output[agent.id]['output']['content']}")
    sleep(3)
    print("--- Testing Pinecone Memory: Step 2 - General Question ---")
    result_2 = wf.run(
        input_data={"input": GENERAL_QUESTION_INPUT, "user_id": user_id, "session_id": session_id},
        config=RUN_CONFIG,
    )

    assert result_2.status == RunnableStatus.SUCCESS
    print(f"Agent response: {result_2.output[agent.id]['output']['content']}")
    sleep(3)
    print("--- Testing Pinecone Memory: Step 3 - Memory Test ---")
    result_3 = wf.run(
        input_data={"input": MEMORY_TEST_INPUT, "user_id": user_id, "session_id": session_id},
        config=RUN_CONFIG,
    )

    assert result_3.status == RunnableStatus.SUCCESS
    memory_response = result_3.output[agent.id]["output"]["content"]
    print(f"Memory test response: {memory_response}")
    sleep(3)

    verify_memory_and_cleanup(memory, memory_response, user_id, session_id)

    print("--- Pinecone Memory Test Passed ---")


@pytest.mark.integration
def test_react_agent_with_qdrant_memory(openai_llm, qdrant_connection, openai_embedder, monkeypatch):
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
        role=AGENT_ROLE,
        inference_mode=InferenceMode.DEFAULT,
        memory=memory,
        verbose=True,
    )

    wf = Workflow(flow=flows.Flow(nodes=[agent]))

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    print(f"\nUsing user_id: {user_id} and session_id: {session_id}")

    print("\n--- Testing Qdrant Memory: Step 1 - Personal Info ---")
    result_1 = wf.run(
        input_data={"input": PERSONAL_INFO_INPUT, "user_id": user_id, "session_id": session_id},
        config=RUN_CONFIG,
    )

    assert result_1.status == RunnableStatus.SUCCESS
    print(f"Agent response: {result_1.output[agent.id]['output']['content']}")
    sleep(3)

    print("--- Testing Qdrant Memory: Step 2 - General Question ---")
    result_2 = wf.run(
        input_data={"input": GENERAL_QUESTION_INPUT, "user_id": user_id, "session_id": session_id},
        config=RUN_CONFIG,
    )

    assert result_2.status == RunnableStatus.SUCCESS
    print(f"Agent response: {result_2.output[agent.id]['output']['content']}")
    sleep(3)

    print("--- Testing Qdrant Memory: Step 3 - Memory Test ---")
    result_3 = wf.run(
        input_data={"input": MEMORY_TEST_INPUT, "user_id": user_id, "session_id": session_id},
        config=RUN_CONFIG,
    )

    assert result_3.status == RunnableStatus.SUCCESS
    memory_response = result_3.output[agent.id]["output"]["content"]
    print(f"Memory test response: {memory_response}")
    sleep(3)

    verify_memory_and_cleanup(memory, memory_response, user_id, session_id)

    print("--- Qdrant Memory Test Passed ---")
