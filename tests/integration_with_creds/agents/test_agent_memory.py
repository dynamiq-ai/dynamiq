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


def verify_memory_and_cleanup(memory, memory_response):
    """Helper function to verify memory contents and clean up afterward."""
    assert "Alex" in memory_response, "Agent failed to remember the user's name"
    assert "TechCorp" in memory_response, "Agent failed to remember the user's workplace"

    all_messages = memory.get_all()
    assert len(all_messages) == 6

    try:
        memory.clear()
        print("Memory successfully cleared")
    except Exception as e:
        print(f"Warning: Failed to clear memory: {e}")


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
        index_name="test-memory",
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

    print("\n--- Testing Pinecone Memory: Step 1 - Personal Info ---")
    result_1 = wf.run(
        input_data={"input": PERSONAL_INFO_INPUT, "user_id": "test-user-1", "session_id": "test-session-1"},
        config=RUN_CONFIG,
    )

    assert result_1.status == RunnableStatus.SUCCESS
    print(f"Agent response: {result_1.output[agent.id]['output']['content']}")

    print("--- Testing Pinecone Memory: Step 2 - General Question ---")
    result_2 = wf.run(
        input_data={"input": GENERAL_QUESTION_INPUT, "user_id": "test-user-1", "session_id": "test-session-1"},
        config=RUN_CONFIG,
    )

    assert result_2.status == RunnableStatus.SUCCESS
    print(f"Agent response: {result_2.output[agent.id]['output']['content']}")

    print("--- Testing Pinecone Memory: Step 3 - Memory Test ---")
    result_3 = wf.run(
        input_data={"input": MEMORY_TEST_INPUT, "user_id": "test-user-1", "session_id": "test-session-1"},
        config=RUN_CONFIG,
    )

    assert result_3.status == RunnableStatus.SUCCESS
    memory_response = result_3.output[agent.id]["output"]["content"]
    print(f"Memory test response: {memory_response}")

    verify_memory_and_cleanup(memory, memory_response)

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

    print("\n--- Testing Qdrant Memory: Step 1 - Personal Info ---")
    result_1 = wf.run(
        input_data={"input": PERSONAL_INFO_INPUT, "user_id": "test-user-2", "session_id": "test-session-2"},
        config=RUN_CONFIG,
    )

    assert result_1.status == RunnableStatus.SUCCESS
    print(f"Agent response: {result_1.output[agent.id]['output']['content']}")

    print("--- Testing Qdrant Memory: Step 2 - General Question ---")
    result_2 = wf.run(
        input_data={"input": GENERAL_QUESTION_INPUT, "user_id": "test-user-2", "session_id": "test-session-2"},
        config=RUN_CONFIG,
    )

    assert result_2.status == RunnableStatus.SUCCESS
    print(f"Agent response: {result_2.output[agent.id]['output']['content']}")

    print("--- Testing Qdrant Memory: Step 3 - Memory Test ---")
    result_3 = wf.run(
        input_data={"input": MEMORY_TEST_INPUT, "user_id": "test-user-2", "session_id": "test-session-2"},
        config=RUN_CONFIG,
    )

    assert result_3.status == RunnableStatus.SUCCESS
    memory_response = result_3.output[agent.id]["output"]["content"]
    print(f"Memory test response: {memory_response}")

    verify_memory_and_cleanup(memory, memory_response)

    print("--- Qdrant Memory Test Passed ---")
