import os
import uuid
from time import sleep

import pytest

from dynamiq import Workflow, connections, flows
from dynamiq.memory import Memory
from dynamiq.memory.backends import DynamoDB, InMemory, Pinecone, Qdrant
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType
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
        model="gpt-5-mini",
        connection=openai_connection,
        max_tokens=1000,
        temperature=0,
    )


@pytest.fixture
def openai_embedder(openai_connection):
    return OpenAIDocumentEmbedder(connection=openai_connection)


@pytest.mark.skip(
    reason="Pinecone operates as an eventual consistency database, "
    "meaning that write operations may not be immediately visible. "
    "This inherent delay can result in intermittent test failures, "
    "so the test is being skipped until a more deterministic strategy is implemented."
)
def test_react_agent_with_pinecone_memory(
    openai_llm,
    pinecone_connection,
    openai_embedder,
    agent_role,
    personal_info_input,
    general_question_input,
    memory_test_input,
    run_config,
    monkeypatch,
):
    """Test Agent with Pinecone memory backend."""
    memory_backend = Pinecone(
        index_name="test-memory-pinecone",
        connection=pinecone_connection,
        embedder=openai_embedder,
        index_type=PineconeIndexType.SERVERLESS,
        cloud="aws",
        region="us-east-1",
    )

    memory = Memory(backend=memory_backend)

    agent = Agent(
        name="PineconeMemoryAgent",
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

    logger.info("\n--- Testing Pinecone Memory: Step 1 - Personal Info ---")
    result_1 = wf.run(
        input_data={"input": personal_info_input, "user_id": user_id, "session_id": session_id},
        config=run_config,
    )

    assert result_1.status == RunnableStatus.SUCCESS
    logger.info(f"Agent response: {result_1.output[agent.id]['output']['content']}")
    sleep(3)
    logger.info("--- Testing Pinecone Memory: Step 2 - General Question ---")
    result_2 = wf.run(
        input_data={"input": general_question_input, "user_id": user_id, "session_id": session_id},
        config=run_config,
    )

    assert result_2.status == RunnableStatus.SUCCESS
    logger.info(f"Agent response: {result_2.output[agent.id]['output']['content']}")
    sleep(3)
    logger.info("--- Testing Pinecone Memory: Step 3 - Memory Test ---")
    result_3 = wf.run(
        input_data={"input": memory_test_input, "user_id": user_id, "session_id": session_id},
        config=run_config,
    )

    assert result_3.status == RunnableStatus.SUCCESS
    memory_response = result_3.output[agent.id]["output"]["content"]
    logger.info(f"Memory test response: {memory_response}")
    sleep(3)

    verify_memory(memory, memory_response, user_id, session_id)

    logger.info("--- Pinecone Memory Test Passed ---")


@pytest.mark.skip(reason="Qdrant test  is skipped due to resolving issues with it")
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
    """Test Agent with Qdrant memory backend."""
    memory_backend = Qdrant(
        connection=qdrant_connection,
        embedder=openai_embedder,
        index_name="test-memory-qdrant",
    )

    memory = Memory(backend=memory_backend)

    agent = Agent(
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


@pytest.fixture(scope="module")
def aws_connection():
    """Provides an AWS connection, reading credentials from environment."""
    try:
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_REGION", "us-east-1")

        if not aws_access_key_id or not aws_secret_access_key:
            pytest.skip("AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) not found in environment.")

        return connections.AWS(
            access_key_id=aws_access_key_id,
            secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )
    except ImportError:
        pytest.skip("boto3 is not installed. Skipping AWS tests.")
    except Exception as e:
        pytest.fail(f"Failed to create AWS connection: {e}")


@pytest.mark.skip(reason="AWS DynamoDB test is skipped due to missing credentials. ")
def test_react_agent_with_dynamodb_memory(
    aws_connection,
    openai_llm,
    agent_role,
    personal_info_input,
    general_question_input,
    memory_test_input,
    run_config,
):
    """Test Agent with DynamoDB memory backend."""

    try:
        memory_backend = DynamoDB(
            connection=aws_connection,
            table_name="messages",
            create_if_not_exist=True,
        )
        logger.info("DynamoDB backend initialized.")
    except Exception as e:
        pytest.fail(f"FATAL: Failed to initialize DynamoDB backend: {e}")

    memory = Memory(backend=memory_backend, message_limit=20)

    agent = Agent(
        name="DynamoDBMemoryAgent",
        llm=openai_llm,
        tools=[],
        role=agent_role,
        inference_mode=InferenceMode.DEFAULT,
        memory=memory,
    )

    wf = Workflow(flow=flows.Flow(nodes=[agent]))

    user_id = f"user_{uuid.uuid4().hex[:6]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    logger.info(f"\nUsing user_id: {user_id} and session_id: {session_id}")

    logger.info("\n--- Testing DynamoDB Memory: Step 1 - Personal Info ---")
    result_1 = wf.run(
        input_data={"input": personal_info_input, "user_id": user_id, "session_id": session_id},
        config=run_config,
    )

    assert result_1.status == RunnableStatus.SUCCESS, f"Run 1 failed: {result_1.error}"
    logger.info(f"Agent response 1: {result_1.output.get(agent.id, {}).get('output', {}).get('content', 'N/A')}")

    logger.info("--- Testing DynamoDB Memory: Step 2 - General Question ---")
    result_2 = wf.run(
        input_data={"input": general_question_input, "user_id": user_id, "session_id": session_id},
        config=run_config,
    )

    assert result_2.status == RunnableStatus.SUCCESS, f"Run 2 failed: {result_2.error}"
    logger.info(f"Agent response 2: {result_2.output.get(agent.id, {}).get('output', {}).get('content', 'N/A')}")

    logger.info("--- Testing DynamoDB Memory: Step 3 - Memory Test ---")
    result_3 = wf.run(
        input_data={"input": memory_test_input, "user_id": user_id, "session_id": session_id},
        config=run_config,
    )

    assert result_3.status == RunnableStatus.SUCCESS, f"Run 3 failed: {result_3.error}"
    memory_response_data = result_3.output.get(agent.id, {}).get("output", {})
    memory_response = memory_response_data.get("content", "N/A")
    logger.info(f"Memory test response: {memory_response}")

    logger.info("Verifying memory contents...")
    try:
        verify_memory(memory, memory_response, user_id, session_id)
    except AssertionError as e:
        filters = {"user_id": user_id, "session_id": session_id}
        messages = memory.get_agent_conversation(filters=filters)
        logger.error(f"Memory verification failed: {e}")
        logger.error(f"Messages found ({len(messages)}): {messages}")
        pytest.fail(f"Memory verification failed: {e}")

    logger.info("--- DynamoDB Memory Test Passed ---")


@pytest.mark.integration
@pytest.mark.flaky(reruns=3)
def test_memory_snapshot_with_inmemory_backend(openai_llm, run_config):
    """Test that memory uses snapshot semantics: clear + rewrite after each run.

    Uses a Python tool so the agent produces tool-call / observation messages.

    Verifies:
    1. After turn 1, memory contains all non-system prompt messages (including
       tool interactions).
    2. No system messages leak into memory.
    3. After turn 2, memory is replaced (not appended) with the full
       conversation snapshot and the agent can recall the tool result from
       the previous turn.
    """
    lookup_tool = Python(
        name="company-lookup",
        description="Look up which company a person works at. Input: {'person': '<name>'}",
        code=(
            "def run(input_data):\n"
            f"    if '{USER_NAME}'.lower() in input_data.get('person', '').lower():\n"
            f"        return '{USER_NAME} works at {USER_COMPANY}.'\n"
            "    return 'Person not found.'\n"
        ),
    )

    memory = Memory(backend=InMemory())

    agent = Agent(
        name="InMemorySnapshotAgent",
        llm=openai_llm,
        tools=[lookup_tool],
        role=(
            "You are a helpful assistant. When asked about a person, "
            "use the company-lookup tool to find where they work. In final answer just provide: 'Found'"
        ),
        inference_mode=InferenceMode.XML,
        memory=memory,
        max_loops=3,
    )

    wf = Workflow(flow=flows.Flow(nodes=[agent]))
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # --- Turn 1: ask about a person (forces tool call) ---
    logger.info("--- InMemory Snapshot: Turn 1 - Tool call ---")
    result_1 = wf.run(
        input_data={
            "input": f"Where does {USER_NAME} work?",
            "user_id": user_id,
            "session_id": session_id,
        },
        config=run_config,
    )
    assert result_1.status == RunnableStatus.SUCCESS, f"Turn 1 failed: {result_1.error}"

    response_1 = result_1.output[agent.id]["output"]["content"]
    assert "Found" in response_1, f"Agent should mention 'Found': {response_1}"

    stored_after_t1 = memory.backend.messages
    assert len(stored_after_t1) == 4, (
        "Turn 1 must persist exactly 4 non-system messages: "
        "input, assistant tool-call message, observation, final answer"
    )

    for msg in stored_after_t1:
        assert msg.role != MessageRole.SYSTEM, f"System message leaked: {msg.content[:80]}"
        assert msg.metadata.get("user_id") == user_id
        assert msg.metadata.get("session_id") == session_id

    prompt_non_system_t1 = [m for m in agent._prompt.messages if m.role != MessageRole.SYSTEM]
    assert len(stored_after_t1) == len(
        prompt_non_system_t1
    ), f"Turn 1: memory ({len(stored_after_t1)}) != prompt ({len(prompt_non_system_t1)})"

    turn_1_contents = [m.content for m in stored_after_t1]
    turn_1_observations = [c for c in turn_1_contents if "Observation:" in c]
    assert len(turn_1_observations) == 1, "Turn 1 must contain exactly one observation message"
    assert USER_COMPANY in turn_1_observations[0], "Observation should contain tool output with company name"

    turn_1_user_count = sum(1 for m in stored_after_t1 if m.role == MessageRole.USER)
    turn_1_assistant_count = sum(1 for m in stored_after_t1 if m.role == MessageRole.ASSISTANT)
    assert turn_1_user_count == 2, "Turn 1 must have 2 USER messages (input + observation)"
    assert turn_1_assistant_count == 2, "Turn 1 must have 2 ASSISTANT messages (tool call + final answer)"

    turn_1_count = len(stored_after_t1)
    logger.info(f"Turn 1 stored {turn_1_count} message(s)")

    agent = Agent(
        name="SecondInMemorySnapshotAgent",
        llm=openai_llm,
        tools=[],
        role=("You are a helpful assistant."),
        inference_mode=InferenceMode.XML,
        memory=memory,
        max_loops=3,
    )

    wf1 = Workflow(flow=flows.Flow(nodes=[agent]))
    # --- Turn 2: ask agent to recall the tool result from turn 1 ---
    logger.info("--- InMemory Snapshot: Turn 2 - Recall ---")
    result_2 = wf1.run(
        input_data={
            "input": (
                f"Based on our previous conversation, remind me: where does {USER_NAME} work? "
                "Do NOT use any tool, just answer from what you already know."
            ),
            "user_id": user_id,
            "session_id": session_id,
        },
        config=run_config,
    )
    assert result_2.status == RunnableStatus.SUCCESS, f"Turn 2 failed: {result_2.error}"

    recall_response = result_2.output[agent.id]["output"]["content"]
    assert USER_COMPANY in recall_response, f"Agent should recall '{USER_COMPANY}' from memory: {recall_response}"

    stored_after_t2 = memory.backend.messages
    prompt_non_system_t2 = [m for m in agent._prompt.messages if m.role != MessageRole.SYSTEM]
    assert len(stored_after_t2) == len(
        prompt_non_system_t2
    ), f"Turn 2: memory ({len(stored_after_t2)}) != prompt ({len(prompt_non_system_t2)})"

    assert len(stored_after_t2) == turn_1_count + 2, (
        "After turn 2, memory must add exactly 2 messages: " "new user input and assistant final answer (no tool call)"
    )

    turn_2_new_messages = stored_after_t2[turn_1_count:]
    assert len(turn_2_new_messages) == 2
    assert turn_2_new_messages[0].role == MessageRole.USER, "Turn 2 first new message must be USER input"
    assert turn_2_new_messages[1].role == MessageRole.ASSISTANT, "Turn 2 second new message must be ASSISTANT output"
    assert "Observation:" not in turn_2_new_messages[0].content
    assert "Observation:" not in turn_2_new_messages[1].content

    for msg in stored_after_t2:
        assert msg.role != MessageRole.SYSTEM

    contents = " ".join(m.content for m in stored_after_t2)
    assert USER_COMPANY in contents, "Memory snapshot should still contain the tool result from turn 1"

    logger.info(f"Turn 2 stored {len(stored_after_t2)} message(s) — snapshot verified")
    logger.info("--- InMemory Snapshot Test Passed ---")
