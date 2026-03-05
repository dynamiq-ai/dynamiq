"""
Example: Memory snapshot with the Dynamiq remote memory backend.

Uses a Python tool so the agent produces tool-call / observation messages,
then verifies snapshot semantics across two turns.

Set the following environment variables (or replace the defaults below):
    DYNAMIQ_URL        - Dynamiq API base URL
    DYNAMIQ_API_KEY    - Dynamiq API key
    DYNAMIQ_MEMORY_ID  - ID of the remote memory resource
    OPENAI_API_KEY     - OpenAI API key

Usage:
    python examples/memory_snapshot_dynamiq_backend.py
"""

import os
import sys
import uuid

from dynamiq import Workflow, connections, flows
from dynamiq.memory import Memory
from dynamiq.memory.backends.dynamiq import Dynamiq as DynamiqBackend
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableConfig, RunnableStatus

DYNAMIQ_URL = os.getenv("DYNAMIQ_URL", "https://api.getdynamiq.ai")
DYNAMIQ_API_KEY = os.getenv("DYNAMIQ_API_KEY", "your_dynamiq_api_key")
DYNAMIQ_MEMORY_ID = os.getenv("DYNAMIQ_MEMORY_ID", "your_dynamiq_memory_id")

USER_NAME = "Alex"
USER_COMPANY = "TechCorp"


def main():
    openai_conn = connections.OpenAI()
    llm = OpenAI(name="OpenAI", model="gpt-4o-mini", connection=openai_conn)
    run_config = RunnableConfig(request_timeout=120)

    dynamiq_conn = connections.Dynamiq(url=DYNAMIQ_URL, api_key=DYNAMIQ_API_KEY)
    backend = DynamiqBackend(connection=dynamiq_conn, memory_id=DYNAMIQ_MEMORY_ID)
    memory = Memory(backend=backend)

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

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    filters = {"user_id": user_id, "session_id": session_id}

    other_user_id = str(uuid.uuid4())
    other_session_id = str(uuid.uuid4())
    other_filters = {"user_id": other_user_id, "session_id": other_session_id}

    # --- Seed another user's data that must survive our agent runs ---
    OTHER_USER_MESSAGE = "I need help with my billing issue."
    OTHER_ASSISTANT_MESSAGE = "Sure, I can help you with billing. What is your account number?"
    memory.add(role=MessageRole.USER, content=OTHER_USER_MESSAGE, metadata=other_filters.copy())
    memory.add(role=MessageRole.ASSISTANT, content=OTHER_ASSISTANT_MESSAGE, metadata=other_filters.copy())
    print(f"Seeded other user's data ({other_user_id[:8]}...): 2 messages")

    # --- Turn 1: ask about a person (forces tool call) ---

    agent = Agent(
        name="DynamiqSnapshotAgent",
        llm=llm,
        tools=[lookup_tool],
        role=(
            "You are a helpful assistant. When asked about a person, "
            "use the company-lookup tool to find where they work. "
            "In final answer just provide: 'Found'"
        ),
        inference_mode=InferenceMode.XML,
        memory=memory,
        max_loops=3,
    )

    wf = Workflow(flow=flows.Flow(nodes=[agent]))

    print(f"Turn 1: Where does {USER_NAME} work?")
    result_1 = wf.run(
        input_data={"input": f"Where does {USER_NAME} work?", **filters},
        config=run_config,
    )

    if result_1.status != RunnableStatus.SUCCESS:
        print(f"Turn 1 failed: {result_1.error}")
        sys.exit(1)

    response_1 = result_1.output[agent.id]["output"]["content"]
    print(f"Turn 1 response: {response_1}")

    stored_after_t1 = memory.get_agent_conversation(filters=filters)
    print(f"Turn 1: {len(stored_after_t1)} messages stored in memory")

    for i, msg in enumerate(stored_after_t1):
        print(f"  [{i}] {msg.role.value}: {msg.content[:80]}...")
        if msg.role == MessageRole.SYSTEM:
            print("  WARNING: System message leaked into memory!")

    prompt_non_system_t1 = [m for m in agent._prompt.messages if m.role != MessageRole.SYSTEM]
    print(f"Turn 1: {len(prompt_non_system_t1)} non-system messages in prompt")

    if len(stored_after_t1) != len(prompt_non_system_t1):
        print(f"MISMATCH: memory ({len(stored_after_t1)}) != prompt ({len(prompt_non_system_t1)})")

    # --- Turn 2: new agent recalls from memory ---

    agent2 = Agent(
        name="SecondDynamiqSnapshotAgent",
        llm=llm,
        tools=[],
        role="You are a helpful assistant.",
        inference_mode=InferenceMode.XML,
        memory=memory,
        max_loops=3,
    )

    wf2 = Workflow(flow=flows.Flow(nodes=[agent2]))

    recall_question = (
        f"Based on our previous conversation, remind me: where does {USER_NAME} work? "
        "Do NOT use any tool, just answer from what you already know."
    )
    print(f"\nTurn 2: {recall_question}")

    result_2 = wf2.run(
        input_data={"input": recall_question, **filters},
        config=run_config,
    )

    if result_2.status != RunnableStatus.SUCCESS:
        print(f"Turn 2 failed: {result_2.error}")
        sys.exit(1)

    recall_response = result_2.output[agent2.id]["output"]["content"]
    print(f"Turn 2 response: {recall_response}")

    stored_after_t2 = memory.get_agent_conversation(filters=filters)
    print(f"Turn 2: {len(stored_after_t2)} messages stored in memory")

    for i, msg in enumerate(stored_after_t2):
        print(f"  [{i}] {msg.role.value}: {msg.content[:80]}...")

    if USER_COMPANY in recall_response:
        print(f"\nSUCCESS: Agent recalled '{USER_COMPANY}' from memory")
    else:
        print(f"\nFAILED: Agent did not recall '{USER_COMPANY}' from memory")

    # --- Verify other user's data is preserved ---
    other_stored = memory.get_agent_conversation(filters=other_filters)
    print(f"\nOther user's messages after agent runs: {len(other_stored)}")
    for i, msg in enumerate(other_stored):
        print(f"  [{i}] {msg.role.value}: {msg.content[:80]}...")

    other_ok = (
        len(other_stored) == 2
        and any(OTHER_USER_MESSAGE in m.content for m in other_stored)
        and any(OTHER_ASSISTANT_MESSAGE in m.content for m in other_stored)
    )
    if other_ok:
        print("SUCCESS: Other user's data is intact")
    else:
        print("FAILED: Other user's data was corrupted or lost!")

    # Cleanup
    try:
        memory.delete(session_id=session_id, user_id=user_id)
        memory.delete(session_id=other_session_id, user_id=other_user_id)
        print("Cleaned up test data from remote store")
    except Exception as e:
        print(f"Warning: Failed to clean up: {e}")


if __name__ == "__main__":
    main()
