"""Agent with a memory store backed by the Dynamiq API.

The memory store gives an agent notes it keeps across conversations, reached through one tool with
actions: list, read, write, edit and delete. Every operation is a platform API call; the SDK never
touches storage directly.

Memory is independent of the workspace, so the same tool appears whether the agent has a file store,
a sandbox, or neither. Run this twice: the first run records a preference in passing, the second
recalls it in a brand-new conversation without being told it exists.

Requires ``DYNAMIQ_API_KEY`` (and ``DYNAMIQ_URL`` if not the default), plus a memory store id.
"""

import os

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.storages.memory import CompositeMemoryStore, DynamiqMemoryStore, MemoryStoreConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

MEMORY_STORE_ID = os.getenv("DYNAMIQ_MEMORY_STORE_ID", "ms-123")
USER_ID = os.getenv("DYNAMIQ_USER_ID", "u-42")

ROLE = "You are a helpful engineering assistant."

FIRST_RUN = (
    "Write me a merge sort in Python and save it to sort.py. "
    "One thing about my code style: every function gets a docstring, at most 3 lines."
)
SECOND_RUN = "How do I like my code written?"


def memory() -> DynamiqMemoryStore:
    """One memory, backed by the API.

    ``user_id`` scopes access within the store and is never supplied by the agent - the API enforces
    isolation from it together with the connection credentials. ``description`` is what the model is
    told this memory holds.
    """
    return DynamiqMemoryStore(
        connection=DynamiqConnection(),
        memory_store_id=MEMORY_STORE_ID,
        user_id=USER_ID,
        description="What you learn about this user: preferences, standing rules, corrections.",
    )


def agent_with_memory() -> Agent:
    """An agent with a workspace for its deliverables and a memory for what it learns."""
    return Agent(
        name="AgentWithMemory",
        llm=setup_llm(),
        role=ROLE,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        memory_store=MemoryStoreConfig(enabled=True, backend=memory()),
    )


def agent_with_two_memories() -> Agent:
    """Several memories, each behind its own prefix.

    There is no catch-all: a path matching no prefix is refused, naming the prefixes that exist, so
    a memory never lands somewhere arbitrary. The agent still gets exactly one tool.
    """
    team = DynamiqMemoryStore(
        connection=DynamiqConnection(),
        memory_store_id=os.getenv("DYNAMIQ_TEAM_MEMORY_STORE_ID", MEMORY_STORE_ID),
        user_id=USER_ID,
        description="Conventions the whole team follows. Shared and curated elsewhere.",
    )
    return Agent(
        name="AgentWithTwoMemories",
        llm=setup_llm(),
        role=ROLE,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        memory_store=MemoryStoreConfig(
            enabled=True,
            backend=CompositeMemoryStore(routes={"user/": memory(), "team/": team}),
        ),
    )


def run(agent: Agent, query: str) -> str:
    result = agent.run(input_data={"input": query, "user_id": USER_ID})
    content = result.output.get("content") if result.output else result.error
    logger.info(f"Tools available: {[tool.name for tool in agent.tools]}")
    logger.info(f"Answer: {content}")
    return str(content)


if __name__ == "__main__":
    print("--- First run: the preference is mentioned in passing, never 'remember this' ---")
    run(agent_with_memory(), FIRST_RUN)

    # A fresh agent, no shared conversation: whatever it knows came from the memory store.
    print("\n--- Second run: a new conversation recalls it ---")
    run(agent_with_memory(), SECOND_RUN)
