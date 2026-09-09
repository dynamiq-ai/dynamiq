"""Agent with a persistent, cross-conversation store backed by the Dynamiq API.

The persistent store gives an agent a durable namespace - ``memories/`` by default - that survives
the conversation. Every operation is a platform API call; the SDK never touches storage directly.

Two ways to attach it:

- **Routed under a file store** (``composite_agent``): the ordinary ``file-*`` tools serve one merged
  namespace, where ``memories/`` persists and everything else is ephemeral.
- **Standalone or beside a sandbox** (``standalone_agent``): the store gets its own ``memory-*``
  tools, since a sandbox owns an absolute filesystem that cannot share a namespace.

Run it twice. The first run records a preference; the second recalls it without being told.

Requires ``DYNAMIQ_API_KEY`` (and ``DYNAMIQ_URL`` if not the default), plus a memory store id.
"""

import os

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.file import DynamiqFileStore, FileStoreConfig, InMemoryFileStore, PersistentStoreConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

MEMORY_STORE_ID = os.getenv("DYNAMIQ_MEMORY_STORE_ID", "ms-123")
USER_ID = os.getenv("DYNAMIQ_USER_ID", "u-42")

AGENT_ROLE = "You are a helpful assistant that remembers what you learn about the user."

FIRST_RUN_QUERY = "I always want answers in British English. Remember that about me."
SECOND_RUN_QUERY = "What do you already know about how I like answers written?"


def persistent_store_config() -> PersistentStoreConfig:
    """Build the API-backed persistent store.

    ``user`` scopes access within the store and is never supplied by the agent - the API enforces
    isolation from it together with the connection credentials.
    """
    return PersistentStoreConfig(
        enabled=True,
        backend=DynamiqFileStore(
            connection=DynamiqConnection(),
            memory_store_id=MEMORY_STORE_ID,
            user=USER_ID,
        ),
        path_prefix="memories/",
        write_enabled=True,
    )


def composite_agent() -> Agent:
    """Persistent store routed under a file store: one namespace, the usual ``file-*`` tools."""
    return Agent(
        name="AgentWithPersistentStore",
        llm=setup_llm(),
        role=AGENT_ROLE,
        inference_mode=InferenceMode.DEFAULT,
        max_loops=8,
        file_store=FileStoreConfig(
            enabled=True,
            backend=InMemoryFileStore(),
            agent_file_write_enabled=True,  # required for the agent to write memories in this mode
        ),
        persistent_store=persistent_store_config(),
    )


def standalone_agent() -> Agent:
    """Persistent store on its own: the agent gets a dedicated ``memory-*`` tool set.

    This is also the shape used alongside a sandbox - pass ``sandbox=SandboxConfig(...)`` and the
    sandbox tools stay untouched beside these. Note that ``write_enabled`` is independent of the
    file store's ``agent_file_write_enabled``, so memory can be writable over a read-only workspace.
    """
    return Agent(
        name="AgentWithPersistentStoreOnly",
        llm=setup_llm(),
        role=AGENT_ROLE,
        inference_mode=InferenceMode.DEFAULT,
        max_loops=8,
        persistent_store=persistent_store_config(),
    )


def run(agent: Agent, query: str) -> str:
    result = agent.run(input_data={"input": query, "user_id": USER_ID})
    content = result.output.get("content")
    logger.info(f"Tools available: {[tool.name for tool in agent.tools]}")
    logger.info(f"Answer: {content}")
    return content


if __name__ == "__main__":
    agent = composite_agent()

    print("--- First run: record the preference ---")
    run(agent, FIRST_RUN_QUERY)

    # A fresh agent, no shared conversation: whatever it knows came from the persistent store.
    print("\n--- Second run: recall it in a new conversation ---")
    run(composite_agent(), SECOND_RUN_QUERY)
