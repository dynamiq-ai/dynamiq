"""Example: agent checkpoint restores the sandbox connection.

Demonstrates that ``AgentCheckpointState`` now carries the sandbox reconnect
info (type, sandbox_id, base_path), so an agent restored from a checkpoint
reattaches to the SAME live sandbox - with all its files intact - instead of
silently starting a fresh empty one.

Flow:
  1. Agent A starts an E2B sandbox and writes a marker file into it.
  2. ``to_checkpoint_state()`` captures the sandbox id; the state is round-tripped
     through JSON to mimic a real checkpoint stored in a database.
  3. Agent B is built fresh (no sandbox_id) and restored from that state.
  4. Agent B's sandbox reconnects lazily and reads the marker file back.

Requires E2B_API_KEY. No LLM call is made (the LLM is postponed-init only).
"""

import json
import logging
import os
import uuid

from dynamiq.connections import E2B
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

MARKER_FILE = "checkpoint_marker.txt"
MARKER_CONTENT = f"written before checkpoint: {uuid.uuid4()}"


def build_agent(agent_id: str) -> Agent:
    return Agent(
        id=agent_id,
        name=agent_id,
        llm=OpenAI(
            model="gpt-4o-mini",
            connection=OpenAIConnection(api_key=os.getenv("OPENAI_API_KEY", "unused-in-this-demo")),
            is_postponed_component_init=True,
        ),
        role="Demo agent",
        sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B())),
    )


def main():
    if not os.getenv("E2B_API_KEY"):
        raise SystemExit("E2B_API_KEY is required for this demo.")

    # 1. Agent A starts a sandbox (lazily, on first file write) and leaves a marker.
    agent_a = build_agent("agent-a")
    sandbox_a = agent_a.sandbox.backend
    sandbox_a.store(MARKER_FILE, MARKER_CONTENT)
    print(f"\nAgent A sandbox started: {sandbox_a.sandbox_id}")
    print(f"Agent A wrote {MARKER_FILE}: {MARKER_CONTENT!r}")

    # 2. Checkpoint the agent and round-trip the state through JSON,
    #    the same way a real checkpoint is persisted to a database.
    state_dict = json.loads(json.dumps(agent_a.to_checkpoint_state().model_dump()))
    print(f"\nCheckpointed sandbox_state: {state_dict['sandbox_state']}")

    # 3. A fresh agent (new process in real life) has no sandbox id ...
    agent_b = build_agent("agent-b")
    sandbox_b = agent_b.sandbox.backend
    if sandbox_b.sandbox_id is not None:
        raise RuntimeError(f"Fresh agent unexpectedly has sandbox_id={sandbox_b.sandbox_id!r}")
    print(f"\nAgent B before restore: sandbox_id={sandbox_b.sandbox_id}")

    # ... until the checkpoint is restored.
    agent_b.from_checkpoint_state(state_dict)
    print(f"Agent B after restore:  sandbox_id={sandbox_b.sandbox_id}")

    try:
        # 4. First sandbox use reconnects to the SAME sandbox - the marker survives.
        restored_content = sandbox_b.retrieve(MARKER_FILE).decode("utf-8")
        print(f"\nAgent B read {MARKER_FILE}: {restored_content!r}")

        same_sandbox = sandbox_b.sandbox_id == sandbox_a.sandbox_id
        same_content = restored_content == MARKER_CONTENT
        if same_sandbox and same_content:
            print("\nPASS: checkpoint restored the sandbox - reconnected to the same "
                  "sandbox and found the file written before the checkpoint.")
        else:
            print(f"\nFAIL: same_sandbox={same_sandbox}, same_content={same_content}")
    finally:
        try:
            sandbox_b.close(kill=True)
            print("Sandbox closed.")
        except Exception as e:
            print(f"Failed to close sandbox: {e}")


if __name__ == "__main__":
    main()
