import os

from dynamiq.memory import Memory
from dynamiq.memory.backends.internal import DynamiqInternalMemoryError
from dynamiq.memory.backends.internal import Internal as InternalMemoryBackend
from dynamiq.nodes.agents import Agent
from examples.llm_setup import setup_llm

MEMORY_ID = os.getenv("DYNAMIQ_INTERNAL_MEMORY_ID")


def build_agent() -> Agent:
    """
    Create a simple chat agent backed by Dynamiq internal memory.

    Required environment variables:
      - DYNAMIQ_INTERNAL_API_URL (optional – defaults to https://api.sandbox.getdynamiq.ai)
      - DYNAMIQ_INTERNAL_API_KEY
      - DYNAMIQ_INTERNAL_MEMORY_ID
    """
    if not MEMORY_ID:
        raise RuntimeError("Set DYNAMIQ_INTERNAL_MEMORY_ID before running this demo.")

    llm = setup_llm()
    memory_backend = InternalMemoryBackend(memory_id=MEMORY_ID)
    memory = Memory(backend=memory_backend)

    return Agent(
        name="InternalMemoryAgent",
        llm=llm,
        role="Helpful assistant that remembers previous messages via the Dynamiq internal memory backend.",
        id="internal-memory-agent",
        memory=memory,
    )


def chat_loop(agent: Agent) -> None:
    """Simple interactive loop that persists conversation state remotely."""
    print("Dynamiq Internal Memory Chat (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        try:
            response = agent.run({"input": user_input, "user_id": "demo-user", "session_id": "demo-session"})
        except DynamiqInternalMemoryError as error:
            print(f"[memory error] {error}")
            continue

        content = response.output.get("content")
        print(f"Agent: {content}")

    print("\nConversation history stored remotely:")
    try:
        for message in agent.memory.get_all():
            print(f"{message.role.value.title()}: {message.content}")
    except DynamiqInternalMemoryError as error:
        print(f"[memory error] Unable to read conversation: {error}")


if __name__ == "__main__":
    try:
        demo_agent = build_agent()
    except DynamiqInternalMemoryError as error:
        raise SystemExit(f"Dynamiq internal API error: {error}") from error
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    chat_loop(demo_agent)
