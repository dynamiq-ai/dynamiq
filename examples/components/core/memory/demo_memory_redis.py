import sys
import uuid

from dynamiq.connections import Redis as RedisConnection
from dynamiq.memory import Memory
from dynamiq.memory.backends.redis import Redis
from dynamiq.nodes.agents.simple import SimpleAgent
from examples.llm_setup import setup_llm

REDIS_HOST = "localhost"
REDIS_PORT = 6380
REDIS_DB = 0
REDIS_KEY_PREFIX = "chat_app_sessions"


def setup_agent():
    """Sets up the SimpleAgent with Redis memory."""
    llm = setup_llm()

    try:
        # Create a RedisConnection object
        redis_connection = RedisConnection(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
        )

        redis_backend = Redis(connection=redis_connection, key_prefix=REDIS_KEY_PREFIX)
        print("Redis backend initialized. Connection test performed during init.")

    except Exception as e:
        print(f"FATAL: Failed to initialize Redis backend: {e}", file=sys.stderr)
        print("Please ensure Redis is running and accessible with correct credentials.", file=sys.stderr)
        raise

    memory = Memory(backend=redis_backend, message_limit=50)

    AGENT_ROLE = "Helpful assistant focusing on the current conversation."
    agent = SimpleAgent(
        name="RedisChatAgent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent_redis",
        memory=memory,
    )
    return agent


def chat_loop(agent: SimpleAgent):
    """Runs the main chat loop."""
    print("\nWelcome to the AI Chat (Redis Backend)! (Type 'exit' to end)")

    user_id = f"user_{uuid.uuid4().hex[:6]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    print("--- Starting New Chat ---")
    print(f"   User ID: {user_id}")
    print(f"   Session ID: {session_id}")
    print("-------------------------")

    while True:
        try:
            user_input = input(f"{user_id} You: ")
            if user_input.lower() == "exit":
                break
            if not user_input.strip():
                continue

            agent_input = {"input": user_input, "user_id": user_id, "session_id": session_id}

            response = agent.run(agent_input)
            response_content = response.output.get("content", "...")
            print(f"AI ({agent.name}): {response_content}")

        except KeyboardInterrupt:
            print("\nExiting chat loop.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)
            break

    print("\n--- Chat Session Ended ---")


if __name__ == "__main__":
    try:
        print("Setting up agent with Redis backend...")
        chat_agent = setup_agent()
        print("Agent setup complete.")
        chat_loop(chat_agent)
    except Exception as e:
        print(f"\nApplication failed during setup or execution: {e}", file=sys.stderr)
    print("\nChat application finished.")
