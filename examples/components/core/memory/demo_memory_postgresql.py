import os
import sys
import uuid

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.memory import Memory
from dynamiq.memory.backends.postgresql import PostgreSQL as PostgreSQLMemoryBackend
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import OpenAI as OpenAI_LLM

PG_HOST = os.getenv("POSTGRESQL_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRESQL_PORT", 5439))
PG_DB = os.getenv("POSTGRESQL_DATABASE", "default")
PG_USER = os.getenv("POSTGRESQL_USER", "default")

OPENAI_MODEL = "gpt-4o-mini"

POSTGRES_TABLE_NAME = "default"
MEMORY_MESSAGE_LIMIT = 50


def setup_llm():
    """Sets up the LLM."""
    try:
        openai_connection = OpenAIConnection()
        llm = OpenAI_LLM(connection=openai_connection, model=OPENAI_MODEL)
        print(f"LLM ({llm.name}) initialized.")
        return llm
    except Exception as e:
        print(f"FATAL: Failed to initialize OpenAI LLM: {e}", file=sys.stderr)
        print("Please ensure the 'openai' library is installed and OPENAI_API_KEY is set.", file=sys.stderr)
        raise


def setup_agent():
    """Sets up the SimpleAgent with PostgreSQL memory."""
    llm = setup_llm()

    try:
        pg_connection = PostgreSQLConnection(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB,
            user=PG_USER,
        )
        print(f"Attempting to connect to PostgreSQL at {PG_HOST}:{PG_PORT} DB: {PG_DB}...")

        pg_backend = PostgreSQLMemoryBackend(
            connection=pg_connection,
            table_name=POSTGRES_TABLE_NAME,
            create_if_not_exist=True,
        )
        print(f"PostgreSQL memory backend initialized for table '{POSTGRES_TABLE_NAME}'.")

    except Exception as e:
        print(f"FATAL: Failed to initialize PostgreSQL backend: {e}", file=sys.stderr)
        print(
            "Please ensure PostgreSQL is "
            f"running at {PG_HOST}:{PG_PORT}, DB '{PG_DB}' exists, "
            "and credentials are correct.",
            file=sys.stderr,
        )
        raise

    memory = Memory(backend=pg_backend, message_limit=MEMORY_MESSAGE_LIMIT)

    AGENT_ROLE = "Helpful assistant that remembers previous conversations using PostgreSQL."
    agent = SimpleAgent(
        name="ChatAgentPostgres",
        llm=llm,
        role=AGENT_ROLE,
        id="agent-postgres",
        memory=memory,
    )
    return agent


def chat_loop(agent: SimpleAgent):
    """Runs the main chat loop."""
    print("\nWelcome to the AI Chat (PostgreSQL Backend)! (Type 'exit' to end)")

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
            print(f"\nAn error occurred during chat: {e}", file=sys.stderr)
            break

    print("\n--- Chat Session Ended ---")


if __name__ == "__main__":
    try:
        print("Setting up agent with PostgreSQL backend...")
        chat_agent = setup_agent()
        print("Agent setup complete. Starting chat loop.")
        chat_loop(chat_agent)
    except Exception as e:
        print(f"\nApplication failed during setup or execution: {e}", file=sys.stderr)
    print("\nChat application finished.")
