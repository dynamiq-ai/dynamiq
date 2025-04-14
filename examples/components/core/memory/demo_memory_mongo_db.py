import os
import sys
import uuid

from dynamiq.connections import MongoDB as MongoDBConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.memory import Memory
from dynamiq.memory.backends.mongo_db import MongoDB as MongoDBMemoryBackend
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import OpenAI as OpenAI_LLM

MONGO_CONN_STR = os.getenv("MONGODB_CONNECTION_STRING", None)
MONGO_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGODB_PORT", 27017))
MONGO_DB = os.getenv("MONGODB_DATABASE", "chat_memory_mongo_db")
MONGO_USER = os.getenv("MONGODB_USER", None)
MONGO_PASSWORD = os.getenv("MONGODB_PASSWORD", None)

OPENAI_MODEL = "gpt-4o-mini"

MONGO_COLLECTION_NAME = "chat_history_mongo"
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
    """Sets up the SimpleAgent with MongoDB memory."""
    llm = setup_llm()

    try:
        # Setup MongoDB Connection
        if MONGO_CONN_STR:
            mongo_connection = MongoDBConnection(connection_string=MONGO_CONN_STR, database=MONGO_DB)
            print("Attempting to connect to MongoDB using connection string...")
        else:
            mongo_connection = MongoDBConnection(
                host=MONGO_HOST,
                port=MONGO_PORT,
                database=MONGO_DB,
                user=MONGO_USER,
                password=MONGO_PASSWORD,
            )
            print(f"Attempting to connect to MongoDB at {MONGO_HOST}:{MONGO_PORT} DB: {MONGO_DB}...")

        mongo_backend = MongoDBMemoryBackend(
            connection=mongo_connection,
            collection_name=MONGO_COLLECTION_NAME,
            create_indices_if_not_exists=True,
        )
        print(f"MongoDB memory backend initialized for collection '{MONGO_COLLECTION_NAME}'.")

    except Exception as e:
        print(f"FATAL: Failed to initialize MongoDB backend: {e}", file=sys.stderr)
        print(
            "Please ensure MongoDB is running and accessible via the configured connection details.",
            file=sys.stderr,
        )
        raise

    memory = Memory(backend=mongo_backend, message_limit=MEMORY_MESSAGE_LIMIT)

    AGENT_ROLE = "Helpful assistant that remembers previous conversations using MongoDB."
    agent = SimpleAgent(
        name="ChatAgentMongo",
        llm=llm,
        role=AGENT_ROLE,
        id="agent-mongo",
        memory=memory,
    )
    return agent


def chat_loop(agent: SimpleAgent):
    """Runs the main chat loop."""
    print("\nWelcome to the AI Chat (MongoDB Backend)! (Type 'exit' to end)")

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
        print("Setting up agent with MongoDB backend...")
        chat_agent = setup_agent()
        print("Agent setup complete. Starting chat loop.")
        chat_loop(chat_agent)
    except Exception as e:
        print(f"\nApplication failed during setup or execution: {e}", file=sys.stderr)
    print("\nChat application finished.")
