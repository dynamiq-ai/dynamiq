import sys
import uuid

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Weaviate as WeaviateConnection
from dynamiq.memory import Memory
from dynamiq.memory.backends.weaviate import Weaviate as WeaviateMemoryBackend
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.llms import OpenAI as OpenAI_LLM

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

WEAVIATE_INDEX_NAME = "default"
MEMORY_MESSAGE_LIMIT = 50


def setup_llm_and_embedder():
    """Sets up the LLM and Embedder."""
    try:
        openai_connection = OpenAIConnection()

        llm = OpenAI_LLM(connection=openai_connection, model=OPENAI_MODEL)
        print(f"LLM ({llm.name}) initialized.")

        embedder = OpenAIDocumentEmbedder(connection=openai_connection, model=OPENAI_EMBEDDING_MODEL)
        print(f"Embedder ({embedder.name}) initialized.")

        return llm, embedder
    except Exception as e:
        print(f"FATAL: Failed to initialize OpenAI LLM or Embedder: {e}", file=sys.stderr)
        print("Please ensure the 'openai' library is installed and OPENAI_API_KEY is set.", file=sys.stderr)
        raise


def setup_agent():
    """Sets up the SimpleAgent with Weaviate memory."""
    llm, embedder = setup_llm_and_embedder()

    try:

        weaviate_connection = WeaviateConnection()
        weaviate_backend = WeaviateMemoryBackend(
            connection=weaviate_connection,
            embedder=embedder,
            collection_name=WEAVIATE_INDEX_NAME,
            create_if_not_exist=True,
        )
        print(f"Weaviate memory backend initialized for index '{WEAVIATE_INDEX_NAME}'.")

    except Exception as e:
        print(f"FATAL: Failed to initialize Weaviate backend: {e}", file=sys.stderr)
        print("Please ensure Weaviate is running and accessible.", file=sys.stderr)
        raise

    memory = Memory(backend=weaviate_backend, message_limit=MEMORY_MESSAGE_LIMIT)

    AGENT_ROLE = "Helpful assistant that remembers previous conversations using Weaviate."
    agent = SimpleAgent(
        name="ChatAgentWeaviate",
        llm=llm,
        role=AGENT_ROLE,
        id="agent-weaviate",
        memory=memory,
    )
    return agent


def chat_loop(agent: SimpleAgent):
    """Runs the main chat loop."""
    print("\nWelcome to the AI Chat (Weaviate Backend)! (Type 'exit' to end)")

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
        print("Setting up agent with Weaviate backend...")
        chat_agent = setup_agent()
        print("Agent setup complete. Starting chat loop.")
        chat_loop(chat_agent)
    except Exception as e:
        print(f"\nApplication failed during setup or execution: {e}", file=sys.stderr)
    print("\nChat application finished.")
