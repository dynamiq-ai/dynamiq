import sys
import uuid

from dynamiq.connections import AWS
from dynamiq.memory import Memory
from dynamiq.memory.backends.dynamo_db import DynamoDB
from dynamiq.nodes.agents.simple import SimpleAgent
from examples.llm_setup import setup_llm


def setup_agent():
    """Sets up the SimpleAgent with DynamoDB memory."""
    llm = setup_llm()

    try:
        aws_connection = AWS()

        dynamod_db = DynamoDB(
            connection=aws_connection,
            table_name="messages",
            create_table_if_not_exists=True,
        )
        print("DynamoDB backend initialized. Connection test performed during init.")

    except Exception as e:
        print(f"FATAL: Failed to initialize DynamoDB backend: {e}", file=sys.stderr)
        print("Please ensure DynamoDB is running and accessible with correct credentials.", file=sys.stderr)
        raise

    memory = Memory(backend=dynamod_db, message_limit=50)

    AGENT_ROLE = "Helpful assistant focusing on the current conversation."
    agent = SimpleAgent(
        name="ChatAgent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
        memory=memory,
    )
    return agent


def chat_loop(agent: SimpleAgent):
    """Runs the main chat loop."""
    print("\nWelcome to the AI Chat (DynamoDB Backend)! (Type 'exit' to end)")

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
    print("Retrieving chat history for this specific session...")

    session_filters = {"user_id": user_id, "session_id": session_id}

    try:
        session_messages = agent.memory.search(filters=session_filters, limit=agent.memory.message_limit * 2)  #

        if session_messages:
            print(f"\n--- History for Session: {session_id} (User: {user_id}) ---")
            formatted_history = agent.memory._format_messages_as_string(session_messages, format_type="plain")
            print(formatted_history)
        else:
            print(f"\nNo messages found in memory for Session ID: {session_id} and User ID: {user_id}")

    except Exception as e:
        print(f"\nError retrieving chat history: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        print("Setting up agent with DynamoDB backend...")
        chat_agent = setup_agent()
        print("Agent setup complete.")
        chat_loop(chat_agent)
    except Exception as e:
        print(f"\nApplication failed during setup or execution: {e}", file=sys.stderr)
        sys.exit(1)
    print("\nChat application finished.")
