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

        dynamo_db = DynamoDB(
            connection=aws_connection,
            table_name="default",
            create_if_not_exist=True,
        )
        print("DynamoDB backend initialized. Connection test performed during init.")

    except Exception as e:
        print(f"FATAL: Failed to initialize DynamoDB backend: {e}", file=sys.stderr)
        print("Please ensure DynamoDB is running and accessible with correct credentials.", file=sys.stderr)
        raise

    memory = Memory(backend=dynamo_db, message_limit=50)

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


if __name__ == "__main__":
    try:
        print("Setting up agent with DynamoDB backend...")
        chat_agent = setup_agent()
        print("Agent setup complete.")
        chat_loop(chat_agent)
    except Exception as e:
        print(f"\nApplication failed during setup or execution: {e}", file=sys.stderr)
    print("\nChat application finished.")
