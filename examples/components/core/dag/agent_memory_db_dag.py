import os
import sys
import uuid

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils import logger

YAML_FILE_PATH = os.path.join(os.path.dirname(__file__), "agent_memory_dynamo_db_dag.yaml")
WORKFLOW_ID = "dynamodb-chat-workflow"


def run_chat_loop(wf: Workflow):
    """Runs the main chat loop using the loaded workflow."""
    print("\nWelcome to the AI Chat (DynamoDB Backend via YAML)! (Type 'exit' to end)")

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

            result = wf.run(input_data=agent_input)

            response_content = result.output.get("content", "...")
            print(f"AI: {response_content}")

        except KeyboardInterrupt:
            print("\nExiting chat loop.")
            break
        except Exception as e:
            logger.error(f"An error occurred during chat: {e}", exc_info=True)
            print(f"\nAn error occurred: {e}", file=sys.stderr)
            break

    print("\n--- Chat Session Ended ---")


if __name__ == "__main__":
    logger.info("Starting DynamoDB Chat Agent from YAML configuration...")

    try:
        with get_connection_manager() as cm:
            logger.info(f"Loading workflow '{WORKFLOW_ID}' from {YAML_FILE_PATH}...")
            wf_data = WorkflowYAMLLoader.load(
                file_path=YAML_FILE_PATH,
                connection_manager=cm,
                init_components=True,
            )

            wf = Workflow.from_yaml_file_data(file_data=wf_data, wf_id=WORKFLOW_ID)
            logger.info(f"Workflow '{wf.id}' loaded successfully.")

            run_chat_loop(wf)

    except FileNotFoundError:
        logger.error(f"FATAL: YAML configuration file not found at {YAML_FILE_PATH}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"FATAL: Error loading workflow '{WORKFLOW_ID}'. Is it defined correctly in the YAML? Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL: Application failed during setup or execution: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Chat application finished.")
