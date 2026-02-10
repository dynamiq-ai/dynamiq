"""
YAML-based agent example with E2B sandbox.

This example demonstrates:
1. Loading a YAML configuration file using Workflow.from_yaml_file()
2. Running an agent with E2B sandbox for shell command execution
"""

import json
import os

from dotenv import load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger

# Load environment variables from .env file in the current or parent directory
load_dotenv()

# Input data for the workflow
INPUT_DATA = (
    "Create file with python code that prints 'Hello, World!'. "
    "Execute this script and save results in file 'result.txt'. "
    "List all files in the current directory to verify there are two files. One for script one for result of execution."
    "Return created script and the result of execution. "
    "Always use relative paths or /home/user/ as the working directory. Use python3 instead of python."
)


def run_workflow():
    """
    Run the agent example using YAML configuration with E2B sandbox.

    Returns:
        tuple: (workflow, traces) for use in graph drawing
    """
    logger.info("Starting YAML-based agent with E2B sandbox example...")

    # Path to the YAML configuration
    dag_yaml_file_path = os.path.join(os.path.dirname(__file__), "agent_e2b_sandbox.yaml")

    try:
        tracing = TracingCallbackHandler()
        with get_connection_manager() as cm:
            # Load the workflow from the YAML file, parse and init components during parsing
            wf = Workflow.from_yaml_file(
                file_path=dag_yaml_file_path,
                connection_manager=cm,
                init_components=True,
            )

            wf.run(
                input_data={"input": INPUT_DATA},
                config=RunnableConfig(callbacks=[tracing]),
            )

        # Check if traces dumped without errors
        _ = json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        logger.info(f"Workflow {wf.id} finished. Results:")
        for node_id, result in wf.flow._results.items():
            logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: \n{result}")

        return wf, tracing.runs

    except Exception as e:
        logger.error(f"Error running YAML-based sandbox agent: {e}")
        raise


if __name__ == "__main__":
    print("=== YAML-based Agent with E2B Sandbox ===")
    print("This example loads configuration from YAML using Workflow.from_yaml_file()")
    print("The agent uses E2B sandbox for executing shell commands.")
    print()

    run_workflow()
