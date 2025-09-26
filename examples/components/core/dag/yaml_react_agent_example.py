"""
YAML-based ReActAgent example with file storage and Python tool.

This example demonstrates:
1. Loading a YAML configuration file using Workflow.from_yaml_file()
2. Running a ReActAgent with file storage and Python tool
"""

import json
import os
import io
from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger

# Input data for the workflow
INPUT_DATA = "What is the content of provided image. Call FileReadTool two times with different instructions."
IMAGE_FILE = ""


def run_yaml_react_agent_example():
    """
    Run the ReActAgent example using YAML configuration with Workflow.from_yaml_file().

    Returns:
        tuple: (workflow, traces) for use in graph drawing
    """
    logger.info("Starting YAML-based ReActAgent file storage example...")

    # Path to the YAML configuration
    dag_yaml_file_path = os.path.join(os.path.dirname(__file__), "react_agent_file_storage.yaml")

    with open(IMAGE_FILE, "rb") as f:
        image_data = f.read()

    image_file = io.BytesIO(image_data)
    image_file.name = "image.png"

    try:
        tracing = TracingCallbackHandler()
        with get_connection_manager() as cm:
            # Load the workflow from the YAML file, parse and init components during parsing
            wf = Workflow.from_yaml_file(file_path=dag_yaml_file_path, connection_manager=cm, init_components=True)

            wf.run(
                input_data={"input": INPUT_DATA, "files": [image_file]},
                config=RunnableConfig(callbacks=[tracing]),
            )
        # Check if traces dumped without errors
        _ = json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        logger.info(f"Workflow {wf.id} finished. Results:")
        for node_id, result in wf.flow._results.items():
            logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: \\n{result}")

        return wf, tracing.runs

    except Exception as e:
        logger.error(f"Error running YAML-based ReActAgent: {e}")
        raise


if __name__ == "__main__":
    # from dynamiq.storages.file.in_memory import InMemoryFileStore
    # print(InMemoryFileStore().to_dict())

    print("=== YAML-based ReActAgent with File Storage and Python Tool ===")
    print("This example loads configuration from YAML using Workflow.from_yaml_file()")
    print()

    try:
        wf, traces = run_yaml_react_agent_example()
        print(f"\\nWorkflow {wf.id} completed successfully!")
        print(f"Generated {len(traces)} traces for graph visualization.")
    except Exception as e:
        print(f"Error: {e}")
