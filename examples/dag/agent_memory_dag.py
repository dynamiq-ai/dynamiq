import json
import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils import JsonWorkflowEncoder

logger = logging.getLogger(__name__)

INPUT_DATA_1 = "Hello! Can you help me with information about artificial intelligence?"
INPUT_DATA_2 = "Can you remind me what we discussed earlier?"

if __name__ == "__main__":
    yaml_file_path = os.path.join(os.path.dirname(__file__), "agent_memory_dag.yaml")
    tracing = TracingCallbackHandler()

    with get_connection_manager() as cm:
        # Load the workflow from the YAML file
        wf_data = WorkflowYAMLLoader.load(
            file_path=yaml_file_path,
            connection_manager=cm,
            init_components=True,
        )
        wf = Workflow.from_yaml_file_data(file_data=wf_data, wf_id="memory-agent-workflow")

        # Run the workflow with the first input
        result_1 = wf.run(
            input_data={"input": INPUT_DATA_1},
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )
        logger.info(f"Result 1: {result_1.output}")

        # Run the workflow with the second input
        result_2 = wf.run(
            input_data={"input": INPUT_DATA_2},
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )
        logger.info(f"Result 2: {result_2.output}")

    # Serialize trace logs
    trace_dump = json.dumps(
        {"runs": [run.to_dict() for run in tracing.runs.values()]},
        cls=JsonWorkflowEncoder,
    )
    logger.info("Trace logs serialized successfully.")

    logger.info(f"Workflow {wf.id} finished. Results:")
    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: \n{result}")
