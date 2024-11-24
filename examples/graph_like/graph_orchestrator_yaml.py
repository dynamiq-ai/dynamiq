import json
import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.utils import JsonWorkflowEncoder

logger = logging.getLogger(__name__)


INPUT_DATA = """
How causality is incorporated in Shapley values.
"""


def run_workflow():
    graph_orchestrator_yaml_file_path = os.path.join(os.path.dirname(__file__), "graph_orchestrator_wf.yaml")
    tracing = TracingCallbackHandler()
    with get_connection_manager() as cm:
        # Load the workflow from the YAML file, parse and init components during parsing
        wf = Workflow.from_yaml_file(
            file_path=graph_orchestrator_yaml_file_path, connection_manager=cm, init_components=True
        )
        wf.run(
            input_data={"input": INPUT_DATA},
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )
    # Check if traces dumped without errors
    json.dumps(
        {"runs": [run.to_dict() for run in tracing.runs.values()]},
        cls=JsonWorkflowEncoder,
    )

    logger.info(f"Workflow {wf.id} finished. Results:")
    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: \n{result}")

    return tracing.runs


if __name__ == "__main__":
    run_workflow()
