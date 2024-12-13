import json
import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.utils import JsonWorkflowEncoder

logger = logging.getLogger(__name__)

INPUT_DATA = "Dubai customs rules"

if __name__ == "__main__":
    yaml_file_path = os.path.join(os.path.dirname(__file__), "agent_rag.yaml")
    tracing = TracingCallbackHandler()

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=yaml_file_path, connection_manager=cm, init_components=True)

        result_1 = wf.run(
            input_data={"input": INPUT_DATA},
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )
        logger.info(f"Result 1: {result_1.output}")

    trace_dump = json.dumps(
        {"runs": [run.to_dict() for run in tracing.runs.values()]},
        cls=JsonWorkflowEncoder,
    )
    logger.info("Trace logs serialized successfully.")

    logger.info(f"Workflow {wf.id} finished. Results:")
    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: \n{result}")
