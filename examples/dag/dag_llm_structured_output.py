import json
import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.utils import JsonWorkflowEncoder

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    dag_yaml_file_path = os.path.join(os.path.dirname(__file__), "dag_llm_structured_output.yaml")
    tracing = TracingCallbackHandler()
    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=dag_yaml_file_path, connection_manager=cm, init_components=True)
        wf.run(
            input_data={},
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )
    _ = json.dumps(
        {"runs": [run.to_dict() for run in tracing.runs.values()]},
        cls=JsonWorkflowEncoder,
    )

    logger.info(f"Workflow {wf.id} finished. Results:")
    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: \n{result}")
