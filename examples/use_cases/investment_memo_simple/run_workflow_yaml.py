import json
import logging
from pathlib import Path

from dynamiq import Workflow, runnables
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.utils import JsonWorkflowEncoder

logger = logging.getLogger(__name__)


DEFAULT_COMPANY = "Acme Robotics"


def run_workflow(company: str = DEFAULT_COMPANY, trace_ui: bool = True):
    yaml_path = Path(__file__).parent / "workflow.yaml"
    tracing = TracingCallbackHandler()
    callbacks = [tracing]
    if trace_ui:
        callbacks.append(DynamiqTracingCallbackHandler(access_key="",
                                                       base_url=""))

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=str(yaml_path), connection_manager=cm, init_components=True)
        wf.run(
            input_data={"input": company},
            config=runnables.RunnableConfig(callbacks=callbacks),
        )

    # Validate traces serializable
    json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)

    logger.info(f"Workflow {wf.id} finished. Results:")
    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: \n{result}")

    return tracing.runs


if __name__ == "__main__":
    run_workflow()
