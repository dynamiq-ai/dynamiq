import json
import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.utils import JsonWorkflowEncoder

logger = logging.getLogger(__name__)


INPUT_DATA = """
I need to write a literature overview on the topic of `SOLID on interview` for my article.
Use the latest and most relevant information from the internet and articles. Try to keep simple format like:
- Introduction
- Main concepts
- Conclusion
Also include the sources in the end of the document. Double check that the information is up-to-date and relevant.
Final result must be provided in a markdown format.
"""


if __name__ == "__main__":
    dag_yaml_file_path = os.path.join(os.path.dirname(__file__), "orchestrator_dag.yaml")
    tracing = TracingCallbackHandler()
    with get_connection_manager() as cm:
        # Load the workflow from the YAML file, parse and init components during parsing
        wf = Workflow.from_yaml_file(file_path=dag_yaml_file_path, connection_manager=cm, init_components=True)
        wf.run(
            input_data={"input": INPUT_DATA},
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )
    # Check if traces dumped without errors
    _ = json.dumps(
        {"runs": [run.to_dict() for run in tracing.runs.values()]},
        cls=JsonWorkflowEncoder,
    )

    logger.info(f"Workflow {wf.id} finished. Results:")
    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: \n{result}")
