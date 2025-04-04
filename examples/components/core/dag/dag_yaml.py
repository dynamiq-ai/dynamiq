import asyncio
import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.connections.managers import get_connection_manager

logger = logging.getLogger(__name__)


def run_sync():
    dag_yaml_file_path = os.path.join(os.path.dirname(__file__), "dag.yaml")
    with get_connection_manager() as cm:
        # Load the workflow from the YAML file, parse and init components during parsing
        wf = Workflow.from_yaml_file(file_path=dag_yaml_file_path, connection_manager=cm, init_components=True)
        wf.run(
            input_data={"date": "4 May 2024", "next_date": "6 May 2024"},
            config=runnables.RunnableConfig(callbacks=[]),
        )
    logger.info(f"Workflow {wf.id} sync finished. Results: ")
    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: {result}")


async def run_async():
    dag_yaml_file_path = os.path.join(os.path.dirname(__file__), "dag.yaml")
    with get_connection_manager() as cm:
        # Load the workflow from the YAML file, parse and init components during parsing
        wf = Workflow.from_yaml_file(file_path=dag_yaml_file_path, connection_manager=cm, init_components=True)
        await wf.run(
            input_data={"date": "4 May 2024", "next_date": "6 May 2024"},
            config=runnables.RunnableConfig(callbacks=[]),
        )
    logger.info(f"Workflow {wf.id} async finished. Results: ")
    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: {result}")


if __name__ == "__main__":
    run_sync()
    asyncio.run(run_async())
