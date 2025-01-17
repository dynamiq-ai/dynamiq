import json
from pathlib import Path

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger


def run_csv_embedding_workflow(csv_file_path: str, yaml_file_path: str):
    """
    Run the CSV embedding workflow.

    Args:
        csv_file_path: Path to the CSV file
        yaml_file_path: Path to the workflow YAML file
    """
    tracer = TracingCallbackHandler()

    input_data = {
        "file_paths": [csv_file_path],
        "delimiter": ",",
        "content_column": "Target",
        "metadata_columns": ["Feature_1", "Feature_2"],
    }

    with get_connection_manager() as cm:
        workflow = Workflow.from_yaml_file(file_path=yaml_file_path, connection_manager=cm, init_components=True)

        result = workflow.run(input_data=input_data, config=runnables.RunnableConfig(callbacks=[tracer]))

        trace_logs = json.dumps({"runs": [run.to_dict() for run in tracer.runs.values()]}, cls=JsonWorkflowEncoder)

        logger.info(f"Workflow {workflow.id} finished. Results:")
        for node_id, node_result in workflow.flow._results.items():
            logger.info(f"Node {node_id}-{workflow.flow._node_by_id[node_id].name}:")
            logger.info(f"Result: {node_result}")

        return result, trace_logs


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    csv_file_path = ".data/sample_regression_data.csv"
    yaml_file_path = current_dir / "csv_embedding_flow.yaml"

    result, trace_logs = run_csv_embedding_workflow(
        csv_file_path=str(csv_file_path), yaml_file_path=str(yaml_file_path)
    )

    embedded_docs = result.output.get("document-embedder", {}).get("output", {}).get("documents", [])
    logger.info(f"Number of embedded documents: {len(embedded_docs)}")
    if embedded_docs:
        logger.info("Sample embedded document:")
        logger.info(f"Content: {embedded_docs[0]['content']}")
        logger.info(f"Embedding dimension: {len(embedded_docs[0]['embedding'])}")
