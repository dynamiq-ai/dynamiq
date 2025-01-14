import json
from io import BytesIO

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.components.converters.unstructured import DocumentCreationMode
from dynamiq.connections import connections
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes.converters import CSVConverter, UnstructuredFileConverter
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger

connection_manager = ConnectionManager()

REGRESSION_DATA_PATH = ".data/df_final_Cluster_5.csv"


def create_embedder_node(source_node) -> OpenAIDocumentEmbedder:
    """Create an OpenAI embedder node with dependencies on the source node."""
    return OpenAIDocumentEmbedder(
        name="OpenAIEmbedder",
        depends=[NodeDependency(source_node)],
        input_transformer=InputTransformer(selector={"documents": f"${[source_node.id]}.output.documents"}),
    )


def process_csv_with_embeddings(
        file_content: BytesIO,
        delimiter: str = ",",
        content_column: str = "Target",
        metadata_columns: list[str] = ["Feature_1", "Feature_2"],
) -> dict:
    """
    Process CSV file and generate embeddings for specified columns.

    Args:
        file_content: CSV file content as BytesIO
        delimiter: CSV delimiter character
        content_column: Column to generate embeddings for
        metadata_columns: Additional columns to include as metadata

    Returns:
        dict: Processing results including embeddings
    """
    local_tracer = TracingCallbackHandler()

    csv_node = CSVConverter()
    embedder_node = create_embedder_node(csv_node)

    workflow = Workflow(
        id="csv_embedding_workflow",
        flow=Flow(id="csv_embedding_flow", nodes=[csv_node, embedder_node], connection_manager=connection_manager),
    )

    result = workflow.run(
        input_data={
            "files": [file_content],
            "delimiter": delimiter,
            "content_column": content_column,
            "metadata_columns": metadata_columns,
        },
        config=runnables.RunnableConfig(callbacks=[local_tracer]),
    )

    runs_json = json.dumps({"runs": [run.to_dict() for run in local_tracer.runs.values()]}, cls=JsonWorkflowEncoder)
    logger.debug(f"Workflow runs: {runs_json}")
    logger.info(f"CSV workflow result: {result}")

    return result


def process_unstructured_file(file_content: BytesIO) -> dict:
    """
    Process unstructured file and generate embeddings.

    Args:
        file_content: File content as BytesIO

    Returns:
        dict: Processing results including embeddings
    """
    local_tracer = TracingCallbackHandler()

    unstructured_connection = connections.Unstructured()
    converter_node = UnstructuredFileConverter(
        connection=unstructured_connection, document_creation_mode=DocumentCreationMode.ONE_DOC_PER_PAGE
    )
    embedder_node = create_embedder_node(converter_node)

    workflow = Workflow(
        id="unstructured_workflow",
        flow=Flow(
            id="unstructured_flow",
            nodes=[converter_node, embedder_node],
            connection_manager=connection_manager,
        ),
    )

    result = workflow.run(
        input_data={"files": [file_content]},
        config=runnables.RunnableConfig(callbacks=[local_tracer]),
    )

    runs_json = json.dumps({"runs": [run.to_dict() for run in local_tracer.runs.values()]}, cls=JsonWorkflowEncoder)
    logger.debug(f"Workflow runs: {runs_json}")
    logger.info(f"Unstructured workflow result: {result}")

    return result


if __name__ == "__main__":
    with open(REGRESSION_DATA_PATH, "rb") as csv_file:
        file_buffer = BytesIO(csv_file.read())
        file_buffer.name = csv_file.name

        csv_result = process_csv_with_embeddings(file_content=file_buffer)
        unstructured_result = process_unstructured_file(file_buffer)