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


def process_csv_with_embeddings(
    file_content: BytesIO,
    delimiter: str = ",",
    content_column: str = "Target",
    metadata_columns: list[str] = ["Feature_1", "Feature_2"],
):
    csv_node = CSVConverter()

    embedder_node = OpenAIDocumentEmbedder(
        name="OpenAIEmbedder",
        depends=[NodeDependency(csv_node)],
        input_transformer=InputTransformer(selector={"documents": f"${[csv_node.id]}.output.documents"}),
    )

    workflow = Workflow(
        id="csv_embedding_workflow",
        flow=Flow(id="csv_embedding_flow", nodes=[csv_node, embedder_node], connection_manager=connection_manager),
    )

    tracer = TracingCallbackHandler()
    result = workflow.run(
        input_data={
            "files": [file_content],
            "delimiter": delimiter,
            "content_column": content_column,
            "metadata_columns": metadata_columns,
        },
        config=runnables.RunnableConfig(callbacks=[tracer]),
    )

    json.dumps({"runs": [run.to_dict() for run in tracer.runs.values()]}, cls=JsonWorkflowEncoder)

    logger.info(f"Workflow result: {result}")


def unstructured_converter(pptx_file: BytesIO):
    unstructured_connection = connections.Unstructured()

    unstructured_converter_node = UnstructuredFileConverter(
        connection=unstructured_connection, document_creation_mode=DocumentCreationMode.ONE_DOC_PER_PAGE
    )
    embedder_node = OpenAIDocumentEmbedder(
        name="OpenAIEmbedder",
        depends=[NodeDependency(unstructured_converter_node)],
        input_transformer=InputTransformer(
            selector={"documents": f"${[unstructured_converter_node.id]}.output.documents"}
        ),
    )

    wf = Workflow(
        id="wf",
        flow=Flow(
            id="wf",
            nodes=[unstructured_converter_node, embedder_node],
            connection_manager=connection_manager,
        ),
    )

    tracing = TracingCallbackHandler()
    output = wf.run(
        input_data={
            "files": [pptx_file],
        },
        config=runnables.RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )
    json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)
    logger.info(f"Workflow result:{output}")


if __name__ == "__main__":
    with open(REGRESSION_DATA_PATH, "rb") as csv_file:
        file_buffer = BytesIO(csv_file.read())
        file_buffer.name = csv_file.name

    process_csv_with_embeddings(file_content=file_buffer)
    unstructured_converter(file_buffer)
