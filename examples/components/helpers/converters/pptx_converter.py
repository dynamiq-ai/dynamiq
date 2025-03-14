import json
import logging
import os
from io import BytesIO

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.components.converters.unstructured import DocumentCreationMode
from dynamiq.connections import connections
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes.converters import PPTXFileConverter, UnstructuredFileConverter
from dynamiq.nodes.embedders import MistralDocumentEmbedder
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.utils import JsonWorkflowEncoder

CM = ConnectionManager()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())


# Please use your own pptx file path
PPTX_FILE_PATH = "introduction-to-llm.pptx"


def pptx_converter(pptx_file: BytesIO):
    pptx_converter_node = PPTXFileConverter(document_creation_mode=DocumentCreationMode.ONE_DOC_PER_FILE)

    mistral_text_embedder_node_pptx = MistralDocumentEmbedder(
        name="MistralDocumentEmbedderPPTX",
        depends=[
            NodeDependency(pptx_converter_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[pptx_converter_node.id]}.output.documents",
            },
        ),
    )

    wf = Workflow(
        id="wf",
        flow=Flow(
            id="wf",
            nodes=[pptx_converter_node, mistral_text_embedder_node_pptx],
            connection_manager=CM,
        ),
    )

    tracing = TracingCallbackHandler()
    output = wf.run(
        input_data={
            "files": [pptx_file],
        },
        config=runnables.RunnableConfig(callbacks=[tracing]),
    )
    # Ensure trace logs can be serialized to JSON
    json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)

    logger.info(f"Workflow result:{output}")


def unstructured_converter(pptx_file: BytesIO):
    unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
    unstructured_connection = connections.Unstructured(api_key=unstructured_api_key)

    unstructured_converter_node = UnstructuredFileConverter(
        connection=unstructured_connection, document_creation_mode=DocumentCreationMode.ONE_DOC_PER_FILE
    )

    mistral_text_embedder_node_unstructured = MistralDocumentEmbedder(
        name="MistralDocumentEmbedderUnstructured",
        depends=[
            NodeDependency(unstructured_converter_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[unstructured_converter_node.id]}.output.documents",
            },
        ),
    )
    wf = Workflow(
        id="wf",
        flow=Flow(
            id="wf",
            nodes=[unstructured_converter_node, mistral_text_embedder_node_unstructured],
            connection_manager=CM,
        ),
    )

    tracing = TracingCallbackHandler()
    output = wf.run(
        input_data={
            "files": [pptx_file],
        },
        config=runnables.RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )
    # Ensure trace logs can be serialized to JSON
    json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)
    logger.info(f"Workflow result:{output}")


if __name__ == "__main__":
    with open(PPTX_FILE_PATH, "rb") as upload_file:
        file = BytesIO(upload_file.read())
        file.name = upload_file.name

    unstructured_converter(file)
    pptx_converter(file)
