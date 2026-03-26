import json
import logging
from io import BytesIO

from dotenv import find_dotenv, load_dotenv
from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.components.converters.unstructured import DocumentCreationMode
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import MistralDocumentEmbedder
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.utils import JsonWorkflowEncoder

from dynamiq.nodes.converters import MultiFileTypeConverter

CM = ConnectionManager()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())


DOCX_FILE_PATH = "../data/file.docx"
PPTX_FILE_PATH = "../data/file.pptx"
TXT_FILE_PATH = "../data/file.txt"
MARKDOWN_FILE_PATH = "../data/file.md"
HTML_FILE_PATH = "../data/file.html"
PDF_FILE_PATH = "../data/file.pdf"


def multi_file_type_converter(files: list[BytesIO]):
    multi_file_type_converter_node = MultiFileTypeConverter(
        document_creation_mode=DocumentCreationMode.ONE_DOC_PER_FILE
    )

    mistral_text_embedder_node_docx = MistralDocumentEmbedder(
        name="MistralDocumentEmbedderDOCX",
        depends=[
            NodeDependency(multi_file_type_converter_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[multi_file_type_converter_node.id]}.output.documents",
            },
        ),
    )

    wf = Workflow(
        id="wf",
        flow=Flow(
            id="wf",
            nodes=[multi_file_type_converter_node, mistral_text_embedder_node_docx],
            connection_manager=CM,
        ),
    )

    tracing = TracingCallbackHandler()
    output = wf.run(
        input_data={
            "files": files,
        },
        config=runnables.RunnableConfig(callbacks=[tracing]),
    )

    json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)

    logger.info(f"Workflow result:{output}")

    with open("output.md", "w") as f:
        print(output.output[multi_file_type_converter_node.id])
        f.write(output.output[multi_file_type_converter_node.id]["output"]["documents"][0]["content"])


if __name__ == "__main__":
    files_to_convert = []
    for file in [DOCX_FILE_PATH, PPTX_FILE_PATH, TXT_FILE_PATH, MARKDOWN_FILE_PATH, HTML_FILE_PATH, PDF_FILE_PATH]:
        with open(file, "rb") as upload_file:
            file = BytesIO(upload_file.read())
            file.name = upload_file.name
            files_to_convert.append(file)

    multi_file_type_converter(files_to_convert)
