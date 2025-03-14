import logging
from io import BytesIO

from dynamiq import Workflow
from dynamiq.components.converters.unstructured import DocumentCreationMode
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes.converters import PyPDFConverter
from dynamiq.nodes.embedders import MistralDocumentEmbedder
from dynamiq.nodes.node import InputTransformer, NodeDependency

# Please use your own pdf file path
PYPDF_FILE_PATH = "introduction-to-llm.pdf"


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cm = ConnectionManager()

    file_converter_node = PyPDFConverter(document_creation_mode=DocumentCreationMode.ONE_DOC_PER_PAGE)
    mistral_text_embedder_node = MistralDocumentEmbedder(
        name="MistralDocumentEmbedder",
        depends=[
            NodeDependency(file_converter_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[file_converter_node.id]}.output.documents",
            },
        ),
    )

    wf = Workflow(
        id="wf",
        flow=Flow(
            id="wf",
            nodes=[file_converter_node, mistral_text_embedder_node],
            connection_manager=cm,
        ),
    )

    with open(PYPDF_FILE_PATH, "rb") as upload_file:
        file = BytesIO(upload_file.read())
        file.name = upload_file.name

    output = wf.run(
        input_data={
            "files": [file],
        }
    )
    logger.info(f"Workflow result:{output}")


if __name__ == "__main__":
    main()
