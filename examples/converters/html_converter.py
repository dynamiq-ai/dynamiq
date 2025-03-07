import logging
from io import BytesIO

from dynamiq import Workflow
from dynamiq.components.converters.unstructured import DocumentCreationMode
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes.converters import HTMLConverter
from dynamiq.nodes.embedders import MistralDocumentEmbedder
from dynamiq.nodes.node import InputTransformer, NodeDependency

# Please use your own HTML file path
HTML_FILE_PATH = "example.html"


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cm = ConnectionManager()

    # Create the HTML converter node
    html_converter_node = HTMLConverter(document_creation_mode=DocumentCreationMode.ONE_DOC_PER_FILE)

    # Create the embedder node that depends on the converter
    mistral_text_embedder_node = MistralDocumentEmbedder(
        name="MistralDocumentEmbedder",
        depends=[
            NodeDependency(html_converter_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${html_converter_node.id}.output.documents",
            },
        ),
    )

    # Create the workflow with both nodes
    wf = Workflow(
        id="wf",
        flow=Flow(
            id="wf",
            nodes=[html_converter_node, mistral_text_embedder_node],
            connection_manager=cm,
        ),
    )

    # Read and process the HTML file
    with open(HTML_FILE_PATH, "rb") as upload_file:
        file = BytesIO(upload_file.read())
        file.name = upload_file.name

    # Run the workflow
    output = wf.run(
        input_data={
            "files": [file],
        }
    )
    logger.info(f"Workflow result:{output}")


if __name__ == "__main__":
    main()
