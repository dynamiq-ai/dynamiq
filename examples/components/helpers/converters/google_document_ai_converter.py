import logging
import os
from io import BytesIO
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow
from dynamiq.components.splitters.document import DocumentSplitBy
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes.converters import GoogleDocumentAIFileConverter
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.splitters import DocumentSplitter
from dynamiq.types import DocumentCreationMode

CM = ConnectionManager()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

PDF_FILE_PATH = Path(__file__).resolve().parents[2] / "data" / "file.pdf"
PROCESSOR_ID = os.getenv("GOOGLE_DOCUMENT_AI_PROCESSOR_ID", "your-processor-id")


def google_document_ai_converter(pdf_file: BytesIO):
    """Extract the text of a PDF with Document AI, then split it for downstream indexing.

    The connection reads its project and location from GOOGLE_CLOUD_PROJECT_ID and
    GOOGLE_DOCUMENT_AI_LOCATION. Credentials come from the GOOGLE_CLOUD_* variables, or from
    Application Default Credentials when those are not set (`gcloud auth application-default login`).
    """
    converter_node = GoogleDocumentAIFileConverter(
        processor_id=PROCESSOR_ID,
        document_creation_mode=DocumentCreationMode.ONE_DOC_PER_FILE,
    )

    splitter_node = DocumentSplitter(
        split_by=DocumentSplitBy.PASSAGE,
        split_length=10,
        split_overlap=1,
        depends=[
            NodeDependency(converter_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[converter_node.id]}.output.documents",
            },
        ),
    )

    wf = Workflow(
        id="wf",
        flow=Flow(
            id="wf",
            nodes=[converter_node, splitter_node],
            connection_manager=CM,
        ),
    )

    output = wf.run(input_data={"files": [pdf_file]})

    documents = output.output[converter_node.id]["output"]["documents"]
    chunks = output.output[splitter_node.id]["output"]["documents"]

    logger.info(f"Extracted {documents[0]['metadata']['document_ai_pages']} pages into {len(chunks)} chunks.")
    logger.info(f"Extracted text:\n{documents[0]['content'][:2000]}")


if __name__ == "__main__":
    with open(PDF_FILE_PATH, "rb") as upload_file:
        file = BytesIO(upload_file.read())
        file.name = PDF_FILE_PATH.name

    google_document_ai_converter(file)
