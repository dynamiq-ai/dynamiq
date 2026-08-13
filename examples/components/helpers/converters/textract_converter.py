import logging

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow
from dynamiq.components.converters.textract import TextractFeatureType, TextractQuery
from dynamiq.connections import AWSTextract
from dynamiq.flows import Flow
from dynamiq.nodes.converters import TextractFileConverter
from dynamiq.types import DocumentCreationMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())


SCANNED_FILE_PATH = "../../data/scanned-invoice.pdf"


def run_converter(converter: TextractFileConverter, input_data: dict) -> None:
    wf = Workflow(flow=Flow(nodes=[converter]))
    output = wf.run(input_data=input_data)

    for document in output.output[converter.id]["output"]["documents"]:
        logger.info(f"--- {document['metadata'].get('file_path')} ---")
        logger.info(document["content"])
        for table in document["metadata"].get("tables", []):
            logger.info(f"table on page {table['page']}: {table['rows']}")
        for form in document["metadata"].get("forms", []):
            logger.info(f"form field: {form['key']} = {form['value']}")
        for query in document["metadata"].get("queries", []):
            logger.info(f"query {query['alias']}: {query['answer']} ({query['confidence']})")


def plain_ocr() -> None:
    """Cheapest mode: DetectDocumentText, plain lines, no analysis features."""
    run_converter(TextractFileConverter(), {"file_paths": [SCANNED_FILE_PATH]})


def structured_extraction() -> None:
    """LAYOUT + TABLES + FORMS: markdown structure, markdown tables and key-value pairs."""
    converter = TextractFileConverter(
        feature_types=[
            TextractFeatureType.LAYOUT,
            TextractFeatureType.TABLES,
            TextractFeatureType.FORMS,
        ],
        skip_headers_and_footers=True,
        min_confidence=60,
    )
    run_converter(converter, {"file_paths": [SCANNED_FILE_PATH]})


def answer_queries() -> None:
    """QUERIES: ask questions directly against the document instead of post-processing text."""
    converter = TextractFileConverter(
        feature_types=[TextractFeatureType.QUERIES],
        queries=[
            TextractQuery(text="What is the invoice total?", alias="total"),
            TextractQuery(text="What is the invoice date?", alias="date"),
        ],
    )
    run_converter(converter, {"file_paths": [SCANNED_FILE_PATH]})


def large_document_via_s3() -> None:
    """Multi-page PDFs go through the asynchronous API, which reads the document from S3.

    Documents already in S3 are passed via `s3_objects` and are never read locally. Local files
    are staged in `staging_s3_bucket` and the staged copy is deleted once the job completes.
    """
    converter = TextractFileConverter(
        connection=AWSTextract(region="us-east-1"),
        feature_types=[TextractFeatureType.LAYOUT, TextractFeatureType.TABLES],
        staging_s3_bucket="my-textract-staging-bucket",
    )
    run_converter(
        converter,
        {
            "file_paths": [SCANNED_FILE_PATH],
            "s3_objects": [{"bucket": "my-documents-bucket", "name": "contracts/lease.pdf"}],
        },
    )


def one_document_per_page() -> None:
    """Split the output per page, which keeps page numbers on the resulting Documents."""
    converter = TextractFileConverter(
        document_creation_mode=DocumentCreationMode.ONE_DOC_PER_PAGE,
        feature_types=[TextractFeatureType.LAYOUT],
    )
    run_converter(converter, {"file_paths": [SCANNED_FILE_PATH]})


if __name__ == "__main__":
    plain_ocr()
    structured_extraction()
    answer_queries()
    one_document_per_page()
    large_document_via_s3()
