"""Experiment harness for the Google Document AI converter.

Runs the converter against the real Document AI API across several scenarios and dumps the
results to .txt (human-readable text) and .json (full Document structure) so the output shape
can be inspected.

Credentials: either a service account JSON passed via GOOGLE_APPLICATION_CREDENTIALS_JSON_FILE,
or the GOOGLE_CLOUD_* env vars, or Application Default Credentials. See
`dag_google_document_ai.yaml` for the full list.

    export GOOGLE_APPLICATION_CREDENTIALS_JSON_FILE=./dynamiq-sandbox-xxxx.json
    export GOOGLE_DOCUMENT_AI_PROCESSOR_ID=1e0795e517c827d1
    python examples/components/helpers/converters/experiment_google_document_ai.py
"""

import json
import logging
import os
import time
from io import BytesIO
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from dynamiq.connections import GoogleDocumentAI
from dynamiq.nodes.converters import GoogleDocumentAIFileConverter
from dynamiq.runnables import RunnableStatus
from dynamiq.types import DocumentCreationMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

REPO_ROOT = Path(__file__).resolve().parents[4]
PDF_FILE_PATH = REPO_ROOT / "examples" / "components" / "data" / "file.pdf"
OUTPUT_DIR = REPO_ROOT / ".data" / "document_ai_output"

PROCESSOR_ID = os.getenv("GOOGLE_DOCUMENT_AI_PROCESSOR_ID", "your-processor-id")
LONG_PDF_PAGE_COUNT = 65


def load_service_account_env():
    """Populate the GOOGLE_CLOUD_* vars from a service account JSON file, when one is configured."""
    key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON_FILE")
    if not key_file or not os.path.exists(key_file):
        return

    with open(key_file) as f:
        key = json.load(f)

    os.environ.update(
        {
            "GOOGLE_CLOUD_PROJECT_ID": key["project_id"],
            "GOOGLE_CLOUD_PRIVATE_KEY": key["private_key"],
            "GOOGLE_CLOUD_PRIVATE_KEY_ID": key.get("private_key_id", ""),
            "GOOGLE_CLOUD_CLIENT_EMAIL": key["client_email"],
            "GOOGLE_CLOUD_TOKEN_URI": key["token_uri"],
        }
    )
    logger.info(f"Using service account {key['client_email']}")


def build_long_pdf(page_count: int) -> BytesIO:
    """Build a PDF with a unique marker per page, to prove the split/stitch keeps page order."""
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    for page_number in range(1, page_count + 1):
        pdf.drawString(72, 720, f"MARKER PAGE {page_number}")
        pdf.showPage()
    pdf.save()

    long_pdf = BytesIO(buffer.getvalue())
    long_pdf.name = f"generated_{page_count}_pages.pdf"
    return long_pdf


def run_scenario(name: str, node: GoogleDocumentAIFileConverter, input_data: dict) -> dict:
    """Run one converter scenario and return a serializable record of the result."""
    started = time.perf_counter()
    result = node.run(input_data=input_data)
    elapsed = time.perf_counter() - started

    if result.status != RunnableStatus.SUCCESS:
        logger.error(f"{name}: FAILED - {result.output}")
        return {"scenario": name, "status": result.status.value, "error": str(result.output)}

    documents = result.output["documents"]
    logger.info(f"{name}: {len(documents)} document(s) in {elapsed:.2f}s")

    return {
        "scenario": name,
        "status": result.status.value,
        "elapsed_seconds": round(elapsed, 2),
        "document_count": len(documents),
        "documents": [document.to_dict() for document in documents],
    }


def write_outputs(records: list[dict]):
    """Dump the run to a readable .txt and a full .json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "document_ai_result.json"
    text_path = OUTPUT_DIR / "document_ai_result.txt"

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    with open(text_path, "w") as f:
        for record in records:
            f.write(f"{'=' * 78}\nSCENARIO: {record['scenario']}\nSTATUS: {record['status']}\n")
            if record["status"] != RunnableStatus.SUCCESS.value:
                f.write(f"ERROR: {record.get('error')}\n\n")
                continue

            f.write(f"DOCUMENTS: {record['document_count']} | TOOK: {record['elapsed_seconds']}s\n")
            for index, document in enumerate(record["documents"], start=1):
                f.write(f"\n--- document {index}/{record['document_count']} ---\n")
                f.write(f"id: {document['id']}\n")
                f.write(f"metadata: {json.dumps(document['metadata'], indent=2)}\n")
                f.write(f"content ({len(document['content'])} chars):\n{document['content']}\n")
            f.write("\n")

    logger.info(f"Wrote {json_path}")
    logger.info(f"Wrote {text_path}")
    return json_path, text_path


def main():
    load_service_account_env()
    connection = GoogleDocumentAI()
    logger.info(f"project={connection.project_id} location={connection.location} endpoint={connection.api_endpoint}")

    per_file = GoogleDocumentAIFileConverter(connection=connection, processor_id=PROCESSOR_ID)
    per_page = GoogleDocumentAIFileConverter(
        connection=connection,
        processor_id=PROCESSOR_ID,
        document_creation_mode=DocumentCreationMode.ONE_DOC_PER_PAGE,
    )

    pdf_path = str(PDF_FILE_PATH)
    records = [
        run_scenario("one-doc-per-file (2-page pdf)", per_file, {"file_paths": [pdf_path]}),
        run_scenario("one-doc-per-page (2-page pdf)", per_page, {"file_paths": [pdf_path]}),
        run_scenario(
            "one-doc-per-file with metadata",
            per_file,
            {"file_paths": [pdf_path], "metadata": {"source": "experiment", "tenant": "acme"}},
        ),
        run_scenario(
            f"{LONG_PDF_PAGE_COUNT}-page pdf (exceeds the online page limit, split locally)",
            per_file,
            {"files": [build_long_pdf(LONG_PDF_PAGE_COUNT)]},
        ),
    ]

    write_outputs(records)

    long_run = records[-1]
    if long_run["status"] == RunnableStatus.SUCCESS.value:
        content = long_run["documents"][0]["content"]
        pages = range(1, LONG_PDF_PAGE_COUNT + 1)
        positions = [content.index(f"MARKER PAGE {page}") for page in pages if f"MARKER PAGE {page}" in content]
        logger.info(
            f"{LONG_PDF_PAGE_COUNT}-page check: {len(positions)}/{LONG_PDF_PAGE_COUNT} markers, "
            f"ascending={positions == sorted(positions)}"
        )


if __name__ == "__main__":
    main()
