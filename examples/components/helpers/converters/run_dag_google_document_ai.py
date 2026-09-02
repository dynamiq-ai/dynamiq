"""Run the Google Document AI converter from a YAML workflow definition.

Before running, set the credentials described in `dag_google_document_ai.yaml`:

    export GOOGLE_CLOUD_PROJECT_ID=my-project
    export GOOGLE_DOCUMENT_AI_PROCESSOR_ID=1e0795e517c827d1
    export GOOGLE_DOCUMENT_AI_LOCATION=us          # or eu, us-central1, ...
    gcloud auth application-default login          # or set the service account env vars

Then:

    python examples/components/helpers/converters/run_dag_google_document_ai.py
"""

import logging
from io import BytesIO
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

EXAMPLE_DIR = Path(__file__).resolve().parent
YAML_FILE_PATH = EXAMPLE_DIR / "dag_google_document_ai.yaml"
PDF_FILE_PATH = EXAMPLE_DIR.parents[1] / "data" / "file.pdf"

CONVERTER_NODE_ID = "google-document-ai-converter-1"
SPLITTER_NODE_ID = "document-splitter-1"


def main():
    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=str(YAML_FILE_PATH), connection_manager=cm, init_components=True)

        with open(PDF_FILE_PATH, "rb") as upload_file:
            file = BytesIO(upload_file.read())
            file.name = PDF_FILE_PATH.name

        result = wf.run(input_data={"files": [file]})

        documents = result.output[CONVERTER_NODE_ID]["output"]["documents"]
        chunks = result.output[SPLITTER_NODE_ID]["output"]["documents"]

        logger.info(f"Extracted {documents[0]['metadata']['document_ai_pages']} pages into {len(chunks)} chunks.")
        logger.info(f"Extracted text:\n{documents[0]['content'][:2000]}")


if __name__ == "__main__":
    main()
