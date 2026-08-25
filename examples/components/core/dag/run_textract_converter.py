"""Run the Amazon Textract converter workflow defined in `textract_converter.yaml`.
"""

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils import JsonWorkflowEncoder

load_dotenv(find_dotenv())

EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLES_DIR.parent.parent / "data"
YAML_PATH = EXAMPLES_DIR / "textract_converter.yaml"

DEFAULT_FILE_PATH = DATA_DIR / "scanned-invoice.png"

CONVERTER_NODE_ID = "textract-file-converter"

QUERIES = [
    {"text": "What is the invoice number?", "alias": "invoice_number"},
    {"text": "What is the total amount due?", "alias": "total_due"},
    {"text": "What is the due date?", "alias": "due_date"},
    {"text": "Who is the invoice billed to?", "alias": "bill_to"},
]


def load_file(file_path: str | Path) -> BytesIO:
    """Read a document into a named BytesIO, the shape the `list[file]` input schema expects."""
    file_path = Path(file_path)
    file = BytesIO(file_path.read_bytes())
    file.name = file_path.name
    return file


def build_tracing() -> TracingCallbackHandler:
    """Trace to the Dynamiq UI when DYNAMIQ_TRACE_ACCESS_KEY is set, otherwise trace in memory."""
    access_key = os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY") or os.environ.get("DYNAMIQ_ACCESS_KEY")
    if not access_key:
        return TracingCallbackHandler()

    return DynamiqTracingCallbackHandler(
        base_url=os.environ.get("DYNAMIQ_TRACE_BASE_URL", "https://collector.sandbox.getdynamiq.ai"),
        access_key=access_key,
    )


def run_textract_converter(
    file_path: str | Path = DEFAULT_FILE_PATH,
    use_queries: bool = False,
    one_doc_per_page: bool = False,
    callbacks: list | None = None,
):
    """Load the workflow from YAML and convert a single document with Textract.

    Args:
        file_path: Local scanned document (PDF, PNG, JPEG or TIFF).
        use_queries: Add the QUERIES feature and ask `QUERIES` against the document. The answers
            land in `metadata["queries"]` and in a "## Query answers" section of the content.
        one_doc_per_page: Emit one Document per page instead of one per file.
        callbacks: Runnable callbacks, e.g. a tracing handler.
    """
    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=YAML_PATH, connection_manager=cm, init_components=True)

        converter = wf.flow._node_by_id[CONVERTER_NODE_ID]
        if use_queries or one_doc_per_page:
            if use_queries:
                converter.feature_types = converter.feature_types + ["QUERIES"]
                converter.queries = QUERIES
            if one_doc_per_page:
                converter.document_creation_mode = "one-doc-per-page"
            converter.file_converter = None
            converter.init_components(cm)

        result = wf.run(
            input_data={"files": [load_file(file_path)], "metadata": {"source": "textract-example"}},
            config=RunnableConfig(callbacks=callbacks or []),
        )
    return wf, result


def render_documents(result) -> str:
    """Render the Markdown content plus the structured data Textract returned alongside it."""
    documents = result.output["output"]["output"]["output"]

    out = [f"=== {len(documents)} document(s) ==="]
    for document in documents:
        metadata = document["metadata"]
        out += [
            "",
            "-" * 80,
            f"file:     {metadata.get('file_path')}",
            f"page:     {metadata.get('page_number', '-')}",
            f"pages:    {metadata.get('textract_pages')}",
            f"mode:     {metadata.get('textract_processing_mode')}",
            f"features: {metadata.get('textract_feature_types')}",
            "-" * 80,
            document["content"],
        ]

        for table in metadata.get("tables", []):
            out.append(f"\n[table] page {table['page']}, {len(table['rows'])} row(s):")
            out += [f"   {row}" for row in table["rows"]]

        if forms := metadata.get("forms", []):
            out.append("\n[forms]")
            out += [f"   {form['key']} = {form['value']}  (confidence {form['confidence']:.1f})" for form in forms]

        if queries := metadata.get("queries", []):
            out.append("\n[queries]")
            out += [f"   {query['alias']}: {query['answer']} (confidence {query['confidence']})" for query in queries]

        if signatures := metadata.get("signatures", []):
            out.append(f"\n[signatures] {len(signatures)} detected")

    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Amazon Textract converter workflow.")
    parser.add_argument("--file", default=str(DEFAULT_FILE_PATH), help="Scanned PDF, PNG, JPEG or TIFF to convert.")
    parser.add_argument("--queries", action="store_true", help="Also ask the example queries against the document.")
    parser.add_argument("--per-page", action="store_true", help="Emit one Document per page.")
    parser.add_argument("--out", help="Write the extracted text to this file instead of stdout.")
    parser.add_argument("--dump-traces", action="store_true", help="Print the collected trace runs as JSON.")
    args = parser.parse_args()

    tracing = build_tracing()
    print(f"Tracing: {type(tracing).__name__}")

    _, run_result = run_textract_converter(
        file_path=args.file,
        use_queries=args.queries,
        one_doc_per_page=args.per_page,
        callbacks=[tracing],
    )

    if run_result.status == RunnableStatus.SUCCESS:
        report = render_documents(run_result)
    else:
        report = f"FAILED\n{run_result.output}"

    if args.out:
        Path(args.out).write_text(report)
        print(f"Status: {run_result.status} -> wrote {args.out}")
    else:
        print(f"\nStatus: {run_result.status}\n{report}")

    if args.dump_traces:
        print("\n=== traces ===")
        print(json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder))
