from io import BytesIO
from unittest.mock import MagicMock

import pytest
from google.cloud import documentai
from pypdf import PdfWriter

from dynamiq import Workflow, flows
from dynamiq.components.converters.google_document_ai import MAX_PAGES_PER_REQUEST_IMAGELESS
from dynamiq.connections import GoogleDocumentAI
from dynamiq.nodes.converters.google_document_ai import GoogleDocumentAIFileConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableStatus
from dynamiq.types import DocumentCreationMode

PROCESSOR_PATH = "projects/proj/locations/eu/processors/abc123"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Google env vars would otherwise seed the connection defaults."""
    for key in list(__import__("os").environ):
        if key.startswith(("GOOGLE_CLOUD_", "GOOGLE_DOCUMENT_AI_")):
            monkeypatch.delenv(key, raising=False)


def build_pdf(page_count: int, filename: str = "file.pdf") -> BytesIO:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    buffer = BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    buffer.name = filename
    return buffer


def build_response(page_texts: list[str]) -> documentai.ProcessResponse:
    """Build a Document AI response whose pages anchor into a single flat text, as the API does."""
    text = ""
    pages = []
    for page_number, page_text in enumerate(page_texts, start=1):
        start = len(text)
        text += page_text
        pages.append(
            documentai.Document.Page(
                page_number=page_number,
                layout=documentai.Document.Page.Layout(
                    text_anchor=documentai.Document.TextAnchor(
                        text_segments=[
                            documentai.Document.TextAnchor.TextSegment(start_index=start, end_index=len(text))
                        ]
                    )
                ),
            )
        )
    return documentai.ProcessResponse(document=documentai.Document(text=text, pages=pages))


@pytest.fixture
def document_ai_connection():
    return GoogleDocumentAI(project_id="proj", location="eu")


@pytest.fixture
def document_ai_client():
    client = MagicMock(spec=documentai.DocumentProcessorServiceClient)
    client.processor_path.return_value = PROCESSOR_PATH
    client.process_document.return_value = build_response(["PAGE ONE\n", "PAGE TWO\n"])
    return client


@pytest.fixture
def document_ai_node(document_ai_connection, document_ai_client):
    return GoogleDocumentAIFileConverter(
        id="document_ai_converter",
        name="Test Google Document AI Converter",
        connection=document_ai_connection,
        client=document_ai_client,
        processor_id="abc123",
    )


@pytest.fixture
def output_node(document_ai_node):
    return Output(id="output_node", depends=[NodeDependency(document_ai_node)])


@pytest.fixture
def workflow(document_ai_node, output_node):
    return Workflow(
        id="test_workflow",
        flow=flows.Flow(nodes=[document_ai_node, output_node]),
        version="1",
    )


@pytest.mark.parametrize("input_type", ["file_paths", "files"], ids=["from_path", "from_bytesio"])
def test_workflow_converts_a_pdf_to_one_document(workflow, document_ai_node, tmp_path, input_type):
    if input_type == "file_paths":
        path = tmp_path / "file.pdf"
        path.write_bytes(build_pdf(2).getvalue())
        expected_path = str(path)
        input_data = {"file_paths": [expected_path]}
    else:
        expected_path = "file.pdf"
        input_data = {"files": [build_pdf(2)]}

    response = workflow.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    documents = response.output[document_ai_node.id]["output"]["documents"]
    assert len(documents) == 1
    assert documents[0]["content"] == "PAGE ONE\n\n\nPAGE TWO\n"
    assert documents[0]["metadata"] == {
        "file_path": expected_path,
        "document_ai_processor_id": "abc123",
        "document_ai_pages": 2,
    }


def test_documents_reach_the_downstream_node(workflow, document_ai_node, output_node):
    response = workflow.run(input_data={"files": [build_pdf(2)]})

    assert response.status == RunnableStatus.SUCCESS
    converter_documents = response.output[document_ai_node.id]["output"]["documents"]
    downstream_output = response.output[output_node.id]["output"][document_ai_node.id]["output"]
    assert downstream_output["documents"] == converter_documents


def test_one_doc_per_page_mode_emits_a_document_per_page(document_ai_connection, document_ai_client):
    node = GoogleDocumentAIFileConverter(
        id="document_ai_converter",
        connection=document_ai_connection,
        client=document_ai_client,
        processor_id="abc123",
        document_creation_mode=DocumentCreationMode.ONE_DOC_PER_PAGE,
    )
    workflow = Workflow(flow=flows.Flow(nodes=[node]))

    response = workflow.run(input_data={"files": [build_pdf(2)]})

    assert response.status == RunnableStatus.SUCCESS
    documents = response.output[node.id]["output"]["documents"]
    assert [document["content"] for document in documents] == ["PAGE ONE\n", "PAGE TWO\n"]
    assert [document["metadata"]["page_number"] for document in documents] == [1, 2]


def test_oversized_pdf_is_split_across_requests(workflow, document_ai_node, document_ai_client):
    document_ai_client.process_document.side_effect = [
        build_response([f"P{page}\n" for page in range(1, MAX_PAGES_PER_REQUEST_IMAGELESS + 1)]),
        build_response(["LAST\n"]),
    ]

    response = workflow.run(input_data={"files": [build_pdf(MAX_PAGES_PER_REQUEST_IMAGELESS + 1)]})

    assert response.status == RunnableStatus.SUCCESS
    assert document_ai_client.process_document.call_count == 2
    documents = response.output[document_ai_node.id]["output"]["documents"]
    assert documents[0]["content"].endswith("LAST\n")
    assert documents[0]["metadata"]["document_ai_pages"] == MAX_PAGES_PER_REQUEST_IMAGELESS + 1


def test_metadata_from_input_is_merged_into_the_documents(workflow, document_ai_node):
    response = workflow.run(input_data={"files": [build_pdf(2)], "metadata": {"source": "user_upload"}})

    assert response.status == RunnableStatus.SUCCESS
    documents = response.output[document_ai_node.id]["output"]["documents"]
    assert documents[0]["metadata"]["source"] == "user_upload"


def test_api_error_fails_the_node_and_skips_the_downstream_node(
    workflow, document_ai_node, output_node, document_ai_client
):
    from google.api_core.exceptions import InvalidArgument

    document_ai_client.process_document.side_effect = InvalidArgument("Document pages exceed the limit")

    response = workflow.run(input_data={"files": [build_pdf(2)]})

    assert response.status == RunnableStatus.FAILURE
    assert response.output[document_ai_node.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[document_ai_node.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_missing_input_fails_the_node(workflow, document_ai_node):
    response = workflow.run(input_data={})

    assert response.status == RunnableStatus.FAILURE
    assert response.output[document_ai_node.id]["status"] == RunnableStatus.FAILURE.value


def test_node_is_serializable_without_its_component(document_ai_node):
    data = document_ai_node.to_dict()

    assert "file_converter" not in data
    assert data["type"] == "dynamiq.nodes.converters.GoogleDocumentAIFileConverter"
    assert data["processor_id"] == "abc123"
    assert data["connection"]["location"] == "eu"
