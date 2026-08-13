from io import BytesIO
from unittest.mock import MagicMock

import pytest

from dynamiq import Workflow, flows
from dynamiq.components.converters.textract import (
    TextractFeatureType,
    TextractProcessingMode,
    TextractQuery,
    TextractS3Object,
)
from dynamiq.connections import AWSTextract
from dynamiq.nodes.converters.textract import TextractFileConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableStatus
from dynamiq.types import DocumentCreationMode

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def block(block_type: str, **kwargs) -> dict:
    """Build a Textract block, mirroring the API's PascalCase keys."""
    return {"BlockType": block_type, **kwargs}


def line(block_id: str, text: str, page: int = 1, confidence: float = 99.0) -> dict:
    return block("LINE", Id=block_id, Text=text, Page=page, Confidence=confidence)


def child(*ids: str) -> list[dict]:
    return [{"Type": "CHILD", "Ids": list(ids)}]


def geometry(left: float, top: float, width: float, height: float) -> dict:
    return {"BoundingBox": {"Left": left, "Top": top, "Width": width, "Height": height}}


@pytest.fixture
def connection():
    return AWSTextract(
        access_key_id="test-key",
        secret_access_key="test-secret",
        region="us-east-1",
        profile=None,
    )


@pytest.fixture
def client():
    # A bare MagicMock auto-creates a truthy `closed`, which makes the framework think the client
    # is dead and silently replace it with a real one. `spec` keeps it faithful to boto3.
    return MagicMock(
        spec=[
            "analyze_document",
            "detect_document_text",
            "start_document_analysis",
            "start_document_text_detection",
            "get_document_analysis",
            "get_document_text_detection",
        ]
    )


@pytest.fixture
def converter(connection, client):
    return TextractFileConverter(
        id="textract_converter",
        name="Test Textract Converter",
        connection=connection,
        client=client,
    )


@pytest.fixture
def png_file():
    file = BytesIO(PNG_HEADER + b"fake image bytes")
    file.name = "scan.png"
    return file


@pytest.fixture
def two_page_pdf():
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    buffer.name = "report.pdf"
    return buffer


def run_converter(converter, input_data) -> list:
    """Run the converter through a workflow, as the framework would, and return documents."""
    output_node = Output(id="output_node", depends=[NodeDependency(converter)])
    workflow = Workflow(flow=flows.Flow(nodes=[converter, output_node]), version="1")
    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.SUCCESS
    return result.output[converter.id]["output"]["documents"]


class TestDetectDocumentText:
    def test_lines_are_joined_in_reading_order(self, converter, client, png_file):
        client.detect_document_text.return_value = {
            "DocumentMetadata": {"Pages": 1},
            "DetectDocumentTextModelVersion": "1.0",
            "Blocks": [
                block("PAGE", Id="page-1", Relationships=child("line-1", "line-2")),
                line("line-1", "Invoice 42"),
                line("line-2", "Total: $10.00"),
            ],
        }

        documents = run_converter(converter, {"files": [png_file]})

        assert len(documents) == 1
        assert documents[0]["content"] == "Invoice 42\nTotal: $10.00"
        assert documents[0]["metadata"]["file_path"] == "scan.png"
        assert documents[0]["metadata"]["textract_processing_mode"] == TextractProcessingMode.SYNC.value
        assert documents[0]["metadata"]["textract_model_version"] == "1.0"
        # No features requested, so the cheaper API is used.
        client.analyze_document.assert_not_called()

    def test_low_confidence_lines_are_dropped(self, connection, client, png_file):
        client.detect_document_text.return_value = {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                line("line-1", "Confident text", confidence=99.0),
                line("line-2", "Smudged text", confidence=42.0),
            ],
        }
        converter = TextractFileConverter(connection=connection, client=client, min_confidence=80)

        documents = run_converter(converter, {"files": [png_file]})

        assert documents[0]["content"] == "Confident text"

    def test_unsupported_extension_is_rejected(self, converter, client):
        file = BytesIO(b"PK\x03\x04 docx bytes")
        file.name = "report.docx"

        output_node = Output(id="output_node", depends=[NodeDependency(converter)])
        workflow = Workflow(flow=flows.Flow(nodes=[converter, output_node]), version="1")
        result = workflow.run(input_data={"files": [file]})

        assert result.status == RunnableStatus.FAILURE
        client.detect_document_text.assert_not_called()

    def test_no_input_is_rejected(self, converter):
        output_node = Output(id="output_node", depends=[NodeDependency(converter)])
        workflow = Workflow(flow=flows.Flow(nodes=[converter, output_node]), version="1")

        assert workflow.run(input_data={}).status == RunnableStatus.FAILURE


class TestTables:
    @pytest.fixture
    def table_response(self):
        return {
            "DocumentMetadata": {"Pages": 1},
            "AnalyzeDocumentModelVersion": "1.0",
            "Blocks": [
                block("TABLE", Id="table-1", Confidence=98.0, Relationships=child("c11", "c12", "c21", "c22")),
                block("CELL", Id="c11", RowIndex=1, ColumnIndex=1, Relationships=child("w11")),
                block("CELL", Id="c12", RowIndex=1, ColumnIndex=2, Relationships=child("w12")),
                block("CELL", Id="c21", RowIndex=2, ColumnIndex=1, Relationships=child("w21")),
                block("CELL", Id="c22", RowIndex=2, ColumnIndex=2, Relationships=child("w22")),
                block("WORD", Id="w11", Text="Item"),
                block("WORD", Id="w12", Text="Price"),
                block("WORD", Id="w21", Text="Widget"),
                block("WORD", Id="w22", Text="10.00"),
            ],
        }

    def test_table_is_rendered_as_markdown_and_exposed_in_metadata(self, connection, client, png_file, table_response):
        client.analyze_document.return_value = table_response
        converter = TextractFileConverter(
            connection=connection, client=client, feature_types=[TextractFeatureType.TABLES]
        )

        documents = run_converter(converter, {"files": [png_file]})

        assert "| Item | Price |" in documents[0]["content"]
        assert "| --- | --- |" in documents[0]["content"]
        assert "| Widget | 10.00 |" in documents[0]["content"]

        tables = documents[0]["metadata"]["tables"]
        assert len(tables) == 1
        assert tables[0]["rows"] == [["Item", "Price"], ["Widget", "10.00"]]
        assert tables[0]["confidence"] == 98.0
        assert client.analyze_document.call_args.kwargs["FeatureTypes"] == ["TABLES"]

    def test_merged_cells_keep_column_alignment(self, connection, client, png_file):
        client.analyze_document.return_value = {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                block(
                    "TABLE",
                    Id="table-1",
                    Relationships=[
                        {"Type": "CHILD", "Ids": ["c11", "c12", "c21", "c22"]},
                        {"Type": "MERGED_CELL", "Ids": ["merged-1"]},
                    ],
                ),
                block("CELL", Id="c11", RowIndex=1, ColumnIndex=1, Relationships=child("w11")),
                block("CELL", Id="c12", RowIndex=1, ColumnIndex=2, Relationships=child("w12")),
                block("CELL", Id="c21", RowIndex=2, ColumnIndex=1, Relationships=child("w21")),
                block("CELL", Id="c22", RowIndex=2, ColumnIndex=2, Relationships=child("w22")),
                block(
                    "MERGED_CELL",
                    Id="merged-1",
                    RowIndex=1,
                    ColumnIndex=1,
                    RowSpan=1,
                    ColumnSpan=2,
                    Relationships=child("c11", "c12"),
                ),
                block("WORD", Id="w11", Text="Quarterly"),
                block("WORD", Id="w12", Text="Summary"),
                block("WORD", Id="w21", Text="Widget"),
                block("WORD", Id="w22", Text="10.00"),
            ],
        }
        converter = TextractFileConverter(
            connection=connection, client=client, feature_types=[TextractFeatureType.TABLES]
        )

        rows = run_converter(converter, {"files": [png_file]})[0]["metadata"]["tables"][0]["rows"]

        # The merged span's text lands in its top-left cell; the covered cell stays empty.
        assert rows == [["Quarterly Summary", ""], ["Widget", "10.00"]]

    def test_pipe_characters_in_cells_are_escaped(self, connection, client, png_file):
        client.analyze_document.return_value = {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                block("TABLE", Id="table-1", Relationships=child("c11", "c21")),
                block("CELL", Id="c11", RowIndex=1, ColumnIndex=1, Relationships=child("w11")),
                block("CELL", Id="c21", RowIndex=2, ColumnIndex=1, Relationships=child("w21")),
                block("WORD", Id="w11", Text="Header"),
                block("WORD", Id="w21", Text="a|b"),
            ],
        }
        converter = TextractFileConverter(
            connection=connection, client=client, feature_types=[TextractFeatureType.TABLES]
        )

        assert "| a\\|b |" in run_converter(converter, {"files": [png_file]})[0]["content"]


class TestForms:
    def test_key_value_pairs_are_rendered_and_exposed(self, connection, client, png_file):
        client.analyze_document.return_value = {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                block(
                    "KEY_VALUE_SET",
                    Id="key-1",
                    EntityTypes=["KEY"],
                    Confidence=97.5,
                    Relationships=[
                        {"Type": "CHILD", "Ids": ["w-key"]},
                        {"Type": "VALUE", "Ids": ["value-1"]},
                    ],
                ),
                block("KEY_VALUE_SET", Id="value-1", EntityTypes=["VALUE"], Relationships=child("w-value")),
                block("WORD", Id="w-key", Text="Invoice number"),
                block("WORD", Id="w-value", Text="INV-42"),
            ],
        }
        converter = TextractFileConverter(
            connection=connection, client=client, feature_types=[TextractFeatureType.FORMS]
        )

        documents = run_converter(converter, {"files": [png_file]})

        assert "## Form fields" in documents[0]["content"]
        assert "- **Invoice number**: INV-42" in documents[0]["content"]
        assert documents[0]["metadata"]["forms"] == [
            {"key": "Invoice number", "value": "INV-42", "page": 1, "confidence": 97.5}
        ]

    def test_low_confidence_key_value_pairs_are_dropped(self, connection, client, png_file):
        client.analyze_document.return_value = {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                block(
                    "KEY_VALUE_SET",
                    Id="key-good",
                    EntityTypes=["KEY"],
                    Confidence=95.0,
                    Relationships=[
                        {"Type": "CHILD", "Ids": ["w-key-good"]},
                        {"Type": "VALUE", "Ids": ["value-good"]},
                    ],
                ),
                block("KEY_VALUE_SET", Id="value-good", EntityTypes=["VALUE"], Relationships=child("w-value-good")),
                block("WORD", Id="w-key-good", Text="Invoice number", Confidence=95.0),
                block("WORD", Id="w-value-good", Text="INV-42", Confidence=95.0),
                block(
                    "KEY_VALUE_SET",
                    Id="key-noise",
                    EntityTypes=["KEY"],
                    Confidence=41.0,
                    Relationships=[
                        {"Type": "CHILD", "Ids": ["w-key-noise"]},
                        {"Type": "VALUE", "Ids": ["value-noise"]},
                    ],
                ),
                block("KEY_VALUE_SET", Id="value-noise", EntityTypes=["VALUE"], Relationships=child("w-value-noise")),
                block("WORD", Id="w-key-noise", Text="accrue interest at", Confidence=41.0),
                block("WORD", Id="w-value-noise", Text="1.5%", Confidence=41.0),
            ],
        }
        converter = TextractFileConverter(
            connection=connection,
            client=client,
            feature_types=[TextractFeatureType.FORMS],
            min_confidence=60,
        )

        document = run_converter(converter, {"files": [png_file]})[0]

        assert [form["key"] for form in document["metadata"]["forms"]] == ["Invoice number"]
        assert "accrue interest at" not in document["content"]

    def test_checkbox_selection_status_is_rendered(self, connection, client, png_file):
        client.analyze_document.return_value = {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                block(
                    "KEY_VALUE_SET",
                    Id="key-1",
                    EntityTypes=["KEY"],
                    Relationships=[
                        {"Type": "CHILD", "Ids": ["w-key"]},
                        {"Type": "VALUE", "Ids": ["value-1"]},
                    ],
                ),
                block("KEY_VALUE_SET", Id="value-1", EntityTypes=["VALUE"], Relationships=child("sel-1")),
                block("WORD", Id="w-key", Text="Agreed"),
                block("SELECTION_ELEMENT", Id="sel-1", SelectionStatus="SELECTED"),
            ],
        }
        converter = TextractFileConverter(
            connection=connection, client=client, feature_types=[TextractFeatureType.FORMS]
        )

        assert run_converter(converter, {"files": [png_file]})[0]["metadata"]["forms"][0]["value"] == "[X]"


class TestQueries:
    def test_answers_are_rendered_and_exposed(self, connection, client, png_file):
        client.analyze_document.return_value = {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                block(
                    "QUERY",
                    Id="query-1",
                    Query={"Text": "What is the total?", "Alias": "total"},
                    Relationships=[{"Type": "ANSWER", "Ids": ["answer-1", "answer-2"]}],
                ),
                block("QUERY_RESULT", Id="answer-1", Text="$10.00", Confidence=95.0),
                block("QUERY_RESULT", Id="answer-2", Text="$1.00", Confidence=40.0),
            ],
        }
        converter = TextractFileConverter(
            connection=connection,
            client=client,
            feature_types=[TextractFeatureType.QUERIES],
            queries=[TextractQuery(text="What is the total?", alias="total")],
        )

        documents = run_converter(converter, {"files": [png_file]})

        # The highest-confidence answer wins.
        assert "- **total**: $10.00" in documents[0]["content"]
        assert documents[0]["metadata"]["queries"][0]["answer"] == "$10.00"
        assert client.analyze_document.call_args.kwargs["QueriesConfig"] == {
            "Queries": [{"Text": "What is the total?", "Alias": "total"}]
        }

    def test_queries_require_the_queries_feature(self, connection, client):
        with pytest.raises(ValueError, match="requires TextractFeatureType.QUERIES"):
            TextractFileConverter(
                connection=connection,
                client=client,
                queries=[TextractQuery(text="What is the total?")],
            )

    def test_queries_feature_requires_queries(self, connection, client):
        with pytest.raises(ValueError, match="requires at least one entry in `queries`"):
            TextractFileConverter(connection=connection, client=client, feature_types=[TextractFeatureType.QUERIES])


class TestLayout:
    @pytest.fixture
    def layout_response(self):
        return {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                block("LAYOUT_TITLE", Id="layout-title", Relationships=child("line-title")),
                block("LAYOUT_SECTION_HEADER", Id="layout-header", Relationships=child("line-header")),
                block("LAYOUT_TEXT", Id="layout-text", Relationships=child("line-text")),
                block("LAYOUT_LIST", Id="layout-list", Relationships=child("line-item-1", "line-item-2")),
                block("LAYOUT_FIGURE", Id="layout-figure", Relationships=child("line-caption")),
                block("LAYOUT_FOOTER", Id="layout-footer", Relationships=child("line-footer")),
                line("line-title", "Annual Report"),
                line("line-header", "Revenue"),
                line("line-text", "Revenue grew year over year."),
                line("line-item-1", "North America"),
                line("line-item-2", "Europe"),
                line("line-caption", "Figure 1: revenue by region"),
                line("line-footer", "Page 1 of 9"),
            ],
        }

    def test_layout_regions_become_markdown_structure(self, connection, client, png_file, layout_response):
        client.analyze_document.return_value = layout_response
        converter = TextractFileConverter(
            connection=connection, client=client, feature_types=[TextractFeatureType.LAYOUT]
        )

        content = run_converter(converter, {"files": [png_file]})[0]["content"]

        assert "# Annual Report" in content
        assert "## Revenue" in content
        assert "Revenue grew year over year." in content
        assert "- North America\n- Europe" in content
        assert "<!-- figure: Figure 1: revenue by region -->" in content
        assert "Page 1 of 9" in content

    def test_headers_and_footers_can_be_skipped(self, connection, client, png_file, layout_response):
        client.analyze_document.return_value = layout_response
        converter = TextractFileConverter(
            connection=connection,
            client=client,
            feature_types=[TextractFeatureType.LAYOUT],
            skip_headers_and_footers=True,
        )

        content = run_converter(converter, {"files": [png_file]})[0]["content"]

        assert "Page 1 of 9" not in content
        assert "# Annual Report" in content

    def test_layout_table_is_replaced_by_the_markdown_table(self, connection, client, png_file):
        table_box = geometry(0.1, 0.5, 0.8, 0.2)
        client.analyze_document.return_value = {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                block("LAYOUT_TITLE", Id="layout-title", Relationships=child("line-title")),
                block("LAYOUT_TABLE", Id="layout-table", Geometry=table_box, Relationships=child("line-row")),
                line("line-title", "Annual Report"),
                line("line-row", "Widget 10.00"),
                block("TABLE", Id="table-1", Geometry=table_box, Relationships=child("c11", "c12", "c21", "c22")),
                block("CELL", Id="c11", RowIndex=1, ColumnIndex=1, Relationships=child("w11")),
                block("CELL", Id="c12", RowIndex=1, ColumnIndex=2, Relationships=child("w12")),
                block("CELL", Id="c21", RowIndex=2, ColumnIndex=1, Relationships=child("w21")),
                block("CELL", Id="c22", RowIndex=2, ColumnIndex=2, Relationships=child("w22")),
                block("WORD", Id="w11", Text="Item"),
                block("WORD", Id="w12", Text="Price"),
                block("WORD", Id="w21", Text="Widget"),
                block("WORD", Id="w22", Text="10.00"),
            ],
        }
        converter = TextractFileConverter(
            connection=connection,
            client=client,
            feature_types=[TextractFeatureType.LAYOUT, TextractFeatureType.TABLES],
        )

        content = run_converter(converter, {"files": [png_file]})[0]["content"]

        assert "| Item | Price |" in content
        # The region's raw layout lines are not emitted alongside the table.
        assert content.count("| Widget | 10.00 |") == 1
        assert "Widget 10.00" not in content


class TestSignatures:
    def test_signatures_are_exposed_in_metadata(self, connection, client, png_file):
        client.analyze_document.return_value = {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                line("line-1", "Signed by"),
                block("SIGNATURE", Id="sig-1", Confidence=88.0, Geometry=geometry(0.1, 0.8, 0.2, 0.05)),
            ],
        }
        converter = TextractFileConverter(
            connection=connection, client=client, feature_types=[TextractFeatureType.SIGNATURES]
        )

        signatures = run_converter(converter, {"files": [png_file]})[0]["metadata"]["signatures"]

        assert signatures == [
            {"page": 1, "confidence": 88.0, "bounding_box": {"Left": 0.1, "Top": 0.8, "Width": 0.2, "Height": 0.05}}
        ]


class TestDocumentCreationMode:
    @pytest.fixture
    def two_page_response(self):
        return {
            "DocumentMetadata": {"Pages": 2},
            "JobStatus": "SUCCEEDED",
            "Blocks": [
                line("line-1", "First page", page=1),
                line("line-2", "Second page", page=2),
            ],
        }

    def test_one_doc_per_page(self, connection, client, two_page_response):
        client.start_document_text_detection.return_value = {"JobId": "job-1"}
        client.get_document_text_detection.return_value = two_page_response
        converter = TextractFileConverter(
            connection=connection,
            client=client,
            document_creation_mode=DocumentCreationMode.ONE_DOC_PER_PAGE,
            processing_mode=TextractProcessingMode.ASYNC,
        )

        documents = run_converter(converter, {"s3_objects": [TextractS3Object(bucket="docs", name="report.pdf")]})

        assert [document["content"] for document in documents] == ["First page", "Second page"]
        assert [document["metadata"]["page_number"] for document in documents] == [1, 2]

    def test_one_doc_per_file_joins_pages(self, connection, client, two_page_response):
        client.start_document_text_detection.return_value = {"JobId": "job-1"}
        client.get_document_text_detection.return_value = two_page_response
        converter = TextractFileConverter(
            connection=connection, client=client, processing_mode=TextractProcessingMode.ASYNC
        )

        documents = run_converter(converter, {"s3_objects": [TextractS3Object(bucket="docs", name="report.pdf")]})

        assert len(documents) == 1
        assert documents[0]["content"] == "First page\n\nSecond page"


class TestAsyncJobs:
    def test_s3_object_analysis_paginates_and_stamps_metadata(self, connection, client):
        client.start_document_analysis.return_value = {"JobId": "job-1"}
        client.get_document_analysis.side_effect = [
            {"JobStatus": "SUCCEEDED", "NextToken": "token-1", "Blocks": [line("line-1", "Page one")]},
            {
                "JobStatus": "SUCCEEDED",
                "Blocks": [line("line-2", "Page one continued")],
                "DocumentMetadata": {"Pages": 1},
                "AnalyzeDocumentModelVersion": "1.0",
            },
        ]
        converter = TextractFileConverter(
            connection=connection,
            client=client,
            feature_types=[TextractFeatureType.LAYOUT],
            processing_mode=TextractProcessingMode.ASYNC,
        )

        documents = run_converter(converter, {"s3_objects": [{"bucket": "docs", "name": "contracts/lease.pdf"}]})

        assert documents[0]["content"] == "Page one\nPage one continued"
        metadata = documents[0]["metadata"]
        assert metadata["textract_job_id"] == "job-1"
        assert metadata["textract_processing_mode"] == TextractProcessingMode.ASYNC.value
        assert metadata["textract_source_uri"] == "s3://docs/contracts/lease.pdf"
        assert client.start_document_analysis.call_args.kwargs["DocumentLocation"] == {
            "S3Object": {"Bucket": "docs", "Name": "contracts/lease.pdf"}
        }

    def test_failed_job_surfaces_as_node_failure(self, connection, client):
        client.start_document_text_detection.return_value = {"JobId": "job-1"}
        client.get_document_text_detection.return_value = {
            "JobStatus": "FAILED",
            "StatusMessage": "Document is encrypted",
        }
        converter = TextractFileConverter(
            connection=connection, client=client, processing_mode=TextractProcessingMode.ASYNC
        )

        output_node = Output(id="output_node", depends=[NodeDependency(converter)])
        workflow = Workflow(flow=flows.Flow(nodes=[converter, output_node]), version="1")
        result = workflow.run(input_data={"s3_objects": [{"bucket": "docs", "name": "report.pdf"}]})

        assert result.status == RunnableStatus.FAILURE
        assert "Document is encrypted" in str(result.output[converter.id]["error"])

    def test_in_progress_job_is_polled_until_it_finishes(self, connection, client):
        client.start_document_text_detection.return_value = {"JobId": "job-1"}
        client.get_document_text_detection.side_effect = [
            {"JobStatus": "IN_PROGRESS"},
            {"JobStatus": "SUCCEEDED", "Blocks": [line("line-1", "Done")], "DocumentMetadata": {"Pages": 1}},
        ]
        converter = TextractFileConverter(
            connection=connection,
            client=client,
            processing_mode=TextractProcessingMode.ASYNC,
            poll_interval_seconds=0.01,
        )

        documents = run_converter(converter, {"s3_objects": [{"bucket": "docs", "name": "report.pdf"}]})

        assert documents[0]["content"] == "Done"
        assert client.get_document_text_detection.call_count == 2


class TestMultiPagePdf:
    def test_pages_are_split_locally_when_staging_is_not_possible(self, connection, client, two_page_pdf, monkeypatch):
        def deny_s3(self, service_name="textract"):
            raise RuntimeError("s3:PutObject denied")

        monkeypatch.setattr(AWSTextract, "get_client", deny_s3)
        client.detect_document_text.side_effect = [
            {"DocumentMetadata": {"Pages": 1}, "Blocks": [line("line-1", "First page")]},
            {"DocumentMetadata": {"Pages": 1}, "Blocks": [line("line-2", "Second page")]},
        ]
        converter = TextractFileConverter(
            connection=connection,
            client=client,
            document_creation_mode=DocumentCreationMode.ONE_DOC_PER_PAGE,
        )

        documents = run_converter(converter, {"files": [two_page_pdf]})

        assert client.detect_document_text.call_count == 2
        # Each single-page request reports itself as page 1; numbering is rewritten to the original.
        assert [document["metadata"]["page_number"] for document in documents] == [1, 2]
        assert [document["content"] for document in documents] == ["First page", "Second page"]

    def test_multi_page_pdf_defaults_to_a_per_account_staging_bucket(
        self, connection, client, two_page_pdf, monkeypatch
    ):
        s3_client = MagicMock()
        s3_client.meta.region_name = "us-west-2"
        sts_client = MagicMock()
        sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        monkeypatch.setattr(
            AWSTextract,
            "get_client",
            lambda self, service_name="textract": sts_client if service_name == "sts" else s3_client,
        )
        client.start_document_text_detection.return_value = {"JobId": "job-1"}
        client.get_document_text_detection.return_value = {
            "JobStatus": "SUCCEEDED",
            "DocumentMetadata": {"Pages": 2},
            "Blocks": [line("line-1", "First page", page=1), line("line-2", "Second page", page=2)],
        }
        converter = TextractFileConverter(connection=connection, client=client)

        documents = run_converter(converter, {"files": [two_page_pdf]})

        assert documents[0]["metadata"]["textract_processing_mode"] == "async"
        assert s3_client.put_object.call_args.kwargs["Bucket"] == "dynamiq-textract-staging-123456789012-us-west-2"
        client.detect_document_text.assert_not_called()

    def test_missing_staging_bucket_is_created_in_the_connection_region(
        self, connection, client, two_page_pdf, monkeypatch
    ):
        s3_client = MagicMock()
        s3_client.meta.region_name = "eu-central-1"
        s3_client.head_bucket.side_effect = RuntimeError("404")
        monkeypatch.setattr(AWSTextract, "get_client", lambda self, service_name="textract": s3_client)
        client.start_document_text_detection.return_value = {"JobId": "job-1"}
        client.get_document_text_detection.return_value = {
            "JobStatus": "SUCCEEDED",
            "DocumentMetadata": {"Pages": 2},
            "Blocks": [line("line-1", "First page", page=1)],
        }
        converter = TextractFileConverter(connection=connection, client=client, staging_s3_bucket="staging-bucket")

        run_converter(converter, {"files": [two_page_pdf]})

        s3_client.create_bucket.assert_called_once_with(
            Bucket="staging-bucket",
            CreateBucketConfiguration={"LocationConstraint": "eu-central-1"},
        )
        # A bucket the converter created is locked down and expires anything a killed run leaves.
        assert s3_client.put_public_access_block.call_args.kwargs["Bucket"] == "staging-bucket"
        lifecycle = s3_client.put_bucket_lifecycle_configuration.call_args.kwargs["LifecycleConfiguration"]
        assert lifecycle["Rules"][0]["Filter"] == {"Prefix": "dynamiq/textract/"}
        assert lifecycle["Rules"][0]["Expiration"] == {"Days": 1}

    def test_an_existing_staging_bucket_is_left_untouched(self, connection, client, two_page_pdf, monkeypatch):
        s3_client = MagicMock()
        monkeypatch.setattr(AWSTextract, "get_client", lambda self, service_name="textract": s3_client)
        client.start_document_text_detection.return_value = {"JobId": "job-1"}
        client.get_document_text_detection.return_value = {
            "JobStatus": "SUCCEEDED",
            "DocumentMetadata": {"Pages": 2},
            "Blocks": [line("line-1", "First page", page=1)],
        }
        converter = TextractFileConverter(connection=connection, client=client, staging_s3_bucket="staging-bucket")

        run_converter(converter, {"files": [two_page_pdf]})

        # head_bucket succeeded, so the bucket is pre-existing and its settings are left alone.
        s3_client.create_bucket.assert_not_called()
        s3_client.put_bucket_lifecycle_configuration.assert_not_called()
        # The bucket is reused across runs, never deleted - only the staged document goes away.
        s3_client.delete_bucket.assert_not_called()
        s3_client.delete_object.assert_called_once()

    def test_creation_can_be_disabled_for_out_of_band_buckets(self, connection, client, two_page_pdf, monkeypatch):
        s3_client = MagicMock()
        s3_client.head_bucket.side_effect = RuntimeError("404")
        monkeypatch.setattr(AWSTextract, "get_client", lambda self, service_name="textract": s3_client)
        client.start_document_text_detection.return_value = {"JobId": "job-1"}
        client.get_document_text_detection.return_value = {
            "JobStatus": "SUCCEEDED",
            "DocumentMetadata": {"Pages": 2},
            "Blocks": [line("line-1", "First page", page=1)],
        }
        converter = TextractFileConverter(
            connection=connection,
            client=client,
            staging_s3_bucket="staging-bucket",
            create_staging_bucket=False,
        )

        run_converter(converter, {"files": [two_page_pdf]})

        s3_client.head_bucket.assert_not_called()
        s3_client.create_bucket.assert_not_called()
        s3_client.put_object.assert_called_once()

    def test_staging_bucket_routes_multi_page_pdf_through_s3(self, connection, client, two_page_pdf, monkeypatch):
        s3_client = MagicMock()
        monkeypatch.setattr(AWSTextract, "get_client", lambda self, service_name="textract": s3_client)
        client.start_document_text_detection.return_value = {"JobId": "job-1"}
        client.get_document_text_detection.return_value = {
            "JobStatus": "SUCCEEDED",
            "DocumentMetadata": {"Pages": 2},
            "Blocks": [line("line-1", "First page", page=1), line("line-2", "Second page", page=2)],
        }
        converter = TextractFileConverter(connection=connection, client=client, staging_s3_bucket="staging-bucket")

        documents = run_converter(converter, {"files": [two_page_pdf]})

        assert documents[0]["content"] == "First page\n\nSecond page"
        client.detect_document_text.assert_not_called()

        staged_key = s3_client.put_object.call_args.kwargs["Key"]
        assert staged_key.startswith("dynamiq/textract/")
        assert staged_key.endswith("report.pdf")
        # The staged copy is removed once the job completes.
        s3_client.delete_object.assert_called_once_with(Bucket="staging-bucket", Key=staged_key)

    def test_staged_object_is_removed_even_when_the_job_fails(self, connection, client, two_page_pdf, monkeypatch):
        s3_client = MagicMock()
        monkeypatch.setattr(AWSTextract, "get_client", lambda self, service_name="textract": s3_client)
        client.start_document_text_detection.side_effect = RuntimeError("throttled")
        converter = TextractFileConverter(connection=connection, client=client, staging_s3_bucket="staging-bucket")

        output_node = Output(id="output_node", depends=[NodeDependency(converter)])
        workflow = Workflow(flow=flows.Flow(nodes=[converter, output_node]), version="1")
        result = workflow.run(input_data={"files": [two_page_pdf]})

        assert result.status == RunnableStatus.FAILURE
        s3_client.delete_object.assert_called_once()

    def test_oversized_file_without_staging_bucket_is_rejected(self, connection, client):
        file = BytesIO(PNG_HEADER + b"x" * (11 * 1024 * 1024))
        file.name = "huge.png"
        converter = TextractFileConverter(connection=connection, client=client)

        output_node = Output(id="output_node", depends=[NodeDependency(converter)])
        workflow = Workflow(flow=flows.Flow(nodes=[converter, output_node]), version="1")
        result = workflow.run(input_data={"files": [file]})

        assert result.status == RunnableStatus.FAILURE
        client.detect_document_text.assert_not_called()


class TestSerialization:
    def test_node_round_trips_through_to_dict(self, connection):
        converter = TextractFileConverter(
            connection=connection,
            feature_types=[TextractFeatureType.LAYOUT, TextractFeatureType.TABLES],
            staging_s3_bucket="staging-bucket",
        )

        data = converter.to_dict()

        assert data["type"] == "dynamiq.nodes.converters.TextractFileConverter"
        assert data["feature_types"] == [TextractFeatureType.LAYOUT, TextractFeatureType.TABLES]
        assert data["staging_s3_bucket"] == "staging-bucket"
        assert data["connection"]["type"] == "dynamiq.connections.AWSTextract"
        assert "file_converter" not in data

    def test_an_auto_derived_staging_bucket_does_not_leak_into_the_config(
        self, connection, client, two_page_pdf, monkeypatch
    ):
        s3_client = MagicMock()
        s3_client.meta.region_name = "us-east-1"
        sts_client = MagicMock()
        sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        monkeypatch.setattr(
            AWSTextract,
            "get_client",
            lambda self, service_name="textract": sts_client if service_name == "sts" else s3_client,
        )
        client.start_document_text_detection.return_value = {"JobId": "job-1"}
        client.get_document_text_detection.return_value = {
            "JobStatus": "SUCCEEDED",
            "DocumentMetadata": {"Pages": 2},
            "Blocks": [line("line-1", "First page", page=1)],
        }
        converter = TextractFileConverter(connection=connection, client=client)

        run_converter(converter, {"files": [two_page_pdf]})

        # Re-serializing a converter that ran must not pin it to one account's bucket.
        assert converter.to_dict()["staging_s3_bucket"] is None
        assert converter.file_converter.staging_s3_bucket is None


class TestStagingPrefix:
    @pytest.mark.parametrize("prefix", ["/staged/textract/", "staged/textract"])
    def test_surrounding_slashes_are_normalized(self, connection, client, prefix):
        converter = TextractFileConverter(connection=connection, client=client, staging_s3_prefix=prefix)

        assert converter.staging_s3_prefix == "staged/textract"
        assert converter.file_converter.staging_s3_prefix == "staged/textract"

    @pytest.mark.parametrize("prefix", ["", "/"])
    def test_an_empty_prefix_is_rejected(self, connection, client, prefix):
        with pytest.raises(ValueError, match="staging_s3_prefix"):
            TextractFileConverter(connection=connection, client=client, staging_s3_prefix=prefix)
