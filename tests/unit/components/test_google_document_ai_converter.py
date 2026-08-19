import mimetypes
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from google.cloud import documentai
from pydantic import ValidationError
from pypdf import PdfWriter

from dynamiq.components.converters.google_document_ai import (
    MAX_PAGES_PER_REQUEST_IMAGELESS,
    MAX_PAGES_PER_REQUEST_WITH_IMAGES,
    DocumentAIMimeType,
    GoogleDocumentAIFileConverter,
)
from dynamiq.connections import GoogleDocumentAI
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
def connection():
    return GoogleDocumentAI(project_id="proj", location="eu")


@pytest.fixture
def client():
    client = MagicMock(spec=documentai.DocumentProcessorServiceClient)
    client.processor_path.return_value = PROCESSOR_PATH
    client.processor_version_path.return_value = f"{PROCESSOR_PATH}/processorVersions/pretrained-ocr-v2.0"
    client.process_document.return_value = build_response(["PAGE ONE\n", "PAGE TWO\n"])
    return client


@pytest.fixture
def converter(connection, client):
    return GoogleDocumentAIFileConverter(connection=connection, client=client, processor_id="abc123")


class TestTextExtraction:
    def test_one_doc_per_file_concatenates_page_texts(self, converter):
        documents = converter.run(files=[build_pdf(2)])["documents"]

        assert len(documents) == 1
        assert documents[0].content == "PAGE ONE\n\n\nPAGE TWO\n"

    def test_one_doc_per_page_creates_one_document_per_page(self, connection, client):
        converter = GoogleDocumentAIFileConverter(
            connection=connection,
            client=client,
            processor_id="abc123",
            document_creation_mode=DocumentCreationMode.ONE_DOC_PER_PAGE,
        )

        documents = converter.run(files=[build_pdf(2)])["documents"]

        assert [document.content for document in documents] == ["PAGE ONE\n", "PAGE TWO\n"]
        assert [document.metadata["page_number"] for document in documents] == [1, 2]

    def test_text_without_page_anchors_is_kept(self, converter, client):
        """A processor may return text with no page layout; the text must not be dropped."""
        client.process_document.return_value = documentai.ProcessResponse(
            document=documentai.Document(text="ANCHORLESS")
        )

        documents = converter.run(files=[build_pdf(1)])["documents"]

        assert documents[0].content == "ANCHORLESS"

    def test_pages_with_no_text_yield_empty_strings(self, converter, client):
        client.process_document.return_value = build_response(["", "SECOND\n"])

        documents = converter.run(files=[build_pdf(2)])["documents"]

        assert documents[0].content == "\n\nSECOND\n"

    def test_custom_page_separator_is_used(self, connection, client):
        converter = GoogleDocumentAIFileConverter(
            connection=connection, client=client, processor_id="abc123", page_separator="\n---\n"
        )

        documents = converter.run(files=[build_pdf(2)])["documents"]

        assert documents[0].content == "PAGE ONE\n\n---\nPAGE TWO\n"


class TestMetadata:
    def test_metadata_carries_file_and_processor_details(self, converter):
        documents = converter.run(files=[build_pdf(2)], metadata={"request_id": "request-1"})["documents"]

        metadata = documents[0].metadata
        assert metadata["request_id"] == "request-1"
        assert metadata["file_path"] == "file.pdf"
        assert metadata["document_ai_processor_id"] == "abc123"
        assert metadata["document_ai_pages"] == 2

    def test_processor_version_is_recorded_only_when_pinned(self, connection, client):
        without_version = GoogleDocumentAIFileConverter(connection=connection, client=client, processor_id="abc123")
        with_version = GoogleDocumentAIFileConverter(
            connection=connection,
            client=client,
            processor_id="abc123",
            processor_version="pretrained-ocr-v2.0",
        )

        assert "document_ai_processor_version" not in without_version.run(files=[build_pdf(1)])["documents"][0].metadata
        assert (
            with_version.run(files=[build_pdf(1)])["documents"][0].metadata["document_ai_processor_version"]
            == "pretrained-ocr-v2.0"
        )

    def test_file_path_input_is_recorded(self, converter, tmp_path):
        path = tmp_path / "from-disk.pdf"
        path.write_bytes(build_pdf(2).getvalue())

        documents = converter.run(file_paths=[str(path)])["documents"]

        assert documents[0].metadata["file_path"] == str(path)


class TestProcessorName:
    def test_processor_path_is_used_without_a_version(self, converter, client):
        assert converter.processor_name == PROCESSOR_PATH
        client.processor_path.assert_called_once_with("proj", "eu", "abc123")

    def test_processor_version_path_is_used_with_a_version(self, connection, client):
        converter = GoogleDocumentAIFileConverter(
            connection=connection,
            client=client,
            processor_id="abc123",
            processor_version="pretrained-ocr-v2.0",
        )

        assert converter.processor_name.endswith("/processorVersions/pretrained-ocr-v2.0")
        client.processor_version_path.assert_called_once_with("proj", "eu", "abc123", "pretrained-ocr-v2.0")

    @pytest.mark.parametrize("processor_id", ["", "   "], ids=["empty", "blank"])
    def test_blank_processor_id_is_rejected(self, connection, client, processor_id):
        with pytest.raises(ValidationError, match="processor_id"):
            GoogleDocumentAIFileConverter(connection=connection, client=client, processor_id=processor_id)

    def test_processor_id_is_stripped(self, connection, client):
        """A processor id copied out of the console tends to carry whitespace."""
        converter = GoogleDocumentAIFileConverter(connection=connection, client=client, processor_id="  abc123\n")

        assert converter.processor_id == "abc123"


class TestRequestOptions:
    def test_request_carries_the_ocr_options(self, converter, client):
        converter.run(files=[build_pdf(1)])

        request = client.process_document.call_args.kwargs["request"]
        assert request.name == PROCESSOR_PATH
        assert request.imageless_mode is True
        assert request.process_options.ocr_config.enable_native_pdf_parsing is True
        assert request.raw_document.mime_type == "application/pdf"

    def test_options_can_be_disabled(self, connection, client):
        converter = GoogleDocumentAIFileConverter(
            connection=connection,
            client=client,
            processor_id="abc123",
            enable_native_pdf_parsing=False,
            imageless_mode=False,
        )

        converter.run(files=[build_pdf(1)])

        request = client.process_document.call_args.kwargs["request"]
        assert request.imageless_mode is False
        assert request.process_options.ocr_config.enable_native_pdf_parsing is False

    def test_mime_type_is_guessed_from_the_filename(self, converter, client):
        image = BytesIO(b"not-really-a-png")
        image.name = "scan.png"

        converter.run(files=[image])

        assert client.process_document.call_args.kwargs["request"].raw_document.mime_type == "image/png"


class TestMimeTypes:
    @pytest.mark.parametrize("mime_type", list(DocumentAIMimeType), ids=lambda value: value.name)
    def test_every_supported_type_is_declared_verbatim(self, converter, client, mime_type):
        extension = mimetypes.guess_extension(mime_type.value)
        file = BytesIO(b"file-bytes")
        file.name = f"scan{extension}"

        converter.run(files=[file])

        assert client.process_document.call_args.kwargs["request"].raw_document.mime_type == mime_type.value

    @pytest.mark.parametrize("filename", ["notes.docx", "notes.txt", "sheet.csv"])
    def test_unsupported_type_is_rejected_before_the_api_call(self, converter, client, filename):
        file = BytesIO(b"file-bytes")
        file.name = filename

        with pytest.raises(ValueError, match="unsupported MIME type"):
            converter.run(files=[file])

        client.process_document.assert_not_called()

    def test_unguessable_type_falls_back_to_pdf(self, converter, client, tmp_path):
        """Document AI's own error is more useful than a local guess from a bare filename."""
        path = tmp_path / "scan-without-extension"
        path.write_bytes(build_pdf(1).getvalue())

        converter.run(file_paths=[str(path)])

        assert client.process_document.call_args.kwargs["request"].raw_document.mime_type == "application/pdf"


class TestPageLimitSplitting:
    @pytest.mark.parametrize(
        ("imageless_mode", "expected_limit"),
        [(True, MAX_PAGES_PER_REQUEST_IMAGELESS), (False, MAX_PAGES_PER_REQUEST_WITH_IMAGES)],
        ids=["imageless", "with_images"],
    )
    def test_page_limit_depends_on_imageless_mode(self, connection, client, imageless_mode, expected_limit):
        converter = GoogleDocumentAIFileConverter(
            connection=connection, client=client, processor_id="abc123", imageless_mode=imageless_mode
        )

        assert converter.max_pages_per_request == expected_limit

    def test_pdf_within_the_limit_is_sent_in_one_request(self, converter, client):
        converter.run(files=[build_pdf(MAX_PAGES_PER_REQUEST_IMAGELESS)])

        assert client.process_document.call_count == 1

    def test_oversized_pdf_is_split_into_batches(self, converter, client):
        client.process_document.return_value = build_response(["X"])

        converter.run(files=[build_pdf(MAX_PAGES_PER_REQUEST_IMAGELESS + 1)])

        assert client.process_document.call_count == 2

    def test_split_batches_respect_the_page_limit(self, converter, client):
        from pypdf import PdfReader

        client.process_document.return_value = build_response(["X"])

        converter.run(files=[build_pdf(MAX_PAGES_PER_REQUEST_IMAGELESS * 2 + 3)])

        batch_page_counts = [
            len(PdfReader(BytesIO(call.kwargs["request"].raw_document.content)).pages)
            for call in client.process_document.call_args_list
        ]
        assert batch_page_counts == [MAX_PAGES_PER_REQUEST_IMAGELESS, MAX_PAGES_PER_REQUEST_IMAGELESS, 3]

    def test_split_batch_texts_are_concatenated_in_page_order(self, converter, client):
        client.process_document.side_effect = [
            build_response([f"BATCH1-P{page}\n" for page in range(1, MAX_PAGES_PER_REQUEST_IMAGELESS + 1)]),
            build_response(["BATCH2-P1\n"]),
        ]

        documents = converter.run(files=[build_pdf(MAX_PAGES_PER_REQUEST_IMAGELESS + 1)])["documents"]

        assert documents[0].content.startswith("BATCH1-P1\n")
        assert documents[0].content.endswith("BATCH2-P1\n")
        assert documents[0].metadata["document_ai_pages"] == MAX_PAGES_PER_REQUEST_IMAGELESS + 1

    def test_non_pdf_input_is_never_split(self, converter, client):
        image = BytesIO(b"not-really-a-tiff")
        image.name = "scan.tiff"

        converter.run(files=[image])

        assert client.process_document.call_count == 1
        assert client.process_document.call_args.kwargs["request"].raw_document.content == b"not-really-a-tiff"

    def test_unreadable_pdf_is_sent_unsplit(self, converter, client):
        broken = BytesIO(b"%PDF-1.7 truncated")
        broken.name = "broken.pdf"

        converter.run(files=[broken])

        assert client.process_document.call_count == 1
        assert client.process_document.call_args.kwargs["request"].raw_document.content == b"%PDF-1.7 truncated"


class TestInputValidation:
    def test_empty_file_is_rejected(self, converter):
        empty = BytesIO(b"")
        empty.name = "empty.pdf"

        with pytest.raises(ValueError, match="is empty"):
            converter.run(files=[empty])

    def test_missing_input_is_rejected(self, converter):
        with pytest.raises(ValueError, match="No input provided"):
            converter.run()

    def test_client_is_built_from_the_connection_once_when_not_supplied(self, connection):
        converter = GoogleDocumentAIFileConverter(connection=connection, processor_id="abc123")

        with patch.object(GoogleDocumentAI, "connect", return_value="a-client") as connect:
            assert converter.documentai_client == "a-client"
            assert converter.documentai_client == "a-client"

        connect.assert_called_once()
