from io import BytesIO

import pytest
from pypdf import PdfWriter

from dynamiq.components.converters.pypdf import PyPDFFileConverter
from dynamiq.types import DocumentCreationMode


def build_pdf_with_metadata() -> BytesIO:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata(
        {
            "/Producer": "Prince 16",
            "/CreationDate": "D:20260721120000+02'00'",
            "/Custom Field": "custom value",
        }
    )
    output = BytesIO()
    writer.write(output)
    output.seek(0)
    output.name = "metadata.pdf"
    return output


@pytest.mark.parametrize(
    ("document_creation_mode", "expected_document_count"),
    [
        (DocumentCreationMode.ONE_DOC_PER_FILE, 1),
        (DocumentCreationMode.ONE_DOC_PER_PAGE, 2),
    ],
)
def test_pdf_metadata_is_normalized_in_all_document_creation_modes(
    document_creation_mode: DocumentCreationMode,
    expected_document_count: int,
):
    converter = PyPDFFileConverter(document_creation_mode=document_creation_mode)

    documents = converter.run(
        files=[build_pdf_with_metadata()],
        metadata={"request_id": "request-1"},
    )["documents"]

    assert len(documents) == expected_document_count
    for document in documents:
        assert document.metadata["request_id"] == "request-1"
        assert document.metadata["file_path"] == "metadata.pdf"
        assert document.metadata["pdf_producer"] == "Prince 16"
        assert document.metadata["pdf_creation_date"] == "D:20260721120000+02'00'"
        assert document.metadata["pdf_custom_field"] == "custom value"
        assert "/Producer" not in document.metadata


def test_pdf_metadata_normalization_disambiguates_colliding_keys():
    normalized = PyPDFFileConverter._normalize_pdf_metadata(
        {
            "/Custom Field": "first",
            "/Custom-Field": "second",
        }
    )

    assert normalized == {
        "pdf_custom_field": "first",
        "pdf_custom_field_2": "second",
    }
