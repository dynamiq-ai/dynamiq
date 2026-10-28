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


@pytest.mark.parametrize("extraction_mode", ["plain", "layout"])
def test_pdf_pages_are_separated_in_one_document(extraction_mode):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    output = BytesIO()
    pdf = canvas.Canvas(output)
    for text in ("Filter QF-03", "Page two"):
        pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    output.seek(0)
    output.name = "two_pages.pdf"

    documents = PyPDFFileConverter(extraction_mode=extraction_mode).run(files=[output])["documents"]

    assert "QF-03Page" not in documents[0].content
    assert documents[0].content.split() == ["Filter", "QF-03", "Page", "two"]


def test_unreadable_pdf_is_skipped(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    valid = tmp_path / "valid.pdf"
    valid.write_bytes(build_pdf_with_metadata().getvalue())

    documents = PyPDFFileConverter().run(file_paths=[str(broken), str(valid)])["documents"]

    assert [document.metadata["file_path"] for document in documents] == [str(valid)]


def test_only_unreadable_pdfs_raise():
    broken = BytesIO(b"not a pdf")
    broken.name = "broken.pdf"

    with pytest.raises(ValueError, match="No documents were created"):
        PyPDFFileConverter().run(files=[broken])
