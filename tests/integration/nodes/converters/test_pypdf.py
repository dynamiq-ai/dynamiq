from io import BytesIO
from pathlib import Path

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.pypdf import PyPDFConverter, PyPDFConverterInputSchema
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document


def write_pdf_to_path(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return str(path)


def write_pdf_to_bytesio(content: bytes, filename: str = "file.pdf") -> BytesIO:
    pdf_buffer = BytesIO(content)
    pdf_buffer.name = filename
    return pdf_buffer


@pytest.fixture
def pypdf_converter():
    return PyPDFConverter(
        id="pypdf_converter",
        name="Test PyPDF Converter",
    )


@pytest.fixture
def output_node(pypdf_converter):
    return Output(id="output_node", depends=[NodeDependency(pypdf_converter)])


@pytest.fixture
def workflow_with_pypdf_converter_and_output(pypdf_converter, output_node):
    return Workflow(
        id="test_workflow",
        flow=Flow(
            nodes=[pypdf_converter, output_node],
        ),
        version="1",
    )


@pytest.fixture
def valid_pdf_content():
    return (
        b"%PDF-1.7\n\n1 0 obj  % entry point\n<<\n  /Type /Catalog\n  /Pages 2 0 R\n>>\nendobj\n\n2 0 obj\n<<\n  "
        b"/Type /Pages\n  /MediaBox [ 0 0 200 200 ]\n  /Count 1\n  /Kids [ 3 0 R ]\n>>\nendobj\n\n3 0 obj\n<<\n  "
        b"/Type /Page\n  /Parent 2 0 R\n  /Resources <<\n    /Font <<\n      /F1 4 0 R \n    >>\n  >>\n  /Contents "
        b"5 0 R\n>>\nendobj\n\n4 0 obj\n<<\n  /Type /Font\n  /Subtype /Type1\n  /BaseFont /Times-Roman\n>>\nendobj\n\n"
        b"5 0 obj  % page content\n<<\n  /Length 44\n>>\nstream\nBT\n70 50 TD\n/F1 12 Tf\n(Hello, world!) "
        b"Tj\nET\nendstream\nendobj\n\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n "
        b"\n0000000173 00000 n \n0000000301 00000 n \n0000000380 00000 n \ntrailer\n<<\n  /Size 6\n  /Root 1 0 "
        b"R\n>>\nstartxref\n492\n%%EOF"
    )


@pytest.fixture
def valid_pdf_file_path(tmp_path, valid_pdf_content):
    file_path = tmp_path / "mock.pdf"
    return write_pdf_to_path(file_path, valid_pdf_content)


@pytest.fixture
def valid_pdf_bytesio(valid_pdf_content):
    return write_pdf_to_bytesio(valid_pdf_content, "mock.pdf")


@pytest.fixture
def pdf_no_pages_content():
    return b"%PDF-1.7\n\n1 0 obj\n<<\n  /Type /Catalog\n>>\nendobj\n\ntrailer\n<<\n  /Root 1 0 R\n>>\n%%EOF"


@pytest.fixture
def pdf_no_pages_file_path(tmp_path, pdf_no_pages_content):
    file_path = tmp_path / "no_pages.pdf"
    return write_pdf_to_path(file_path, pdf_no_pages_content)


@pytest.fixture
def pdf_no_pages_bytesio(pdf_no_pages_content):
    return write_pdf_to_bytesio(pdf_no_pages_content, "no_pages.pdf")


@pytest.fixture
def corrupted_pdf_content():
    return b"%PDF-corrupted-content"


@pytest.fixture
def corrupted_pdf_file_path(tmp_path, corrupted_pdf_content):
    file_path = tmp_path / "corrupted.pdf"
    return write_pdf_to_path(file_path, corrupted_pdf_content)


@pytest.fixture
def corrupted_pdf_bytesio(corrupted_pdf_content):
    return write_pdf_to_bytesio(corrupted_pdf_content, "corrupted.pdf")


@pytest.fixture
def empty_pdf_file_path(tmp_path):
    empty_file_path = tmp_path / "empty.pdf"
    empty_file_path.touch()
    return str(empty_file_path)


@pytest.fixture
def empty_pdf_bytesio():
    empty_buffer = BytesIO()
    empty_buffer.name = "empty.pdf"
    return empty_buffer


@pytest.fixture
def non_existent_pdf_file(tmp_path):
    return str(tmp_path / "non_existent_file.pdf")


@pytest.fixture
def unsupported_content():
    return b"This is not a PDF file, just plain text"


@pytest.fixture
def unsupported_file_path(tmp_path, unsupported_content):
    file_path = tmp_path / "text.txt"
    return write_pdf_to_path(file_path, unsupported_content)


@pytest.fixture
def unsupported_bytesio(unsupported_content):
    return write_pdf_to_bytesio(unsupported_content, "text.txt")


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "valid_pdf_file_path"),
        ("files", "valid_pdf_bytesio"),
    ],
)
def test_workflow_with_pypdf_converter(request, input_type, input_fixture):
    pdf_input = request.getfixturevalue(input_fixture)
    wf_pypdf = Workflow(
        flow=Flow(nodes=[PyPDFConverter()]),
    )
    input_data = {input_type: [pdf_input]}
    response = wf_pypdf.run(input_data=input_data)
    document_id = response.output[next(iter(response.output))]["output"]["documents"][0]["id"]

    expected_path = pdf_input if input_type == "file_paths" else pdf_input.name

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=dict(PyPDFConverterInputSchema(**input_data)),
        output={
            "documents": [Document(id=document_id, content="Hello, world!", metadata={"file_path": expected_path})]
        },
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {wf_pypdf.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )

    node_id = wf_pypdf.flow.nodes[0].id
    assert len(response.output[node_id]["output"]["documents"]) == 1


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "corrupted_pdf_file_path"),
        ("files", "corrupted_pdf_bytesio"),
    ],
)
def test_workflow_with_pypdf_converter_parsing_error(
    request, workflow_with_pypdf_converter_and_output, pypdf_converter, output_node, input_type, input_fixture
):
    pdf_input = request.getfixturevalue(input_fixture)
    input_data = {input_type: [pdf_input]}

    response = workflow_with_pypdf_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.FAILURE
    assert response.output[pypdf_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pypdf_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_pypdf_converter_file_not_found(
    workflow_with_pypdf_converter_and_output, pypdf_converter, output_node, non_existent_pdf_file
):
    input_data = {"file_paths": [non_existent_pdf_file]}

    response = workflow_with_pypdf_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.FAILURE
    assert response.output[pypdf_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pypdf_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "empty_pdf_file_path"),
        ("files", "empty_pdf_bytesio"),
    ],
)
def test_workflow_with_pypdf_converter_empty_file(
    request, workflow_with_pypdf_converter_and_output, pypdf_converter, output_node, input_type, input_fixture
):
    pdf_input = request.getfixturevalue(input_fixture)
    input_data = {input_type: [pdf_input]}

    response = workflow_with_pypdf_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.FAILURE
    assert response.output[pypdf_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pypdf_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "pdf_no_pages_file_path"),
        ("files", "pdf_no_pages_bytesio"),
    ],
)
def test_workflow_with_pypdf_converter_no_pages(
    request, workflow_with_pypdf_converter_and_output, pypdf_converter, output_node, input_type, input_fixture
):
    pdf_input = request.getfixturevalue(input_fixture)
    input_data = {input_type: [pdf_input]}

    response = workflow_with_pypdf_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.FAILURE
    assert response.output[pypdf_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pypdf_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "unsupported_file_path"),
        ("files", "unsupported_bytesio"),
    ],
)
def test_workflow_with_pypdf_converter_unsupported_file(
    request, workflow_with_pypdf_converter_and_output, pypdf_converter, output_node, input_type, input_fixture
):
    pdf_input = request.getfixturevalue(input_fixture)
    input_data = {input_type: [pdf_input]}

    response = workflow_with_pypdf_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.FAILURE
    assert response.output[pypdf_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pypdf_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value
