import os
from io import BytesIO
from pathlib import Path

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.pypdf import PyPDFConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document


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


def test_workflow_with_pypdf_converter():
    wf_pypdf = Workflow(
        flow=Flow(nodes=[PyPDFConverter()]),
    )
    file = BytesIO(
        b"%PDF-1.7\n\n1 0 obj  % entry point\n<<\n  /Type /Catalog\n  /Pages 2 0 R\n>>\nendobj\n\n2 0 obj\n<<\n  "
        b"/Type /Pages\n  /MediaBox [ 0 0 200 200 ]\n  /Count 1\n  /Kids [ 3 0 R ]\n>>\nendobj\n\n3 0 obj\n<<\n  "
        b"/Type /Page\n  /Parent 2 0 R\n  /Resources <<\n    /Font <<\n      /F1 4 0 R \n    >>\n  >>\n  /Contents "
        b"5 0 R\n>>\nendobj\n\n4 0 obj\n<<\n  /Type /Font\n  /Subtype /Type1\n  /BaseFont /Times-Roman\n>>\nendobj\n\n"
        b"5 0 obj  % page content\n<<\n  /Length 44\n>>\nstream\nBT\n70 50 TD\n/F1 12 Tf\n(Hello, world!) "
        b"Tj\nET\nendstream\nendobj\n\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n "
        b"\n0000000173 00000 n \n0000000301 00000 n \n0000000380 00000 n \ntrailer\n<<\n  /Size 6\n  /Root 1 0 "
        b"R\n>>\nstartxref\n492\n%%EOF"
    )
    file.name = "mock.pdf"
    input_data = {"files": [file]}
    response = wf_pypdf.run(input_data=input_data)
    document_id = response.output[next(iter(response.output))]["output"]["documents"][0]["id"]
    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"documents": [Document(id=document_id, content="Hello, world!", metadata={"file_path": file.name})]},
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {wf_pypdf.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )

    node_id = wf_pypdf.flow.nodes[0].id
    assert len(response.output[node_id]["output"]["documents"]) == 1


def test_workflow_with_pypdf_converter_parsing_error(
    workflow_with_pypdf_converter_and_output, pypdf_converter, output_node
):
    file = BytesIO(b"%PDF-corrupted-content")
    file.name = "corrupted.pdf"
    input_data = {"files": [file]}

    response = workflow_with_pypdf_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pypdf_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pypdf_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_pypdf_converter_file_not_found(
    workflow_with_pypdf_converter_and_output, pypdf_converter, output_node
):
    non_existent_path = str(Path("/tmp") / f"non_existent_file_{os.getpid()}.pdf")
    input_data = {"file_paths": [non_existent_path]}

    response = workflow_with_pypdf_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pypdf_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pypdf_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_pypdf_converter_empty_file(
    workflow_with_pypdf_converter_and_output, pypdf_converter, output_node
):
    empty_file = BytesIO(b"")
    empty_file.name = "empty.pdf"
    input_data = {"files": [empty_file]}

    response = workflow_with_pypdf_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pypdf_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pypdf_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_pypdf_converter_no_pages(workflow_with_pypdf_converter_and_output, pypdf_converter, output_node):
    file = BytesIO(b"%PDF-1.7\n\n1 0 obj\n<<\n  /Type /Catalog\n>>\nendobj\n\ntrailer\n<<\n  /Root 1 0 R\n>>\n%%EOF")
    file.name = "no_pages.pdf"
    input_data = {"files": [file]}

    response = workflow_with_pypdf_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    if response.output[pypdf_converter.id]["status"] == RunnableStatus.FAILURE.value:
        assert "error" in response.output[pypdf_converter.id]
        assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value
    else:
        assert "output" in response.output[pypdf_converter.id]
        assert "documents" in response.output[pypdf_converter.id]["output"]
        assert len(response.output[pypdf_converter.id]["output"]["documents"]) >= 0


def test_workflow_with_pypdf_converter_unsupported_file(
    workflow_with_pypdf_converter_and_output, pypdf_converter, output_node
):
    wrong_file = BytesIO(b"This is not a PDF file, just plain text")
    wrong_file.name = "text.txt"
    input_data = {"files": [wrong_file]}

    response = workflow_with_pypdf_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pypdf_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pypdf_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value
