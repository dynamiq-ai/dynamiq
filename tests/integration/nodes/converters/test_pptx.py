import os
from io import BytesIO
from pathlib import Path

import pytest
from pptx import Presentation

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.pptx import PPTXFileConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document


@pytest.fixture
def pptx_converter():
    return PPTXFileConverter(
        id="pptx_converter",
        name="Test PPTX Converter",
    )


@pytest.fixture
def output_node(pptx_converter):
    return Output(id="output_node", depends=[NodeDependency(pptx_converter)])


@pytest.fixture
def workflow_with_pptx_converter_and_output(pptx_converter, output_node):
    return Workflow(
        id="test_workflow",
        flow=Flow(
            nodes=[pptx_converter, output_node],
        ),
        version="1",
    )


def test_workflow_with_pptx_converter():
    content = "Hello, World!"
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    title.text = content

    pptx_converter = PPTXFileConverter()
    wf_pptx = Workflow(flow=Flow(nodes=[pptx_converter]))
    file = BytesIO()
    prs.save(file)
    file.name = "mock.pptx"
    input_data = {"files": [file]}

    response = wf_pptx.run(input_data=input_data)
    document_id = response.output[next(iter(response.output))]["output"]["documents"][0]["id"]
    pptx_converter_expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"documents": [Document(id=document_id, content=content, metadata={"file_path": file.name})]},
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {pptx_converter.id: pptx_converter_expected_result}
    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)


def test_workflow_with_pptx_converter_parsing_error(
    workflow_with_pptx_converter_and_output, pptx_converter, output_node
):
    file = BytesIO(b"This is not a valid PPTX file content")
    file.name = "invalid.pptx"
    input_data = {"files": [file]}

    response = workflow_with_pptx_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pptx_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pptx_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_pptx_converter_file_not_found(
    workflow_with_pptx_converter_and_output, pptx_converter, output_node
):
    non_existent_path = str(Path("/tmp") / f"non_existent_file_{os.getpid()}.pptx")
    input_data = {"file_paths": [non_existent_path]}

    response = workflow_with_pptx_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pptx_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "No files found in the provided paths" in response.output[pptx_converter.id]["error"]["message"]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_pptx_converter_empty_file(workflow_with_pptx_converter_and_output, pptx_converter, output_node):
    empty_file = BytesIO(b"")
    empty_file.name = "empty.pptx"
    input_data = {"files": [empty_file]}

    response = workflow_with_pptx_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pptx_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pptx_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_pptx_converter_empty_presentation(
    workflow_with_pptx_converter_and_output, pptx_converter, output_node
):
    prs = Presentation()
    file = BytesIO()
    prs.save(file)
    file.name = "empty_presentation.pptx"
    file.seek(0)

    input_data = {"files": [file]}
    response = workflow_with_pptx_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pptx_converter.id]["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in response.output[pptx_converter.id]["output"]
    documents = response.output[pptx_converter.id]["output"]["documents"]
    assert len(documents) == 1
    assert documents[0]["content"] == ""
    assert documents[0]["metadata"]["file_path"] == "empty_presentation.pptx"
    assert response.output[output_node.id]["status"] == RunnableStatus.SUCCESS.value


def test_workflow_with_pptx_converter_unsupported_file(
    workflow_with_pptx_converter_and_output, pptx_converter, output_node
):
    wrong_file = BytesIO(b"This is not a PPTX file, just plain text")
    wrong_file.name = "text.txt"
    input_data = {"files": [wrong_file]}

    response = workflow_with_pptx_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pptx_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pptx_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value
