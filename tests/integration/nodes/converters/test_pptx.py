from io import BytesIO

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


@pytest.fixture
def valid_pptx_file():
    content = "Hello, World!"
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    title.text = content

    file = BytesIO()
    prs.save(file)
    file.name = "mock.pptx"
    file.seek(0)
    return file, content


@pytest.fixture
def empty_pptx_presentation():
    prs = Presentation()
    file = BytesIO()
    prs.save(file)
    file.name = "empty_presentation.pptx"
    file.seek(0)
    return file


@pytest.fixture
def empty_pptx_file(tmp_path):
    empty_file_path = tmp_path / "empty.pptx"
    empty_file_path.touch()
    return str(empty_file_path)


@pytest.fixture
def invalid_pptx_file():
    file = BytesIO(b"This is not a valid PPTX file content")
    file.name = "invalid.pptx"
    return file


@pytest.fixture
def unsupported_file():
    wrong_file = BytesIO(b"This is not a PPTX file, just plain text")
    wrong_file.name = "text.txt"
    return wrong_file


@pytest.fixture
def non_existent_pptx_file(tmp_path):
    return str(tmp_path / "non_existent_file.pptx")


def test_workflow_with_pptx_converter(valid_pptx_file):
    file, content = valid_pptx_file
    pptx_converter = PPTXFileConverter()
    wf_pptx = Workflow(flow=Flow(nodes=[pptx_converter]))
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
    workflow_with_pptx_converter_and_output, pptx_converter, output_node, invalid_pptx_file
):
    input_data = {"files": [invalid_pptx_file]}

    response = workflow_with_pptx_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pptx_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pptx_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_pptx_converter_file_not_found(
    workflow_with_pptx_converter_and_output, pptx_converter, output_node, non_existent_pptx_file
):
    input_data = {"file_paths": [non_existent_pptx_file]}

    response = workflow_with_pptx_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pptx_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "No files found in the provided paths" in response.output[pptx_converter.id]["error"]["message"]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_pptx_converter_empty_file(
    workflow_with_pptx_converter_and_output, pptx_converter, output_node, empty_pptx_file
):
    input_data = {"file_paths": [empty_pptx_file]}

    response = workflow_with_pptx_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pptx_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pptx_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_pptx_converter_empty_presentation(
    workflow_with_pptx_converter_and_output, pptx_converter, output_node, empty_pptx_presentation
):
    input_data = {"files": [empty_pptx_presentation]}
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
    workflow_with_pptx_converter_and_output, pptx_converter, output_node, unsupported_file
):
    input_data = {"files": [unsupported_file]}

    response = workflow_with_pptx_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[pptx_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[pptx_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value
