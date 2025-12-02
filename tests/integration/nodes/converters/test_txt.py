from io import BytesIO
from pathlib import Path

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.text import TextFileConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableStatus


def write_txt_to_path(path: Path, content: str) -> str:
    path.write_text(content)
    return str(path)


def write_txt_to_bytesio(content: str, filename: str = "file.txt") -> BytesIO:
    txt_buffer = BytesIO(content.encode("utf-8"))
    txt_buffer.name = filename
    return txt_buffer


@pytest.fixture
def txt_node():
    return TextFileConverter(
        id="txt_converter",
        name="Test TXT Converter",
    )


@pytest.fixture
def output_node(txt_node):
    return Output(id="output_node", depends=[NodeDependency(txt_node)])


@pytest.fixture
def workflow(txt_node, output_node):
    return Workflow(
        id="test_workflow",
        flow=Flow(
            nodes=[txt_node, output_node],
        ),
        version="1",
    )


@pytest.fixture
def basic_txt_content():
    return """
    Hello, World!
    This is a test paragraph for the TextFileConverter.
    Item 1
    Item 2
    Item 3
    """


@pytest.fixture
def basic_txt_file_path(tmp_path, basic_txt_content):
    test_file = tmp_path / "test.txt"
    return write_txt_to_path(test_file, basic_txt_content)


@pytest.fixture
def basic_txt_bytesio(basic_txt_content):
    return write_txt_to_bytesio(basic_txt_content, "test.txt")


@pytest.fixture
def non_existent_txt_file(tmp_path):
    return str(tmp_path / "non_existent_file.txt")


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "basic_txt_file_path"),
        ("files", "basic_txt_bytesio"),
    ],
)
def test_workflow_with_txt_converter(request, input_type, input_fixture):
    txt_input = request.getfixturevalue(input_fixture)
    wf_txt = Workflow(
        flow=Flow(nodes=[TextFileConverter()]),
    )

    input_data = {input_type: [txt_input]}
    response = wf_txt.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS

    node_id = wf_txt.flow.nodes[0].id
    assert "documents" in response.output[node_id]["output"]
    assert len(response.output[node_id]["output"]["documents"]) == 1

    document = response.output[node_id]["output"]["documents"][0]
    assert "Hello, World!" in document["content"]
    assert "This is a test paragraph for the TextFileConverter" in document["content"]
    assert "Item 1" in document["content"]
    assert "Item 2" in document["content"]
    assert "Item 3" in document["content"]

    expected_source = txt_input if input_type == "file_paths" else txt_input.name
    assert document["metadata"]["file_path"] == expected_source


def test_workflow_with_txt_node_file_not_found(workflow, txt_node, output_node, non_existent_txt_file):
    input_data = {"file_paths": [non_existent_txt_file]}

    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.FAILURE

    txt_result = result.output[txt_node.id]
    assert txt_result["status"] == RunnableStatus.FAILURE.value
    assert "No files found in the provided paths" in txt_result["error"]["message"]

    output_result = result.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value
