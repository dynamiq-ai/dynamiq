from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from dynamiq import Workflow, flows
from dynamiq.components.converters.unstructured import ConvertStrategy, DocumentCreationMode
from dynamiq.connections import Unstructured
from dynamiq.nodes.converters.unstructured import UnstructuredFileConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableStatus


def write_text_to_path(path: Path, content: str) -> str:
    path.write_text(content)
    return str(path)


def write_text_to_bytesio(content: str, filename: str = "file.txt") -> BytesIO:
    buffer = BytesIO(content.encode("utf-8"))
    buffer.name = filename
    return buffer


@pytest.fixture
def unstructured_connection():
    return Unstructured(url="test_url", api_key="test-api-key")


@pytest.fixture
def unstructured_node(unstructured_connection):
    return UnstructuredFileConverter(
        id="unstructured_converter",
        name="Test Unstructured Converter",
        connection=unstructured_connection,
        document_creation_mode=DocumentCreationMode.ONE_DOC_PER_FILE,
        strategy=ConvertStrategy.AUTO,
    )


@pytest.fixture
def output_node(unstructured_node):
    return Output(id="output_node", depends=[NodeDependency(unstructured_node)])


@pytest.fixture
def workflow(unstructured_node, output_node):
    return Workflow(
        id="test_workflow",
        flow=flows.Flow(
            nodes=[unstructured_node, output_node],
        ),
        version="1",
    )


@pytest.fixture
def test_text_content():
    return "test content"


@pytest.fixture
def test_text_file_path(tmp_path, test_text_content):
    test_file = tmp_path / "test_file.txt"
    return write_text_to_path(test_file, test_text_content)


@pytest.fixture
def test_text_bytesio(test_text_content):
    return write_text_to_bytesio(test_text_content, "test_file.txt")


@pytest.fixture
def empty_text_file_path(tmp_path):
    empty_file = tmp_path / "empty_file.txt"
    empty_file.touch()
    return str(empty_file)


@pytest.fixture
def empty_text_bytesio():
    empty_buffer = BytesIO(b"")
    empty_buffer.name = "empty_file.txt"
    return empty_buffer


@pytest.fixture
def non_existent_file(tmp_path):
    return str(tmp_path / "non_existent_file.pdf")


@pytest.fixture
def mock_content():
    return "mock content"


@pytest.fixture
def mock_file_path(tmp_path, mock_content):
    mock_file = tmp_path / "mock_file.pdf"
    return write_text_to_path(mock_file, mock_content)


@pytest.fixture
def mock_bytesio(mock_content):
    return write_text_to_bytesio(mock_content, "mock_file.pdf")


@pytest.fixture
def mock_elements_single():
    return [{"text": "This is a document from BytesIO", "category": "Text"}]


@pytest.fixture
def mock_elements_multiple():
    return [
        {"text": "This is title", "category": "Title"},
        {"text": "This is paragraph 1", "category": "Text"},
        {"text": "This is paragraph 2", "category": "Text"},
    ]


@pytest.mark.parametrize(
    "input_type,input_fixture,elements_fixture,expected_content",
    [
        (
            "file_paths",
            "test_text_file_path",
            "mock_elements_multiple",
            ["# This is title", "This is paragraph 1", "This is paragraph 2"],
        ),
        ("files", "mock_bytesio", "mock_elements_single", ["This is a document from BytesIO"]),
    ],
)
def test_workflow_with_unstructured_converter_success(
    request, unstructured_node, input_type, input_fixture, elements_fixture, expected_content
):
    input_file = request.getfixturevalue(input_fixture)
    elements = request.getfixturevalue(elements_fixture)
    wf = Workflow(flow=flows.Flow(nodes=[unstructured_node]))

    with patch("dynamiq.components.converters.unstructured.partition_via_api", return_value=elements):
        input_data = {input_type: [input_file]}
        response = wf.run(input_data=input_data)

        assert response.status == RunnableStatus.SUCCESS
        node_id = unstructured_node.id

        assert response.output[node_id]["status"] == RunnableStatus.SUCCESS.value
        assert "documents" in response.output[node_id]["output"]
        assert len(response.output[node_id]["output"]["documents"]) == 1

        document = response.output[node_id]["output"]["documents"][0]
        for content_piece in expected_content:
            assert content_piece in document["content"]

        expected_path = input_file if input_type == "file_paths" else input_file.name
        assert document["metadata"]["file_path"] == expected_path


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "test_text_file_path"),
        ("files", "test_text_bytesio"),
    ],
)
def test_workflow_with_unstructured_node_failure(
    request, workflow, unstructured_node, output_node, input_type, input_fixture
):
    input_file = request.getfixturevalue(input_fixture)
    with patch("dynamiq.components.converters.unstructured.partition_via_api") as mock_partition:
        error_msg = "File format not supported or invalid"
        mock_partition.side_effect = ValueError(error_msg)

        input_data = {input_type: [input_file]}

        result = workflow.run(input_data=input_data)

        assert result.status == RunnableStatus.FAILURE

        unstructured_result = result.output[unstructured_node.id]
        assert unstructured_result["status"] == RunnableStatus.FAILURE.value
        assert unstructured_result["error"]["type"] == "ValueError"
        assert unstructured_result["error"]["message"] == error_msg

        output_result = result.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


def test_workflow_with_unstructured_node_file_not_found(workflow, unstructured_node, output_node, non_existent_file):
    input_data = {"file_paths": [non_existent_file]}

    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.FAILURE

    unstructured_result = result.output[unstructured_node.id]
    assert unstructured_result["status"] == RunnableStatus.FAILURE.value
    assert "No files found in the provided paths" in unstructured_result["error"]["message"]

    output_result = result.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


def test_workflow_with_unstructured_node_empty_file_path(
    workflow, unstructured_node, output_node, empty_text_file_path
):
    input_data = {"file_paths": [empty_text_file_path]}
    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.FAILURE
    unstructured_result = result.output[unstructured_node.id]
    assert unstructured_result["status"] == RunnableStatus.FAILURE.value
    assert "ValueError" in unstructured_result["error"]["type"]
    assert "Empty file cannot be processed" in unstructured_result["error"]["message"]

    output_result = result.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


def test_workflow_with_unstructured_node_empty_bytesio(workflow, unstructured_node, output_node, empty_text_bytesio):
    with patch("dynamiq.components.converters.unstructured.partition_via_api") as mock_partition:
        mock_partition.side_effect = ValueError("Empty file cannot be processed")

        input_data = {"files": [empty_text_bytesio]}
        result = workflow.run(input_data=input_data)

        assert result.status == RunnableStatus.FAILURE
        unstructured_result = result.output[unstructured_node.id]
        assert unstructured_result["status"] == RunnableStatus.FAILURE.value
        assert "ValueError" in unstructured_result["error"]["type"]
        assert "Empty file cannot be processed" in unstructured_result["error"]["message"]

        output_result = result.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value
