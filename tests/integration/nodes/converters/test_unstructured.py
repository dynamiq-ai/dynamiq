import os
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


def test_workflow_with_unstructured_converter_success(unstructured_node, tmp_path):
    wf = Workflow(flow=flows.Flow(nodes=[unstructured_node]))

    mock_elements = [
        {"text": "This is title", "category": "Title"},
        {"text": "This is paragraph 1", "category": "Text"},
        {"text": "This is paragraph 2", "category": "Text"},
    ]

    test_file = tmp_path / "test_file.txt"
    test_file.write_text("test content")

    with patch("dynamiq.components.converters.unstructured.partition_via_api", return_value=mock_elements):
        input_data = {"file_paths": [str(test_file)]}
        response = wf.run(input_data=input_data)

        assert response.status == RunnableStatus.SUCCESS
        node_id = unstructured_node.id

        assert response.output[node_id]["status"] == RunnableStatus.SUCCESS.value
        assert "documents" in response.output[node_id]["output"]
        assert len(response.output[node_id]["output"]["documents"]) == 1

        document = response.output[node_id]["output"]["documents"][0]
        assert "# This is title" in document["content"]
        assert "This is paragraph 1" in document["content"]
        assert "This is paragraph 2" in document["content"]
        assert document["metadata"]["file_path"] == str(test_file)


def test_workflow_with_bytesio_success(unstructured_node):
    wf = Workflow(flow=flows.Flow(nodes=[unstructured_node]))

    mock_elements = [{"text": "This is a document from BytesIO", "category": "Text"}]

    with patch("dynamiq.components.converters.unstructured.partition_via_api", return_value=mock_elements):
        file = BytesIO(b"mock content")
        file.name = "mock_file.pdf"
        input_data = {"files": [file]}

        response = wf.run(input_data=input_data)

        assert response.status == RunnableStatus.SUCCESS
        node_id = unstructured_node.id

        assert response.output[node_id]["status"] == RunnableStatus.SUCCESS.value
        assert "documents" in response.output[node_id]["output"]
        document = response.output[node_id]["output"]["documents"][0]
        assert "This is a document from BytesIO" in document["content"]
        assert document["metadata"]["file_path"] == "mock_file.pdf"


def test_workflow_with_unstructured_node_failure(workflow, unstructured_node, output_node, tmp_path):
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("test content")

    with patch("dynamiq.components.converters.unstructured.partition_via_api") as mock_partition:
        error_msg = "File format not supported or invalid"
        mock_partition.side_effect = ValueError(error_msg)

        input_data = {"file_paths": [str(test_file)]}

        result = workflow.run(input_data=input_data)

        assert result.status == RunnableStatus.SUCCESS

        unstructured_result = result.output[unstructured_node.id]
        assert unstructured_result["status"] == RunnableStatus.FAILURE.value
        assert unstructured_result["error"]["type"] == "ValueError"
        assert unstructured_result["error"]["message"] == error_msg

        output_result = result.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


def test_workflow_with_unstructured_node_file_not_found(workflow, unstructured_node, output_node):
    non_existent_path = str(Path("/tmp") / f"non_existent_file_{os.getpid()}.pdf")
    input_data = {"file_paths": [non_existent_path]}

    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.SUCCESS

    unstructured_result = result.output[unstructured_node.id]
    assert unstructured_result["status"] == RunnableStatus.FAILURE.value
    assert "No files found in the provided paths" in unstructured_result["error"]["message"]

    output_result = result.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


def test_workflow_with_unstructured_node_empty_file(workflow, unstructured_node, output_node, tmp_path):
    empty_file = tmp_path / "empty_file.txt"
    empty_file.touch()

    input_data = {"file_paths": [str(empty_file)]}

    result = workflow.run(input_data=input_data)

    assert result.status == RunnableStatus.SUCCESS

    unstructured_result = result.output[unstructured_node.id]
    assert unstructured_result["status"] == RunnableStatus.FAILURE.value
    assert "ValueError" in unstructured_result["error"]["type"]
    assert "Empty file cannot be processed" in unstructured_result["error"]["message"]

    output_result = result.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value
