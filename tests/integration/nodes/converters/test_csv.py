import csv
import os
import tempfile
from pathlib import Path

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.csv import CSVConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableResult, RunnableStatus


def create_test_csv(data: list[list[str]], header: list[str]) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", newline="", delete=False, suffix=".csv") as tmp_csv:
        writer = csv.writer(tmp_csv)
        writer.writerow(header)
        writer.writerows(data)
        return tmp_csv.name


@pytest.fixture
def sample_csv():
    header = ["Target", "Feature_1", "Feature_2"]
    rows = [
        ["Document 1", "Value 1A", "Value 2A"],
        ["Document 2", "Value 1B", "Value 2B"],
    ]

    tmp_path = create_test_csv(rows, header)
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


@pytest.fixture
def csv_converter():
    return CSVConverter(
        id="csv_converter",
        name="Test CSV Converter",
        delimiter=",",
        content_column="Target",
        metadata_columns=["Feature_1", "Feature_2"],
    )


@pytest.fixture
def output_node(csv_converter):
    return Output(id="output_node", depends=[NodeDependency(csv_converter)])


@pytest.fixture
def workflow_with_csv_converter_and_output(csv_converter, output_node):
    return Workflow(
        id="test_workflow",
        flow=Flow(
            nodes=[csv_converter, output_node],
        ),
        version="1",
    )


def test_csv_loader_basic_functionality(sample_csv):
    csv_loader = CSVConverter(delimiter=",", content_column="Target", metadata_columns=["Feature_1", "Feature_2"])
    input_data = {"file_paths": [sample_csv]}

    result = csv_loader.run(input_data=input_data, config=None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert "documents" in result.output

    documents = result.output["documents"]
    assert len(documents) == 2

    first_doc = documents[0]
    assert first_doc["content"] == "Document 1"
    assert first_doc["metadata"]["Feature_1"] == "Value 1A"
    assert first_doc["metadata"]["Feature_2"] == "Value 2A"
    assert first_doc["metadata"]["source"] == sample_csv


def test_csv_loader_missing_metadata_columns(sample_csv):
    csv_loader = CSVConverter(
        delimiter=",", content_column="Target", metadata_columns=["Feature_1", "NonExistentFeature"]
    )
    input_data = {"file_paths": [sample_csv]}

    result = csv_loader.run(input_data=input_data, config=None)

    first_doc = result.output["documents"][0]
    assert "Feature_1" in first_doc["metadata"]
    assert "NonExistentFeature" not in first_doc["metadata"]


def test_workflow_with_csv_converter_file_not_found(workflow_with_csv_converter_and_output, csv_converter, output_node):
    non_existent_path = str(Path("/tmp") / f"non_existent_file_{os.getpid()}.csv")
    input_data = {"file_paths": [non_existent_path]}

    response = workflow_with_csv_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[csv_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "No such file or directory" in response.output[csv_converter.id]["error"]["message"]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_csv_converter_empty_file(
    workflow_with_csv_converter_and_output, csv_converter, output_node, tmp_path
):
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()

    input_data = {"file_paths": [str(empty_file)]}

    response = workflow_with_csv_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[csv_converter.id]["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in response.output[csv_converter.id]["output"]
    assert len(response.output[csv_converter.id]["output"]["documents"]) == 0
    assert response.output[output_node.id]["status"] == RunnableStatus.SUCCESS.value


def test_workflow_with_csv_converter_invalid_content(
    workflow_with_csv_converter_and_output, csv_converter, output_node, tmp_path
):
    invalid_file = tmp_path / "invalid.csv"
    invalid_file.write_text('Column1,Column2\n"unterminated quote,value2')

    input_data = {"file_paths": [str(invalid_file)]}

    response = workflow_with_csv_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[csv_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[csv_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


def test_workflow_with_csv_converter_missing_content_column(
    workflow_with_csv_converter_and_output, csv_converter, output_node, tmp_path
):
    header = ["Feature_1", "Feature_2"]
    rows = [
        ["Value 1A", "Value 2A"],
        ["Value 1B", "Value 2B"],
    ]

    missing_column_csv = create_test_csv(rows, header)

    try:
        input_data = {"file_paths": [missing_column_csv]}

        response = workflow_with_csv_converter_and_output.run(input_data=input_data)

        assert response.status == RunnableStatus.SUCCESS
        assert response.output[csv_converter.id]["status"] == RunnableStatus.FAILURE.value
        assert "Content column 'Target' not found" in response.output[csv_converter.id]["error"]["message"]
        assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value
    finally:
        if os.path.exists(missing_column_csv):
            os.remove(missing_column_csv)


def test_workflow_with_csv_converter_corrupted_file(
    workflow_with_csv_converter_and_output, csv_converter, output_node, tmp_path
):
    corrupted_file = tmp_path / "corrupted.csv"
    corrupted_file.write_text('Header1,Header2,Header3\n"Unclosed quote,Value2,Value3\nValue4,Value5,Value6')

    input_data = {"file_paths": [str(corrupted_file)]}

    response = workflow_with_csv_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[csv_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[csv_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value
