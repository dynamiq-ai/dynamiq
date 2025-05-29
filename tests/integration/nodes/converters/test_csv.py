import csv
from io import BytesIO, StringIO
from pathlib import Path

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.csv import CSVConverter
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableResult, RunnableStatus


def write_csv_to_path(path: Path, header: list[str], rows: list[list[str]]) -> str:
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def write_csv_to_bytesio(header: list[str], rows: list[list[str]], filename: str = "file.csv") -> BytesIO:
    string_buffer = StringIO()
    writer = csv.writer(string_buffer)
    writer.writerow(header)
    writer.writerows(rows)
    csv_buffer = BytesIO(string_buffer.getvalue().encode("utf-8"))
    csv_buffer.seek(0)
    csv_buffer.name = filename
    return csv_buffer


@pytest.fixture
def csv_test_data():
    return {
        "header": ["Target", "Feature_1", "Feature_2"],
        "rows": [
            ["Document 1", "Value 1A", "Value 2A"],
            ["Document 2", "Value 1B", "Value 2B"],
        ],
    }


@pytest.fixture
def csv_file_path(tmp_path, csv_test_data):
    test_file = tmp_path / "sample.csv"
    return write_csv_to_path(test_file, csv_test_data["header"], csv_test_data["rows"])


@pytest.fixture
def csv_bytesio(csv_test_data):
    return write_csv_to_bytesio(csv_test_data["header"], csv_test_data["rows"], "sample_bytesio.csv")


@pytest.fixture
def sample_csv(tmp_path):
    return csv_file_path(tmp_path)


@pytest.fixture
def empty_csv_file_path(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()
    return str(empty_file)


@pytest.fixture
def empty_csv_bytesio():
    empty_buffer = BytesIO()
    empty_buffer.name = "empty_bytesio.csv"
    return empty_buffer


@pytest.fixture
def invalid_csv_file_path(tmp_path):
    invalid_file = tmp_path / "invalid.csv"
    invalid_file.write_text('Column1,Column2\n"unterminated quote,value2')
    return str(invalid_file)


@pytest.fixture
def invalid_csv_bytesio():
    invalid_buffer = BytesIO(b'Column1,Column2\n"unterminated quote,value2')
    invalid_buffer.name = "invalid_bytesio.csv"
    return invalid_buffer


@pytest.fixture
def missing_target_column_csv_data():
    return {
        "header": ["Feature_1", "Feature_2"],  # Missing the "Target" column
        "rows": [
            ["Value 1A", "Value 2A"],
            ["Value 1B", "Value 2B"],
        ],
    }


@pytest.fixture
def missing_target_column_csv_file_path(tmp_path, missing_target_column_csv_data):
    test_file = tmp_path / "missing_target_column.csv"
    return write_csv_to_path(
        test_file, missing_target_column_csv_data["header"], missing_target_column_csv_data["rows"]
    )


@pytest.fixture
def missing_target_column_csv_bytesio(missing_target_column_csv_data):
    return write_csv_to_bytesio(
        missing_target_column_csv_data["header"],
        missing_target_column_csv_data["rows"],
        "missing_target_column_bytesio.csv",
    )


@pytest.fixture
def corrupted_csv_content():
    return 'Header1,Header2,Header3\n"Unclosed quote,Value2,Value3\nValue4,Value5,Value6'


@pytest.fixture
def corrupted_csv_file_path(tmp_path, corrupted_csv_content):
    corrupted_file = tmp_path / "corrupted.csv"
    corrupted_file.write_text(corrupted_csv_content)
    return str(corrupted_file)


@pytest.fixture
def corrupted_csv_bytesio(corrupted_csv_content):
    content_bytes = corrupted_csv_content.encode("utf-8")
    corrupted_buffer = BytesIO(content_bytes)
    corrupted_buffer.name = "corrupted_bytesio.csv"
    return corrupted_buffer


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


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "csv_file_path"),
        ("files", "csv_bytesio"),
    ],
)
def test_csv_loader_basic_functionality(request, input_type, input_fixture):
    csv_input = request.getfixturevalue(input_fixture)
    csv_loader = CSVConverter(delimiter=",", content_column="Target", metadata_columns=["Feature_1", "Feature_2"])
    input_data = {input_type: [csv_input]}

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

    expected_source = csv_input if input_type == "file_paths" else csv_input.name
    assert first_doc["metadata"]["source"] == expected_source


def test_csv_loader_missing_metadata_columns(csv_file_path):
    csv_loader = CSVConverter(
        delimiter=",", content_column="Target", metadata_columns=["Feature_1", "NonExistentFeature"]
    )
    input_data = {"file_paths": [csv_file_path]}

    result = csv_loader.run(input_data=input_data, config=None)

    first_doc = result.output["documents"][0]
    assert "Feature_1" in first_doc["metadata"]
    assert "NonExistentFeature" not in first_doc["metadata"]


def test_workflow_with_csv_converter_file_not_found(
    workflow_with_csv_converter_and_output, csv_converter, output_node, tmp_path
):
    non_existent_path = str(tmp_path / "non_existent_file.csv")
    input_data = {"file_paths": [non_existent_path]}

    response = workflow_with_csv_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[csv_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "No such file or directory" in response.output[csv_converter.id]["error"]["message"]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "empty_csv_file_path"),
        ("files", "empty_csv_bytesio"),
    ],
)
def test_workflow_with_csv_converter_empty(
    request, workflow_with_csv_converter_and_output, csv_converter, output_node, input_type, input_fixture
):
    csv_input = request.getfixturevalue(input_fixture)
    input_data = {input_type: [csv_input]}

    response = workflow_with_csv_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[csv_converter.id]["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in response.output[csv_converter.id]["output"]
    assert len(response.output[csv_converter.id]["output"]["documents"]) == 0
    assert response.output[output_node.id]["status"] == RunnableStatus.SUCCESS.value


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "invalid_csv_file_path"),
        ("files", "invalid_csv_bytesio"),
    ],
)
def test_workflow_with_csv_converter_invalid_content(
    request, workflow_with_csv_converter_and_output, csv_converter, output_node, input_type, input_fixture
):
    csv_input = request.getfixturevalue(input_fixture)
    input_data = {input_type: [csv_input]}

    response = workflow_with_csv_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[csv_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[csv_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "missing_target_column_csv_file_path"),
        ("files", "missing_target_column_csv_bytesio"),
    ],
)
def test_workflow_with_csv_converter_missing_content_column(
    request, workflow_with_csv_converter_and_output, csv_converter, output_node, input_type, input_fixture
):
    csv_input = request.getfixturevalue(input_fixture)
    input_data = {input_type: [csv_input]}

    response = workflow_with_csv_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[csv_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "Content column 'Target' not found" in response.output[csv_converter.id]["error"]["message"]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


@pytest.mark.parametrize(
    "input_type,input_fixture",
    [
        ("file_paths", "corrupted_csv_file_path"),
        ("files", "corrupted_csv_bytesio"),
    ],
)
def test_workflow_with_csv_converter_corrupted_file(
    request, workflow_with_csv_converter_and_output, csv_converter, output_node, input_type, input_fixture
):
    csv_input = request.getfixturevalue(input_fixture)
    input_data = {input_type: [csv_input]}

    response = workflow_with_csv_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    assert response.output[csv_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[csv_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value
