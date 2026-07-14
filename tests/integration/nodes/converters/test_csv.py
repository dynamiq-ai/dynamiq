import csv
from io import BytesIO, StringIO
from pathlib import Path

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.converters.csv import CSVConverter, CSVDocumentCreationMode
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
def header_only_csv_bytesio():
    return write_csv_to_bytesio(["Target", "Feature_1", "Feature_2"], [], "header_only.csv")


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
    assert first_doc["metadata"]["file_path"] == expected_source
    assert "source" not in first_doc["metadata"]


def test_csv_loader_missing_metadata_columns(csv_file_path):
    csv_loader = CSVConverter(
        delimiter=",", content_column="Target", metadata_columns=["Feature_1", "NonExistentFeature"]
    )
    input_data = {"file_paths": [csv_file_path]}

    result = csv_loader.run(input_data=input_data, config=None)

    first_doc = result.output["documents"][0]
    assert "Feature_1" in first_doc["metadata"]
    assert "NonExistentFeature" not in first_doc["metadata"]


def test_csv_loader_named_rows_preserve_dict_reader_ragged_row_semantics():
    csv_loader = CSVConverter(content_column="content", metadata_columns=["category"])
    file = BytesIO(b"content,category\nshort\nlong,news,extra\n")
    file.name = "ragged.csv"

    result = csv_loader.run(input_data={"files": [file]})

    documents = result.output["documents"]
    assert documents[0]["content"] == "short"
    assert documents[0]["metadata"]["category"] is None
    assert documents[1]["content"] == "long"
    assert csv_loader._map_named_row(["content", "category"], ["long", "news", "extra"])[None] == ["extra"]


def test_csv_loader_content_column_skips_rows_with_empty_content():
    csv_loader = CSVConverter(content_column="content", metadata_columns=["category"])
    file = BytesIO(b"content,category\n,empty\nkept,first\n   ,blank\nalso kept,second\n")
    file.name = "sparse.csv"

    result = csv_loader.run(input_data={"files": [file]})

    documents = result.output["documents"]
    assert [document["content"] for document in documents] == ["kept", "also kept"]
    assert [document["metadata"]["row_number"] for document in documents] == [3, 5]


def test_csv_loader_content_column_rejects_file_with_only_empty_content():
    csv_loader = CSVConverter(content_column="content")
    file = BytesIO(b"content,category\n,empty\n   ,blank\n")
    file.name = "empty-content.csv"

    result = csv_loader.run(input_data={"files": [file]})

    assert result.status == RunnableStatus.FAILURE
    assert result.error is not None
    assert result.error.message == (
        "No documents were created from the provided inputs. Please check your files and try again."
    )


def test_csv_loader_without_content_column_creates_self_describing_rows(csv_bytesio):
    csv_loader = CSVConverter()

    result = csv_loader.run(input_data={"files": [csv_bytesio]})

    documents = result.output["documents"]
    assert documents[0]["content"] == "Target: Document 1\nFeature_1: Value 1A\nFeature_2: Value 2A"
    assert documents[0]["metadata"]["document_type"] == "table_row"
    assert documents[0]["metadata"]["row_number"] == 2


def test_csv_loader_streams_uploaded_rows_without_reading_the_whole_file():
    class NoReadAllBytesIO(BytesIO):
        def read(self, size=-1):
            if size == -1:
                raise AssertionError("row mode must not read the entire upload at once")
            return super().read(size)

    file = NoReadAllBytesIO(b"name,price\nBasic,10\nPro,20\n")
    file.name = "pricing.csv"

    result = CSVConverter().run(input_data={"files": [file]})

    assert result.status == RunnableStatus.SUCCESS
    assert [document["content"] for document in result.output["documents"]] == [
        "name: Basic\nprice: 10",
        "name: Pro\nprice: 20",
    ]


def test_csv_loader_without_content_column_rejects_header_only_file(header_only_csv_bytesio):
    result = CSVConverter().run(input_data={"files": [header_only_csv_bytesio]})

    assert result.status == RunnableStatus.FAILURE
    assert result.error is not None
    assert result.error.message == (
        "No documents were created from the provided inputs. Please check your files and try again."
    )


def test_csv_loader_generic_rows_preserve_source_url_and_duplicate_headers():
    file = write_csv_to_bytesio(["", "plan", "plan"], [["Feature", "Free", "Pro"]], "pricing.csv")
    source_url = "https://example.com/pricing"
    csv_loader = CSVConverter()

    result = csv_loader.run(
        input_data={
            "files": [file],
            "metadata": [{"dynamiq_item_source_provider_url": source_url}],
        }
    )

    document = result.output["documents"][0]
    assert document["content"] == "column_1: Feature\nplan: Free\nplan_2: Pro"
    assert document["metadata"]["dynamiq_item_source_provider_url"] == source_url
    assert "source" not in document["metadata"]


def test_csv_loader_keeps_generated_headers_globally_unique():
    file = write_csv_to_bytesio(["plan", "plan", "plan_2"], [["Free", "Pro", "Business"]], "plans.csv")

    result = CSVConverter().run(input_data={"files": [file]})

    assert result.output["documents"][0]["content"] == ("plan: Free\nplan_3: Pro\nplan_2: Business")


def test_csv_loader_can_preserve_whole_file_as_plain_text(csv_bytesio):
    csv_loader = CSVConverter(document_creation_mode=CSVDocumentCreationMode.ONE_DOC_PER_FILE)

    result = csv_loader.run(input_data={"files": [csv_bytesio]})

    documents = result.output["documents"]
    assert len(documents) == 1
    assert documents[0]["content"].startswith("Target,Feature_1,Feature_2")
    assert "Document 2,Value 1B,Value 2B" in documents[0]["content"]
    assert documents[0]["metadata"]["document_type"] == "table"
    assert documents[0]["metadata"]["row_count"] == 2


def test_csv_loader_plain_text_mode_preserves_malformed_csv(corrupted_csv_bytesio, corrupted_csv_content):
    csv_loader = CSVConverter(document_creation_mode=CSVDocumentCreationMode.ONE_DOC_PER_FILE)

    result = csv_loader.run(input_data={"files": [corrupted_csv_bytesio]})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["documents"][0]["content"] == corrupted_csv_content
    assert result.output["documents"][0]["metadata"]["row_count"] == 1


def test_csv_input_can_override_document_creation_mode(csv_bytesio):
    csv_loader = CSVConverter(document_creation_mode=CSVDocumentCreationMode.ONE_DOC_PER_FILE)

    result = csv_loader.run(
        input_data={
            "files": [csv_bytesio],
            "document_creation_mode": CSVDocumentCreationMode.ONE_DOC_PER_ROW,
        }
    )

    assert len(result.output["documents"]) == 2


def test_csv_loader_labels_tsv_content_type():
    file = BytesIO(b"name\tprice\nBasic\t10\n")
    file.name = "pricing.tsv"

    result = CSVConverter(delimiter="\t").run(input_data={"files": [file]})

    document = result.output["documents"][0]
    assert document["content"] == "name: Basic\nprice: 10"
    assert document["metadata"]["content_type"] == "text/tab-separated-values"


def test_workflow_with_csv_converter_file_not_found(
    workflow_with_csv_converter_and_output, csv_converter, output_node, tmp_path
):
    non_existent_path = str(tmp_path / "non_existent_file.csv")
    input_data = {"file_paths": [non_existent_path]}

    response = workflow_with_csv_converter_and_output.run(input_data=input_data)

    assert response.status == RunnableStatus.FAILURE
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

    assert response.status == RunnableStatus.FAILURE
    assert response.output[csv_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "No documents were created" in response.output[csv_converter.id]["error"]["message"]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value


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

    assert response.status == RunnableStatus.FAILURE
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

    assert response.status == RunnableStatus.FAILURE
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

    assert response.status == RunnableStatus.FAILURE
    assert response.output[csv_converter.id]["status"] == RunnableStatus.FAILURE.value
    assert "error" in response.output[csv_converter.id]
    assert response.output[output_node.id]["status"] == RunnableStatus.SKIP.value
