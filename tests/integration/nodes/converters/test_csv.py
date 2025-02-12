import csv
import os
import tempfile

import pytest

from dynamiq.nodes.converters.csv import CSVConverter
from dynamiq.runnables import RunnableResult, RunnableStatus


def create_test_csv(data: list[list[str]], header: list[str]) -> str:
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w+", newline="", delete=False, suffix=".csv") as tmp_csv:
        writer = csv.writer(tmp_csv)
        writer.writerow(header)
        writer.writerows(data)
        return tmp_csv.name


@pytest.fixture
def sample_csv():
    """Fixture providing a sample CSV file."""
    header = ["Target", "Feature_1", "Feature_2"]
    rows = [
        ["Document 1", "Value 1A", "Value 2A"],
        ["Document 2", "Value 1B", "Value 2B"],
    ]

    tmp_path = create_test_csv(rows, header)
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_csv_loader_basic_functionality(sample_csv):
    """Test basic CSV loader functionality."""
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
    """Test CSV loader with missing metadata columns."""
    csv_loader = CSVConverter(
        delimiter=",", content_column="Target", metadata_columns=["Feature_1", "NonExistentFeature"]
    )
    input_data = {"file_paths": [sample_csv]}

    result = csv_loader.run(input_data=input_data, config=None)

    first_doc = result.output["documents"][0]
    assert "Feature_1" in first_doc["metadata"]
    assert "NonExistentFeature" not in first_doc["metadata"]
