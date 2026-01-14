from dynamiq.storages.vector.utils import (
    create_file_id_filter,
    create_file_ids_filter,
    create_pgvector_file_id_filter,
    create_pgvector_file_ids_filter,
)


def test_create_file_id_filter():
    result = create_file_id_filter("file1")
    assert result["operator"] == "AND"
    assert len(result["conditions"]) == 1
    assert result["conditions"][0]["field"] == "file_id"
    assert result["conditions"][0]["operator"] == "=="
    assert result["conditions"][0]["value"] == "file1"


def test_create_file_ids_filter_empty():
    result = create_file_ids_filter([])
    assert result["operator"] == "AND"
    assert len(result["conditions"]) == 1
    assert result["conditions"][0]["field"] == "file_id"
    assert result["conditions"][0]["operator"] == "in"
    assert result["conditions"][0]["value"] == []


def test_create_file_ids_filter_single():
    result = create_file_ids_filter(["file1"])
    assert result["conditions"][0]["value"] == ["file1"]


def test_create_file_ids_filter_multiple():
    result = create_file_ids_filter(["file1", "file2", "file3"])
    assert result["conditions"][0]["value"] == ["file1", "file2", "file3"]


def test_create_pgvector_file_id_filter():
    """Test pgvector-specific file_id filter creation."""
    result = create_pgvector_file_id_filter("file1")
    assert result["field"] == "metadata.file_id"
    assert result["operator"] == "=="
    assert result["value"] == "file1"


def test_create_pgvector_file_ids_filter_empty():
    """Test pgvector-specific file_ids filter creation with empty list."""
    result = create_pgvector_file_ids_filter([])
    assert result["field"] == "metadata.file_id"
    assert result["operator"] == "in"
    assert result["value"] == []


def test_create_pgvector_file_ids_filter_single():
    """Test pgvector-specific file_ids filter creation with single item."""
    result = create_pgvector_file_ids_filter(["file1"])
    assert result["field"] == "metadata.file_id"
    assert result["operator"] == "in"
    assert result["value"] == ["file1"]


def test_create_pgvector_file_ids_filter_multiple():
    """Test pgvector-specific file_ids filter creation with multiple items."""
    result = create_pgvector_file_ids_filter(["file1", "file2", "file3"])
    assert result["field"] == "metadata.file_id"
    assert result["operator"] == "in"
    assert result["value"] == ["file1", "file2", "file3"]
