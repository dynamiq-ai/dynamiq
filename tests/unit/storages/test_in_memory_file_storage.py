import pytest

from dynamiq.storages.file_storage.in_memory import InMemoryFileStorage
from dynamiq.storages.file_storage.base import FileInfo, FileNotFoundError, FileExistsError


@pytest.fixture
def storage():
    """Create a fresh storage instance for each test."""
    return InMemoryFileStorage()


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return "Hello, this is a test file content!"


@pytest.fixture
def sample_file_path():
    """Sample file path for testing."""
    return "test/file.txt"


def test_init(storage):
    """Test storage initialization."""
    assert storage._files == {}


def test_store_text_content_basic(storage):
    """Test storing text content with basic functionality."""
    text_content = "text content"
    file_info = storage.store("test.txt", text_content)

    assert file_info.size == len(text_content.encode('utf-8'))
    stored_content = storage.retrieve("test.txt")
    assert stored_content == text_content.encode('utf-8')


def test_store_file_exists_error(storage, sample_text_content):
    """Test that storing existing file raises FileExistsError."""
    file_path = "test/exists.txt"
    storage.store(file_path, sample_text_content)

    with pytest.raises(FileExistsError) as exc_info:
        storage.store(file_path, "new content")

    assert exc_info.value.path == file_path
    assert "already exists" in str(exc_info.value)


def test_store_text_content_full(storage, sample_text_content, sample_file_path):
    """Test storing text content with full validation."""
    file_info = storage.store(sample_file_path, sample_text_content)

    assert isinstance(file_info, FileInfo)
    assert file_info.path == sample_file_path
    assert file_info.size == len(sample_text_content.encode('utf-8'))
    assert file_info.content_type == "text/plain"
    assert file_info.created_at is not None

    # Check internal storage
    assert sample_file_path in storage._files
    stored_content = storage._files[sample_file_path]['content']
    assert stored_content == sample_text_content.encode('utf-8')


def test_retrieve_nonexistent_file(storage):
    """Test that retrieving nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        storage.retrieve("nonexistent.txt")

    assert exc_info.value.path == "nonexistent.txt"
    assert "not found" in str(exc_info.value)


def test_exists(storage, sample_text_content, sample_file_path):
    """Test exists returns True for existing file."""
    storage.store(sample_file_path, sample_text_content)
    assert storage.exists(sample_file_path) is True
    assert storage.exists("nonexistent.txt") is False


def test_delete_existing_file(storage, sample_text_content, sample_file_path):
    """Test deleting existing file."""
    storage.store(sample_file_path, sample_text_content)

    result = storage.delete(sample_file_path)
    assert result is True
    assert not storage.exists(sample_file_path)
    assert sample_file_path not in storage._files


def test_delete_nonexistent_file(storage):
    """Test deleting nonexistent file."""
    result = storage.delete("nonexistent.txt")
    assert result is False


def test_list_files_recursive_and_directory_behavior(storage, sample_text_content):
    """Test recursive vs non-recursive file listing and directory parameter behavior."""

    # Create a simple nested structure
    storage.store("file1.txt", sample_text_content)
    storage.store("docs/file2.txt", sample_text_content)
    storage.store("docs/sub/file3.txt", sample_text_content)
    storage.store("images/photo.jpg", sample_text_content)

    # Test 1: Root level recursive vs non-recursive
    files_non_recursive = storage.list_files(recursive=False)
    files_recursive = storage.list_files(recursive=True)

    # Non-recursive should only show root-level files
    assert len(files_non_recursive) == 2
    assert "file1.txt" in [f.path for f in files_non_recursive]
    assert "docs/file2.txt" in [f.path for f in files_non_recursive]
    assert "docs/sub/file3.txt" not in [f.path for f in files_non_recursive]

    # Recursive should show all files
    assert len(files_recursive) == 4
    assert "file1.txt" in [f.path for f in files_recursive]
    assert "docs/file2.txt" in [f.path for f in files_recursive]
    assert "docs/sub/file3.txt" in [f.path for f in files_recursive]
    assert "images/photo.jpg" in [f.path for f in files_recursive]

    # Test 2: Directory parameter with recursive vs non-recursive
    docs_files_non_recursive = storage.list_files(directory="docs", recursive=False)
    docs_files_recursive = storage.list_files(directory="docs", recursive=True)

    # Non-recursive in docs should only show direct files
    assert len(docs_files_non_recursive) == 1
    assert "docs/file2.txt" in [f.path for f in docs_files_non_recursive]
    assert "docs/sub/file3.txt" not in [f.path for f in docs_files_non_recursive]

    # Recursive in docs should show all nested files
    assert len(docs_files_recursive) == 2
    assert "docs/file2.txt" in [f.path for f in docs_files_recursive]
    assert "docs/sub/file3.txt" in [f.path for f in docs_files_recursive]
