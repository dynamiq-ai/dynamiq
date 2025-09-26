import pytest

from dynamiq.nodes.tools.file_tools import FileListTool, FileReadTool, FileWriteTool
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.storages.file.in_memory import InMemoryFileStore

@pytest.fixture
def file_store():
    """Create an in-memory file storage instance for testing."""
    return InMemoryFileStore()


@pytest.fixture
def sample_file_path():
    """Sample file path for testing."""
    return "test/file.txt"

@pytest.fixture
def llm_model():
    connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="api-key")
    return OpenAI(name="OpenAI", model="gpt-4o-mini", connection=connection)

def test_file_read_tool(file_store, sample_file_path):
    """Test FileReadTool functionality including initialization, successful read, and error handling."""
    # Test initialization
    tool = FileReadTool(file_store=file_store, llm=llm_model)
    assert tool.name == "FileReadTool"
    assert tool.group == "tools"
    assert tool.file_store == file_store

    # Create a test file first
    test_content = "Hello, this is test content!"
    file_store.store(sample_file_path, test_content)

    # Test successful read
    input_data = {"file_path": sample_file_path}
    result = tool.run(input_data)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"].decode("utf-8") == test_content


def test_file_write_tool(file_store):
    """Test FileWriteTool functionality including initialization, successful writes, and error handling."""
    # Test initialization
    tool = FileWriteTool(file_store=file_store)
    assert tool.name == "FileWriteTool"
    assert tool.group == "tools"
    assert tool.file_store == file_store

    # Test successful text write
    input_data = {"file_path": "test/output.txt", "content": "Hello, World!"}
    result = tool.run(input_data)

    assert result.status == RunnableStatus.SUCCESS
    assert "written successfully" in result.output["content"]

    # Verify file was actually written
    assert file_store.exists("test/output.txt")
    stored_content = file_store.retrieve("test/output.txt")
    assert stored_content == b"Hello, World!"

    # Test custom content type
    input_data = {"file_path": "test/data.csv", "content": "name,age\nJohn,30", "content_type": "text/csv"}

    result = tool.run(input_data)
    assert result.status == RunnableStatus.SUCCESS

    # Verify file with custom content type
    assert file_store.exists("test/data.csv")
    stored_content = file_store.retrieve("test/data.csv")
    assert stored_content == b"name,age\nJohn,30"


def test_file_list_tool(file_store):
    """Test FileListTool functionality including initialization, listing files, and empty directory."""
    # Test initialization
    tool = FileListTool(file_store=file_store)
    assert tool.name == "FileListTool"
    assert tool.group == "tools"
    assert tool.file_store == file_store

    # Create some test files
    file_store.store("test/file1.txt", "content1")
    file_store.store("test/file2.txt", "content2")
    file_store.store("docs/readme.md", "readme content")

    # Test listing files from root
    input_data = {"file_path": "", "recursive": True}
    result = tool.run(input_data)

    assert result.status == RunnableStatus.SUCCESS
    assert "Files currently available" in result.output["content"]
    assert "file1.txt" in result.output["content"]
    assert "file2.txt" in result.output["content"]
    assert "readme.md" in result.output["content"]

    # Test listing from specific directory
    input_data = {"file_path": "test/", "recursive": False}
    result = tool.run(input_data)

    assert result.status == RunnableStatus.SUCCESS
    assert "file1.txt" in result.output["content"]
    assert "file2.txt" in result.output["content"]
    assert "readme.md" not in result.output["content"]  # Should not be in test/ directory

    # Test empty directory
    input_data = {"file_path": "empty/", "recursive": False}
    result = tool.run(input_data)

    assert result.status == RunnableStatus.SUCCESS
    assert "Files currently available" in result.output["content"]
    assert "File: " not in result.output["content"]


def test_file_tools_integration(file_store):
    """Test file tools working together."""
    # Test tools working together
    write_tool = FileWriteTool(file_store=file_store)
    read_tool = FileReadTool(file_store=file_store)

    # Write file
    write_input = {"file_path": "test/integration.txt", "content": "Integration test content"}
    write_result = write_tool.run(write_input)
    assert write_result.status == RunnableStatus.SUCCESS

    # Read file
    read_input = {"file_path": "test/integration.txt"}
    read_result = read_tool.run(read_input)
    assert read_result.status == RunnableStatus.SUCCESS
    assert read_result.output["content"].decode("utf-8") == "Integration test content"

    # Test different storage instances
    storage2 = InMemoryFileStore()
    storage2.store("test.txt", "Content from storage 2")

    read_tool2 = FileReadTool(file_store=storage2)
    input_data = {"file_path": "test.txt"}

    result2 = read_tool2.run(input_data)
    assert result2.output["content"].decode("utf-8") == "Content from storage 2"
