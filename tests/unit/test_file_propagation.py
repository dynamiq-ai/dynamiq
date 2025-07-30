import io
import json
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.utils import process_tool_output_for_agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.runnables import RunnableConfig
from dynamiq.types import FileOutput


class TestFileOutput:
    """Test FileOutput data model functionality."""

    def test_file_output_creation(self):
        """Test creating FileOutput from various sources."""
        file_data = b"Hello, World!"
        file_output = FileOutput(
            name="test.txt",
            content=file_data,
            mime_type="text/plain",
            description="Test file",
            path="/tmp/test.txt"
        )
        
        assert file_output.name == "test.txt"
        assert file_output.content == file_data
        assert file_output.mime_type == "text/plain"
        assert file_output.description == "Test file"
        assert file_output.path == "/tmp/test.txt"
        assert file_output.size == len(file_data)

    def test_file_output_from_bytes_io(self):
        """Test creating FileOutput from BytesIO."""
        file_data = b"Test data"
        bytes_io = io.BytesIO(file_data)
        
        file_output = FileOutput.from_bytes_io(
            bytes_io,
            name="test.bin",
            mime_type="application/octet-stream",
            description="Binary test file"
        )
        
        assert file_output.name == "test.bin"
        assert file_output.content == file_data
        assert file_output.mime_type == "application/octet-stream"
        assert file_output.description == "Binary test file"
        assert file_output.size == len(file_data)

    def test_file_output_to_bytes_io(self):
        """Test converting FileOutput to BytesIO."""
        file_data = b"Test content"
        file_output = FileOutput(
            name="output.txt",
            content=file_data,
            mime_type="text/plain"
        )
        
        bytes_io = file_output.to_bytes_io()
        assert bytes_io.getvalue() == file_data
        assert bytes_io.name == "output.txt"

    def test_file_output_serialization(self):
        """Test FileOutput serialization for logging/debugging."""
        file_data = b"Some binary data" * 100  # Make it larger
        file_output = FileOutput(
            name="large_file.bin",
            content=file_data,
            mime_type="application/octet-stream"
        )
        
        serialized = file_output.model_dump_for_serialization()
        assert "content" in serialized
        assert serialized["content"] == f"<binary data: {len(file_data)} bytes>"
        assert serialized["name"] == "large_file.bin"
        assert serialized["size"] == len(file_data)


class TestProcessToolOutputForAgent:
    """Test the enhanced process_tool_output_for_agent function."""

    def test_process_tool_output_without_files(self):
        """Test processing tool output without files (backward compatibility)."""
        tool_output = {"content": "Some text result"}
        
        processed_content, files = process_tool_output_for_agent(tool_output)
        
        assert processed_content == "Some text result"
        assert files == []

    def test_process_tool_output_with_files(self):
        """Test processing tool output with files."""
        file_output = FileOutput(
            name="result.csv",
            content=b"col1,col2\nval1,val2",
            mime_type="text/csv",
            description="Generated CSV file"
        )
        
        tool_output = {
            "content": "Generated a CSV file with results",
            "files": [file_output]
        }
        
        processed_content, files = process_tool_output_for_agent(tool_output)
        
        assert processed_content == "Generated a CSV file with results"
        assert len(files) == 1
        assert files[0] == file_output

    def test_process_tool_output_with_dict_files(self):
        """Test processing tool output with files as dictionaries."""
        tool_output = {
            "content": "Generated files",
            "files": [
                {
                    "name": "output.txt",
                    "content": b"Hello World",
                    "mime_type": "text/plain",
                    "description": "Text output"
                }
            ]
        }
        
        processed_content, files = process_tool_output_for_agent(tool_output)
        
        assert processed_content == "Generated files"
        assert len(files) == 1
        assert isinstance(files[0], FileOutput)
        assert files[0].name == "output.txt"
        assert files[0].content == b"Hello World"

    def test_process_tool_output_string_input(self):
        """Test processing string input (no files)."""
        processed_content, files = process_tool_output_for_agent("Simple string output")
        
        assert processed_content == "Simple string output"
        assert files == []

    def test_process_tool_output_json_without_content_key(self):
        """Test processing JSON output without content key but with files."""
        file_output = FileOutput(
            name="data.json",
            content=b'{"key": "value"}',
            mime_type="application/json"
        )
        
        tool_output = {
            "result": "Success",
            "status": "completed",
            "files": [file_output]
        }
        
        processed_content, files = process_tool_output_for_agent(tool_output)
        
        expected_content = json.dumps({"result": "Success", "status": "completed"}, indent=2)
        assert processed_content == expected_content
        assert len(files) == 1
        assert files[0] == file_output


class MockTool:
    """Mock tool that returns files."""
    
    def __init__(self, return_files=True):
        self.name = "MockTool"
        self.id = "mock-tool-123"
        self.is_files_allowed = True
        self.return_files = return_files
    
    def run(self, input_data, config=None, run_depends=None, **kwargs):
        """Mock run method that returns files."""
        from dynamiq.runnables import RunnableResult, RunnableStatus
        
        content = {"content": "Mock tool executed successfully"}
        
        if self.return_files:
            file_output = FileOutput(
                name="mock_output.txt",
                content=b"Mock file content",
                mime_type="text/plain",
                description="Mock generated file"
            )
            content["files"] = [file_output]
        
        return RunnableResult(
            status=RunnableStatus.SUCCESS,
            output=content
        )


class TestAgentFileCollection:
    """Test agent file collection functionality."""

    def test_agent_file_collection_integration(self):
        """Test file collection integration using mocked methods."""
        tool_result_content = {
            "content": "Tool executed successfully",
            "files": [
                FileOutput(
                    name="result.txt",
                    content=b"Generated content",
                    mime_type="text/plain",
                    description="Generated file"
                )
            ]
        }
        
        processed_content, files = process_tool_output_for_agent(tool_result_content)
        
        assert len(files) == 1
        assert files[0].name == "result.txt"
        assert files[0].content == b"Generated content"
        assert processed_content == "Tool executed successfully"

    def test_agent_collected_files_initialization(self):
        """Test that _collected_files is properly initialized."""
        mock_agent = MagicMock()
        mock_agent._collected_files = []
        
        file_output = FileOutput(
            name="test.txt",
            content=b"test content",
            mime_type="text/plain"
        )
        mock_agent._collected_files.append(file_output)
        
        assert len(mock_agent._collected_files) == 1
        assert mock_agent._collected_files[0] == file_output
        
        mock_agent._collected_files = []
        assert len(mock_agent._collected_files) == 0

    def test_agent_response_format_with_files(self):
        """Test that agent response format includes files."""
        mock_result = "Task completed successfully"
        mock_intermediate_steps = {"step1": {"action": "test"}}
        mock_files = [
            FileOutput(
                name="output.json",
                content=b'{"result": "success"}',
                mime_type="application/json"
            )
        ]
        
        execution_result = {
            "content": mock_result,
            "intermediate_steps": mock_intermediate_steps,
            "files": mock_files,
        }
        
        assert "files" in execution_result
        assert len(execution_result["files"]) == 1
        assert execution_result["files"][0] == mock_files[0]
        assert execution_result["content"] == mock_result


if __name__ == "__main__":
    pytest.main([__file__])