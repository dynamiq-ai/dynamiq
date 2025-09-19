from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file.base import FileStore
from dynamiq.utils.logger import logger


class FileReadInputSchema(BaseModel):
    """Schema for file read input parameters."""

    file_path: str = Field(default="", description="Path of the file to read")


class FileWriteInputSchema(BaseModel):
    """Schema for file write input parameters."""

    file_path: str = Field(..., description="Path where the file should be written")
    content: bytes | str = Field(
        ..., description="File content (string, bytes)"
    )
    content_type: str | None = Field(default=None, description="MIME type (auto-detected if not provided)")
    metadata: str | None = Field(default=None, description="Additional metadata for the file")


class FileReadTool(Node):
    """
    A tool for reading files from storage.

    This tool can be passed to ReAct agents to read files
    from the configured storage backend. For large files, it automatically
    returns chunked content showing first, middle, and last parts.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        file_store (FileStore): File storage to read from.
        max_size (int): Maximum size in bytes before chunking (default: 10000).
        chunk_size (int): Size of each chunk in bytes (default: 1000).
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "FileReadTool"
    description: str = """
        Reads files from storage based on the provided file path.
        For large files (configurable threshold), returns first, middle, and last chunks as bytes with separators.

        Usage Examples:
            - Read small file: {"file_path": "config.txt"}
            - Read large file: {"file_path": "large_data.json"}
            - Read huge file: {"file_path": "huge_file.csv"}

        Parameters:
            - file_path: Path of the file to read
    """

    file_store: FileStore = Field(..., description="File storage to read from.")
    max_size: int = Field(default=10000, description="Maximum size in bytes before chunking (default: 10000)")
    chunk_size: int = Field(default=1000, description="Size of each chunk in bytes (default: 1000)")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[FileReadInputSchema]] = FileReadInputSchema

    def execute(
        self,
        input_data: FileReadInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Executes the file read operation and returns the file content.
        For large files, returns first, middle, and last chunks instead of full content.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            if not self.file_store.exists(input_data.file_path):
                raise ToolExecutionException(
                    f"File '{input_data.file_path}' not found",
                    recoverable=True,
                )

            content = self.file_store.retrieve(input_data.file_path)
            content_size = len(content)

            # If file is small enough, return full content
            if content_size <= self.max_size:
                logger.info(f"Tool {self.name} - {self.id}: returning full content ({content_size} bytes)")
                return {"content": content}

            # For large files, return chunked content
            logger.info(f"Tool {self.name} - {self.id}: file is large ({content_size} bytes), returning chunks")

            chunked_content = self._create_chunked_content(content, self.chunk_size, input_data.file_path)

            return {"content": chunked_content}

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to read file. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to read file. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

    def _create_chunked_content(self, content: bytes, chunk_size: int, file_path: str) -> bytes:
        """
        Create chunked content showing first, middle, and last parts of a large file.

        Args:
            content: The file content as bytes
            chunk_size: Size of each chunk in bytes
            file_path: Path of the file being read

        Returns:
            Concatenated bytes containing first, middle, and last chunks
        """
        total_size = len(content)

        first_chunk = content[:chunk_size]

        middle_start = total_size // 2 - chunk_size // 2
        middle_chunk = content[middle_start : middle_start + chunk_size]

        last_chunk = content[-chunk_size:] if total_size > chunk_size else content

        separator = f"\n\n--- CHUNKED FILE: {file_path} ({total_size:,} bytes total) ---\n".encode()
        first_sep = f"\n--- FIRST {len():,} BYTES ---\n".encode("utf-8")
        middle_sep = f"\n--- MIDDLE {len():,} BYTES (from position {middle_start:,}) ---\n".encode("utf-8")
        last_sep = f"\n--- LAST {len():,} BYTES ---\n".encode("utf-8")

        chunked_bytes = (
            separator
            + first_sep
            + first_chunk
            + middle_sep
            + middle_chunk
            + last_sep
            + last_chunk
            + b"\n\n--- END OF CHUNKED FILE ---\n"
        )

        return chunked_bytes


class FileWriteTool(Node):
    """
    A tool for writing files to storage.

    This tool can be passed to ReAct agents to write files
    to the configured storage backend.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        file_store (FileStore): File storage to write to.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "FileWriteTool"
    description: str = """Writes files to storage based on the provided file path and content.

    Usage Examples:
    - Write text: {"file_path": "readme.txt", "content": "Hello World"}
    - Write JSON: {"file_path": "config.json", "content": {"key": "value"}}
    - Overwrite file: {"file_path": "existing.txt", "content": "new content"}"""

    file_store: FileStore = Field(..., description="File storage to write to.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[FileWriteInputSchema]] = FileWriteInputSchema

    def execute(
        self,
        input_data: FileWriteInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Executes the file write operation and returns the file information.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            content_str = input_data.content
            if input_data.content_type is None:
                content_type = "text/plain"
            else:
                content_type = input_data.content_type

            # Store file
            file_info = self.file_store.store(
                input_data.file_path,
                content_str,
                content_type=content_type,
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(file_info)[:200]}...")
            return {"content": f"File '{input_data.file_path}' written successfully"}

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to write file. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to write file. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )


class FileListInputSchema(BaseModel):
    """Schema for file list input parameters."""

    file_path: str = Field(
        default="", description="Path of the file to list. Default is the root path. Keep empty to list all files."
    )
    recursive: bool = Field(default=True, description="Whether to list files recursively. Default is True.")


class FileListTool(Node):
    """
    A tool for listing files in storage.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "FileListTool"
    description: str = """Lists files in storage based on the provided file path."""

    file_store: FileStore = Field(..., description="File storage to list from.")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[FileListInputSchema]] = FileListInputSchema

    def execute(
        self,
        input_data: FileListInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:

        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            files_list = self.file_store.list_files(directory=input_data.file_path, recursive=input_data.recursive)
            files_string = "Files currently available in the filesystem storage:\n"
            for file in files_list:
                files_string += f"File: {file.name} | Path: {file.path} | Size: {file.size} bytes\n"

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{files_string}")
            return {"content": files_string}

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to list files. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to list files. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )
