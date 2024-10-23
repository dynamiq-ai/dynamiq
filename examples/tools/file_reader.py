import io
from typing import Any, Literal

from pydantic import ConfigDict

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION = """
<tool_description>
This tool reads content from multiple byte streams.
Features:
- Supports both bytes and BytesIO as input
- Handles multiple files in a single operation
Input format:
- List of files with byte content (bytes or BytesIO) with filename and description attributes.
</tool_description>
"""


class FileReaderTool(Node):
    """
    A tool to read the content of one or more files from byte streams.

    Attributes:
        name (str): The name of the tool.
        description (str): The description of the tool.
        files (list[bytes | io.BytesIO] | None): List of files to read content from.
        is_files_allowed (bool): Indicates if the tool supports file operations.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "file-reader-tool"
    description: str = DESCRIPTION
    files: list[bytes | io.BytesIO] | None = None
    is_files_allowed: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _load_file_content(self, file: bytes | io.BytesIO) -> str:
        """
        Load the content of a file from bytes or a byte stream.

        Args:
            file (Union[bytes, io.BytesIO]): The byte content or stream of the file.

        Returns:
            str: The loaded content as string.

        Raises:
            ToolExecutionException: If file reading fails or input type is invalid.
        """
        try:
            file_description = getattr(file, "description", "Unnamed file")
            logger.debug(f"Reading file: {file_description}")

            if isinstance(file, bytes):
                content = file
            elif isinstance(file, io.BytesIO):
                file.seek(0)
                content = file.read()
            else:
                raise ToolExecutionException(
                    f"Invalid input type. Expected bytes or BytesIO, got {type(file)}", recoverable=False
                )

            try:
                return content.decode("utf-8", errors="ignore")
            except UnicodeDecodeError:
                return content.hex()

        except Exception as e:
            logger.error(f"Failed to read file: {file_description}. Error: {e}")
            raise ToolExecutionException(f"Failed to read the file: {file_description}. Error: {e}", recoverable=True)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        Args:
            input_data (dict[str, Any]): The input data containing:
                - files: List of byte streams to process
            config (RunnableConfig, optional): The configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Dictionary containing processed file contents.

        Raises:
            ToolExecutionException: If no files are provided or processing fails.
        """
        logger.debug(f"Tool {self.name} - {self.id}: started with input data {input_data}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        files = input_data.get("files", self.files)

        if not files:
            raise ToolExecutionException(
                "Error: No files provided for reading. Please provide 'files' input.", recoverable=False
            )

        results = {}
        for idx, file in enumerate(files):
            file_description = getattr(file, "description", f"File {idx + 1}")
            content = self._load_file_content(file)
            results[file_description] = content

        logger.debug(f"Tool {self.name} - {self.id}: finished processing files.")
        return {"content": results}
