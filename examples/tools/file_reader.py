import io
from typing import Any, Literal

from pydantic import ConfigDict

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import FileDataModel
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION = """
<tool_description>
This tool reads content from multiple byte streams.
Features:
- Supports both bytes and BytesIO as input
- Configurable to display partial content to manage output size
- Handles multiple files in a single operation
Input format:
- List of files with byte content (bytes or BytesIO) and a file description.
</tool_description>
"""


class FileReaderTool(Node):
    """
    A tool to read the content of one or more files from byte streams.

    Attributes:
        name (str): The name of the tool.
        description (str): The description of the tool.
        files (list[FileDataModel] | None): List of files to read content from.
        show_partial (bool): Whether to show only a part of the content to avoid overwhelming the output.
        max_display_length (int): Maximum number of characters/bytes to display in the output.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "file-reader-tool"
    description: str = DESCRIPTION
    files: list[FileDataModel] | None = None
    show_partial: bool = True
    max_display_length: int = 500
    supports_files: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _load_file_content(self, file: bytes | io.BytesIO, file_description: str) -> str:
        """
        Load the content of a file from bytes or a byte stream.

        Args:
            file (Union[bytes, io.BytesIO]): The byte content or stream of the file.
            file_description (str): Description of the file (for logging or reference).

        Returns:
            str: The loaded content, either fully or partially (based on configuration).
        """
        try:
            logger.debug(f"Reading file: {file_description or 'Unnamed file'}")

            # If the input is bytes
            if isinstance(file, bytes):
                content = file

            # If the input is a BytesIO object
            elif isinstance(file, io.BytesIO):
                file.seek(0)  # Ensure we're at the start of the BytesIO stream
                content = file.read()

            else:
                raise ToolExecutionException(
                    f"Invalid input type. Expected bytes or BytesIO, got {type(file)}", recoverable=False
                )

            # Return partial or full content based on configuration
            return self._get_partial_content(content)

        except Exception as e:
            logger.error(f"Failed to read file: {file_description or 'Unnamed file'}. Error: {e}")
            raise ToolExecutionException(
                f"Failed to read the file: {file_description or 'Unnamed file'}. Error: {e}", recoverable=True
            )

    def _get_partial_content(self, content: bytes) -> str:
        """
        Returns part of the content for display, limited by max_display_length.

        Args:
            content (bytes): The content to be partially displayed.

        Returns:
            str: Partial content to display.
        """
        try:
            # Attempt to decode as UTF-8, fallback to hex for binary data
            content_str = content.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            content_str = content.hex()

        # Return partial content if content exceeds max_display_length
        if self.show_partial and len(content_str) > self.max_display_length:
            return content_str[: self.max_display_length] + "..."  # Append ellipsis for truncated content
        else:
            return content_str

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        Args:
            input_data (dict[str, Any]): The input data containing a list of files (byte streams).
            config (RunnableConfig, optional): The configuration for the runnable. Defaults to None.

        Returns:
            dict[str, Any]: The content of the files or an error message if reading fails.
        """
        logger.debug(f"Tool {self.name} - {self.id}: started with input data {input_data}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        files = input_data.get("files", self.files)
        self.show_partial = input_data.get("show_partial", self.show_partial)
        self.max_display_length = input_data.get("max_display_length", self.max_display_length)

        if not files:
            raise ToolExecutionException(
                "Error: No files provided for reading. Please provide 'files' input.", recoverable=False
            )

        results = {}

        # Iterate over each file and read its content
        for idx, file_model in enumerate(files):
            file_description = file_model.description or f"File {idx + 1}"
            content = self._load_file_content(file_model.file, file_description)
            results[file_description] = content

        logger.debug(f"Tool {self.name} - {self.id}: finished processing files.")
        return {"content": results}
