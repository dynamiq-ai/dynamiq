from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION = """
<tool_description>
This tool reads content from multiple files or byte streams.
Features:
- Supports both local file paths and byte streams as input
- Configurable to display partial content to manage output size
- Handles multiple files in a single operation
Input format:
- List of tuples, where each tuple represents a file:
  - First element: File path (str) or file content (bytes)
  - Second element: File description (str)
</tool_description>
"""


class FileReadTool(Node):
    """
    A tool to read the content of one or more files from either file paths or byte streams.

    Attributes:
        name (str): The name of the tool.
        description (str): The description of the tool.
        files (list[tuple[Union[str, bytes], str]]): List of tuples where each tuple contains
            either a file path (str) or file content (bytes) as the first element,
            and a file description (str) as the second element.
        show_partial (bool): Whether to show only a part of the content to avoid overwhelming the output.
        max_display_length (int): Maximum number of characters/bytes to display in the output.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "file-reader-tool"
    description: str = "A tool that can be used to read content from files (from local storage or byte streams)."
    files: list[tuple[str | bytes, str]] | None = None
    show_partial: bool = True
    max_display_length: int = 500
    support_files: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _load_file_content(self, file_data: str | bytes, file_description: str) -> str:
        """
        Load the content of a single file from a path or byte stream.

        Args:
            file_data (Union[str, bytes]): The path to the file or byte content.
            file_description (str): Description of the file (for logging or reference).

        Returns:
            str: The loaded content, either fully or partially (based on configuration).
        """
        try:
            logger.debug(f"Reading file: {file_description or 'Unnamed file'}")

            # If the input is a file path
            if isinstance(file_data, str):
                file_path = Path(file_data).resolve()
                if not file_path.exists():
                    raise ToolExecutionException(f"Error: File not found at {file_path}", recoverable=False)
                if not file_path.is_file():
                    raise ToolExecutionException(f"Error: {file_path} is not a valid file", recoverable=False)

                with file_path.open("rb") as file:
                    content = file.read()

            # If the input is a bytes object
            elif isinstance(file_data, bytes):
                content = file_data

            else:
                raise ToolExecutionException(
                    f"Invalid input type. Expected file path or bytes, got {type(file_data)}", recoverable=False
                )

            # Return partial content if required
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
            content_str = content.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            content_str = content.hex()

        return content_str[: self.max_display_length] + ("..." if len(content_str) > self.max_display_length else "")

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        Args:
            input_data (dict[str, Any]): The input data containing a list of files (paths or bytes).
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

        for idx, (file_data, file_description) in enumerate(files):
            file_description = file_description or f"File {idx + 1}"
            content = self._load_file_content(file_data, file_description)
            results[file_description] = content

        logger.debug(f"Tool {self.name} - {self.id}: finished processing files.")
        return {"content": results}
