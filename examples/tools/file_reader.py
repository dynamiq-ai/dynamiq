from typing import Any, Literal

from pydantic import ConfigDict

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from dynamiq.nodes.agents.exceptions import ToolExecutionException


class FileReadTool(Node):
    """
    A tool to read the content of a file.

    Attributes:
        name (str): The name of the tool.
        description (str): The description of the tool.
        file_path (str): The file path to read.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Read a file's content"
    description: str = "A tool that can be used to read a file's content from local storage."
    file_path: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _read_file_content(self, file_path: str) -> Any:
        """
        Read the content of the file at the given path.

        Args:
            file_path (str): The path to the file.

        Returns:
            Any: The content of the file.
        """
        try:
            with open(file_path) as file:
                return file.read()
        except Exception as e:
            logger.error(
                f"Tool {self.name} - {self.id}: failed to to read the file {file_path}. Error: {e}"
            )
            raise ToolExecutionException(
                f"Failed to to read the file {file_path}. Error: {e}", recoverable=True
            )

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        Args:
            input_data (dict[str, Any]): The input data containing the file path.
            config (RunnableConfig, optional): The configuration for the runnable. Defaults to None.

        Returns:
            dict[str, Any]: The content of the file or an error message if reading fails.
        """
        logger.debug(
            f"Tool {self.name} - {self.id}: started with input data {input_data}"
        )

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        file_path = input_data.get("file_path", self.file_path)
        result = self._read_file_content(file_path)

        logger.debug(f"Tool {self.name} - {self.id}: finished with result {result}")
        return {"content": result}
