import json
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file_storage.base import FileStorage
from dynamiq.utils.logger import logger


class FileReadInputSchema(BaseModel):
    """Schema for file read input parameters."""
    file_path: str = Field(description="Path of the file to read")


class FileWriteInputSchema(BaseModel):
    """Schema for file write input parameters."""    
    file_path: str = Field(description="Path where the file should be written")
    content: Union[str, bytes, Dict, List, Any] = Field(description="File content (string, bytes, or serializable object)")
    content_type: Optional[str] = Field(default=None, description="MIME type (auto-detected if not provided)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the file")

class FileReadTool(Node):   
    """
    A tool for reading files from storage.

    This tool can be passed to ReAct agents to read files
    from the configured storage backend.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        storage_type (str): Type of storage to use ("in_memory" or "file_system").
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "FileReadTool"
    description: str = """
        Reads files from storage based on the provided file path.
        Usage Examples:
            - Read text file: {"file_path": "config.txt"}
            - Read JSON file: {"file_path": "data.json"}
    """

    file_storage: FileStorage = Field(..., description="File storage to read from.")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[FileReadInputSchema]] = FileReadInputSchema
    
    def execute(self, input_data: FileReadInputSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """
        Executes the file read operation and returns the file content.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            if not self.file_storage.exists(input_data.file_path):
                raise ToolExecutionException(
                    f"File '{input_data.file_path}' not found",
                    recoverable=True,
                )
            
            content = self.file_storage.retrieve(input_data.file_path)

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(content)[:200]}...")
            return {"content": content}

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to read file. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to read file. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

class FileWriteTool(Node):  
    """
    A tool for writing files to storage.

    This tool can be passed to ReAct agents to write files
    to the configured storage backend.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        file_storage (FileStorage): File storage to write to.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "File Write Tool"
    description: str = """Writes files to storage based on the provided file path and content.

    Usage Examples:
    - Write text: {"file_path": "readme.txt", "content": "Hello World"}
    - Write JSON: {"file_path": "config.json", "content": {"key": "value"}}
    - Overwrite file: {"file_path": "existing.txt", "content": "new content"}"""

    file_storage: FileStorage = Field(..., description="File storage to write to.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[FileWriteInputSchema]] = FileWriteInputSchema


    def execute(self, input_data: FileWriteInputSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
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
            file_info = self.file_storage.store(
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
