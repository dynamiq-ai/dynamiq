from io import BytesIO
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.operators import Choice
from dynamiq.nodes.types import ChoiceCondition
from dynamiq.runnables import RunnableConfig


class FileFilterInputSchema(BaseModel):
    files: list[BytesIO] = Field(default=None, description="Parameter to provide files.")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FileFilter(Node):
    group: Literal[NodeGroup.FILTERS] = NodeGroup.FILTERS
    name: str = "File Filter"
    description: str = "Node that returns filtered list of files"
    filters: ChoiceCondition

    input_schema: ClassVar[type[FileFilterInputSchema]] = FileFilterInputSchema

    def execute(self, input_data: FileFilterInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Node for filtering files based on specified filter conditions.

        Args:
            input_data (FileFilterInputSchema): input data for the tool, which includes list for files, filters and
                metadata.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing filtered files.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        files = input_data.files
        filters = self.filters
        try:
            filtered_files = []
            for file_data in files:
                file_info = file_data.__dict__
                if Choice.evaluate(filters, file_info):
                    filtered_files.append(file_data)

            return {"output": filtered_files}
        except Exception as e:
            raise ValueError(f"Error while executing filters: {e}")
