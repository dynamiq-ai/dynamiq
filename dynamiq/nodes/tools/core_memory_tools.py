from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class CoreMemoryWriteInputSchema(BaseModel):
    block_name: str = Field(..., description="Name of the memory block")
    content: str = Field(..., description="Content to write to the memory block")


class CoreMemoryWriteTool(Node):
    """
    A tool for writing to the core memory of an agent.

    This tool allows setting the content of a specific memory block.
    """
    group: str = "tools"
    name: str = "CoreMemoryWriteTool"
    description: str = """
        Writes content to a specific memory block, overwriting any existing content.
        Input: block_name (string) - Name of the memory block, content (string) - The content to write.
        Returns confirmation of the write operation.
    """
    agent: Any = Field(..., description="The agent whose core memory to write to.")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[CoreMemoryWriteInputSchema]] = CoreMemoryWriteInputSchema

    def execute(
        self,
        input_data: CoreMemoryWriteInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Executes the core memory write operation.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            self.agent._core_memory.add_block(input_data.block_name, input_data.content)
            result = f"Created new memory block '{input_data.block_name}' with content."

            self.agent._prompt_variables["core_memory"] = self.agent._core_memory.get_formatted_memory()
            self.agent.update_system_message()

            logger.info(f"Tool {self.name} - {self.id}: finished successfully")
            return {"content": result}

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to write to core memory. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to write to core memory. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )


class CoreMemoryRemoveInputSchema(BaseModel):
    block_name: str = Field(..., description="Name of the memory block to remove")


class CoreMemoryRemoveTool(Node):
    """
    A tool for clearing the core memory of an agent.

    This tool allows clearing content from memory blocks.
    """

    group: str = "tools"
    name: str = "CoreMemoryRemoveTool"
    description: str = """
        Removes a memory block from the core memory.
        Input: block_name (string) - Name of the memory block to remove.
        Returns confirmation of the remove operation.
    """

    agent: Any = Field(..., description="The agent whose core memory to clear.")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[CoreMemoryRemoveInputSchema]] = CoreMemoryRemoveInputSchema

    def execute(
        self,
        input_data: CoreMemoryRemoveInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Executes the core memory clear operation.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            self.agent._core_memory.remove_block(input_data.block_name)
            result = f"Removed memory block '{input_data.block_name}'."

            self.agent._prompt_variables["core_memory"] = self.agent._core_memory.get_formatted_memory()
            self.agent.update_system_message()

            logger.info(f"Tool {self.name} - {self.id}: finished successfully")
            return {"content": result}

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to remove core memory block. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to remove core memory block. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )
