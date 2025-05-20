from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.memory.agent_context import Context, ContextEntry
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

CONTEXT_WRITER_TOOL_DESCRIPTION = """
Context Writer Tool

Writes a memory entries to context. Use this tool to record important information
that may be useful in the future, such as facts, events, decisions, user preferences,
or observations made during the conversation or task execution.

Parameters:
- key (str): A unique identifier for this memory entry. Use a descriptive, stable key
  (e.g., "user_name", "project_goal", "preferred_language").
- data (str): The memory content to store. This can be a sentence, paragraph, or list
  of facts written in natural language.
- description (str | optional): Additional metadata to help organize the memory. This is required to add more
  context of what is saved under specific key.

Guidance:
- Use this tool whenever you encounter new, potentially useful information that may be
  relevant later in the conversation or task.
- Do not repeat or re-write existing memories unless updating or refining them.
- Keep memory entries concise and informative.
- If unsure whether to store something, err on the side of writing it.
"""  # noqa: E501


CONTEXT_RETRIEVER_TOOL_DESCRIPTION = """
Context Retriever Tool

Retrieves relevant memory entries from long-term storage based on a query or context.
Use this tool when you need to recall previously stored information that may help
answer a question, continue a conversation, or inform decision-making.

Parameters:
- key (str): The unique identifier of the memory to retrieve. This must match a key used
  previously with the memory writer tool.

Returns:
- The memory content stored under the key.

Guidance:
- Use this tool when context from earlier interactions or background knowledge might help.
- Prefer specific, goal-oriented queries over vague ones.
- You may rephrase user questions into better search queries if needed.
- Retrieved memories can be used directly in your response or to refine your reasoning.

"""  # noqa: E501


class ContextWriterToolInputSchema(BaseModel):
    key: str
    data: str
    description: str


class ContextWriterTool(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "ContextWriterTool"
    backend: Context = Field(default_factory=Context)

    description: str = CONTEXT_WRITER_TOOL_DESCRIPTION
    input_schema: ClassVar[type[ContextWriterToolInputSchema]] = ContextWriterToolInputSchema

    def execute(
        self, input_data: ContextWriterToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Executes the requested action based on the input data."""
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        entry = ContextEntry(key=input_data.key, data=input_data.data, description=input_data.description)
        self.backend.add_entry(entry)

        logger.info(f"Tool {self.name} - {self.id}: finished.")
        return {"content": "Content was successfully saved"}


class ContextRetrieverToolInputSchema(BaseModel):
    key: str = Field(default="", description="")


class ContextRetrieverTool(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "ContextRetrieverTool"
    backend: Context = Field(default_factory=Context)
    description: str = CONTEXT_RETRIEVER_TOOL_DESCRIPTION
    input_schema: ClassVar[type[ContextRetrieverToolInputSchema]] = ContextRetrieverToolInputSchema

    def execute(
        self, input_data: ContextRetrieverToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Executes the requested action based on the input data."""
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        content = self.backend.get_entry(input_data.key)

        logger.info(f"Tool {self.name} - {self.id}: finished with result: {content}")
        return {"content": content}
