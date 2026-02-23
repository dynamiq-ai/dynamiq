from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class MessageType(str, Enum):
    ANSWER = "answer"


class MessageToolInputSchema(BaseModel):
    type: MessageType = Field(..., description="The type of message")
    text: str = Field(..., description="The text content of the message")
    files: list[str] = Field(
        default_factory=list,
        description="List of file paths to return with the message",
    )


class MessageToolOutput(BaseModel):
    answer: str = ""
    files: list[str] = Field(default_factory=list)


class MessageTool(Node):
    """Tool that packages a text message with optional file paths to return."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "message_tool"
    description: str = (
        "Returns a message to the user along with optional file paths. "
        "Use this tool when you need to provide a final response that includes generated files.\n\n"
        "Parameters:\n"
        '- type: The message type (required, must be "answer")\n'
        '- text: The message text (required)\n'
        '- files: JSON list of file paths to return (optional)\n\n'
        "Examples:\n"
        '- {"type": "answer", "text": "Here is your report.", "files": ["/output/report.pdf"]}\n'
        '- {"type": "answer", "text": "Task complete. No files generated."}'
    )
    error_handling: ErrorHandling = Field(
        default_factory=lambda: ErrorHandling(timeout_seconds=600),
    )

    input_schema: ClassVar[type[MessageToolInputSchema]] = MessageToolInputSchema

    def execute(
        self,
        input_data: MessageToolInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        logger.info(
            f"Tool {self.name} - {self.id}: "
            f"message with {len(input_data.files)} file path(s)"
        )

        return {
            "type": input_data.type.value,
            "content": input_data.text,
            "files": input_data.files,
        }
