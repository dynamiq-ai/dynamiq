from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_MARK_AS_READ = """## Mark As Read Tool
### Description
Marks a Gmail message as read by removing the UNREAD label.

### Parameters
- `message_id` (str): The ID of the message to mark as read.
"""


class MarkAsReadInputSchema(BaseModel):
    message_id: str = Field(..., description="The ID of the message to mark as read.")


class MarkAsRead(ConnectionNode):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "mark-as-read"
    description: str = DESCRIPTION_MARK_AS_READ
    input_schema: ClassVar[type[BaseModel]] = MarkAsReadInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: MarkAsReadInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Marks a Gmail message as read by removing the UNREAD label.

        Args:
            input_data (MarkAsReadInputSchema): Message ID to mark as read.
            config (RunnableConfig | None): Optional configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: Success status and API response.

        Raises:
            ToolExecutionException: If marking as read fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            result = (
                self.client.users()
                .messages()
                .modify(userId="me", id=input_data.message_id, body={"removeLabelIds": ["UNREAD"]})
                .execute()
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
