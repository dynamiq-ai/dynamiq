from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.gmail.gmail_base import GmailBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_ARCHIVE_EMAIL = """## Archive Email Tool
Archives a Gmail message by removing it from the inbox.

### Parameters
- `message_id` (str): The ID of the email message to archive.
"""


class ArchiveEmailInputSchema(BaseModel):
    message_id: str = Field(..., description="The ID of the email message.")


class ArchiveEmail(GmailBase):
    """
    Archives a Gmail message (removes it from inbox).
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "archive-email"
    description: str = DESCRIPTION_ARCHIVE_EMAIL
    input_schema: ClassVar[type[BaseModel]] = ArchiveEmailInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: ArchiveEmailInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Archives the email using the Gmail API.

        Args:
            input_data (ArchiveEmailInputSchema): The input data containing the ID of the email
                message to be archived.
            config (RunnableConfig | None): Optional execution configuration containing
                callback hooks and metadata.
            **kwargs: Additional keyword arguments passed to the node execution lifecycle.

        Returns:
            dict[str, Any]: A dictionary containing the success status and Gmail API response details.

        Raises:
            ToolExecutionException: Raised when an error occurs during the email archiving process.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        try:
            result = (
                self.client.users()
                .messages()
                .modify(userId="me", id=input_data.message_id, body={"removeLabelIds": ["INBOX"]})
                .execute()
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}
        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
