import base64
from email.message import EmailMessage
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.gmail.gmail_base import GmailBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_CREATE_DRAFT_REPLY = """## Create Draft Reply Tool
Creates a draft reply to a specific Gmail message.
### Parameters
- `message_id` (str): The ID of the message to reply to.
- `body` (str): The content of the draft reply.
"""


class CreateDraftReplyInputSchema(BaseModel):
    message_id: str = Field(..., description="The ID of the message to reply to.")
    body: str = Field(..., description="The content of the draft reply.")


class CreateDraftReply(GmailBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "create-draft-reply"
    description: str = DESCRIPTION_CREATE_DRAFT_REPLY
    input_schema: ClassVar[type[BaseModel]] = CreateDraftReplyInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: CreateDraftReplyInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the creation of a draft reply using the Gmail API.

        Args:
            input_data (CreateDraftReplyInputSchema): Contains original message ID and reply body.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: Success status and draft reply details.

        Raises:
            ToolExecutionException: If the operation fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            original = self.client.users().messages().get(userId="me", id=input_data.message_id).execute()
            thread_id = original["threadId"]

            reply_msg = EmailMessage()
            reply_msg.set_content(input_data.body)
            reply_msg["In-Reply-To"] = original["id"]
            reply_msg["References"] = original["id"]

            raw = base64.urlsafe_b64encode(reply_msg.as_bytes()).decode()
            result = (
                self.client.users()
                .drafts()
                .create(userId="me", body={"message": {"raw": raw, "threadId": thread_id}})
                .execute()
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
