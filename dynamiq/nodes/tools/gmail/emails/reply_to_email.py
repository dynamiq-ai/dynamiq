import base64
from email.message import EmailMessage
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_REPLY_TO_EMAIL = """## Reply Email Tool
Replies to an existing Gmail message.

### Parameters
- `message_id` (str): The ID of the email message to reply to.
- `body` (str): The body of the reply message.
"""


class ReplyEmailInputSchema(BaseModel):
    message_id: str = Field(..., description="The ID of the email message to reply to.")
    body: str = Field(..., description="The body of the reply message.")


class ReplyToEmail(ConnectionNode):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "reply-email"
    description: str = DESCRIPTION_REPLY_TO_EMAIL
    input_schema: ClassVar[type[BaseModel]] = ReplyEmailInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: ReplyEmailInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the email reply operation using the Gmail API.

        Args:
            input_data (ReplyEmailInputSchema): Contains message ID and reply body.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: Success status and Gmail API response.

        Raises:
            ToolExecutionException: If the reply fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            original = (
                self.client.users().messages().get(userId="me", id=input_data.message_id, format="metadata").execute()
            )
            thread_id = original["threadId"]
            headers = original["payload"]["headers"]

            # Extract required fields from headers
            from_email = next((h["value"] for h in headers if h["name"].lower() == "from"), None)
            subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "")
            message_id = original["id"]

            if not from_email:
                raise ToolExecutionException("Could not determine sender email to reply to.", recoverable=False)

            reply_msg = EmailMessage()
            reply_msg.set_content(input_data.body)
            reply_msg["To"] = from_email
            reply_msg["Subject"] = f"Re: {subject}"
            reply_msg["In-Reply-To"] = message_id
            reply_msg["References"] = message_id

            raw = base64.urlsafe_b64encode(reply_msg.as_bytes()).decode()
            body = {"raw": raw, "threadId": thread_id}
            result = self.client.users().messages().send(userId="me", body=body).execute()

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            logger.error(f"{self.name} ({self.id}) - Failed to send reply: {e}")
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
