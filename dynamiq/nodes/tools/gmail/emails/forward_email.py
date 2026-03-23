import base64
from email import message_from_bytes
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

DESCRIPTION_FORWARD_EMAIL = """## Forward Email Tool
Forwards a Gmail message to another recipient.

### Parameters
- `message_id` (str): The ID of the email message to forward.
- `to` (str): The recipient to forward the message to.
"""


class ForwardEmailInputSchema(BaseModel):
    message_id: str = Field(..., description="The ID of the email message.")
    to: str = Field(..., description="The recipient to forward the message to.")


class ForwardEmail(GmailBase):
    """
    Forwards a Gmail message to another recipient.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "forward-email"
    description: str = "Forwards a Gmail message to another recipient."
    input_schema: ClassVar[type[BaseModel]] = ForwardEmailInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: ForwardEmailInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the email forwarding operation using the Gmail API.

        Args:
            input_data (ForwardEmailInputSchema): The input data containing the ID of the email
                message to forward and the recipient's email address.
            config (RunnableConfig | None): Optional execution configuration containing
                callback hooks and metadata.
            **kwargs: Additional keyword arguments passed to the node execution lifecycle.

        Returns:
            dict[str, Any]: A dictionary containing the success status and Gmail API response details.

        Raises:
            ToolExecutionException: Raised when an error occurs during the email forwarding process.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        try:
            original = self.client.users().messages().get(userId="me", id=input_data.message_id, format="raw").execute()
            raw_encoded = original["raw"]

            raw_bytes = base64.urlsafe_b64decode(raw_encoded.encode("utf-8"))
            email_msg = EmailMessage()
            email_msg.set_content(raw_bytes.decode("utf-8", errors="ignore"))

            parsed_msg = message_from_bytes(raw_bytes)
            parsed_msg.replace_header("To", input_data.to)
            new_raw = base64.urlsafe_b64encode(parsed_msg.as_bytes()).decode("utf-8")

            result = self.client.users().messages().send(userId="me", body={"raw": new_raw}).execute()

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            logger.error(f"{self.name} ({self.id}) - Failed to forward message: {e}")
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
