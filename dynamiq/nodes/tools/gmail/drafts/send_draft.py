from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.gmail.gmail_base import GmailBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_SENT_DRAFT = """## Send Draft Tool
Sends an existing Gmail draft.

### Parameters
- `draft_id` (str): The ID of the Gmail draft to send.
"""


class SendDraftInputSchema(BaseModel):
    draft_id: str = Field(..., description="The ID of the Gmail draft to send.")


class SendDraft(GmailBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "send-draft"
    description: str = DESCRIPTION_SENT_DRAFT
    input_schema: ClassVar[type[BaseModel]] = SendDraftInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: SendDraftInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the sending of a Gmail draft.

        Args:
            input_data (SendDraftInputSchema): Contains draft ID.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: Success status and send result.

        Raises:
            ToolExecutionException: If sending fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            result = self.client.users().drafts().send(userId="me", body={"id": input_data.draft_id}).execute()
            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
