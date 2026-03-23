from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.tools.gmail.emails.utils import GmailLabel
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_MODIFY_EMAIL_LABELS = """## Modify Email Labels Tool
### Description
Modifies Gmail labels on a specific email message using the Gmail API.
You can add or remove Gmail system labels such as INBOX, SPAM, TRASH, UNREAD, STARRED, IMPORTANT,
CATEGORY_PERSONAL, CATEGORY_SOCIAL, CATEGORY_PROMOTIONS, CATEGORY_UPDATES, CATEGORY_FORUMS.

### Parameters
- `message_id` (str): The ID of the email message to modify.
- `add_label_ids` (list[str], optional): List of GmailLabel enum values to add to the message.
- `remove_label_ids` (list[str], optional): List of GmailLabel enum values to remove from the message.
"""


class ModifyLabelsInputSchema(BaseModel):
    message_id: str = Field(..., description="The ID of the email message.")
    add_label_ids: list[GmailLabel] | None = Field(None, description="List of label IDs to add to the message.")
    remove_label_ids: list[GmailLabel] | None = Field(None, description="List of label IDs to remove from the message.")


class ModifyEmailLabels(ConnectionNode):
    """
    Tool to modify labels on a Gmail message.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "modify-email-labels"
    description: str = DESCRIPTION_MODIFY_EMAIL_LABELS
    input_schema: ClassVar[type[BaseModel]] = ModifyLabelsInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: ModifyLabelsInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the label modification operation on a Gmail message.

        Args:
            input_data (ModifyLabelsInputSchema): Contains message ID, list of labels to add, and list to remove.
            config (RunnableConfig | None): Optional runtime configuration for node execution.
            **kwargs: Additional arguments passed to the execution.

        Returns:
            dict[str, Any]: Dictionary with status and Gmail API response.

        Raises:
            ToolExecutionException: If any error occurs during the label modification process.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if input_data.add_label_ids is None:
            input_data.add_label_ids = []
        if input_data.remove_label_ids is None:
            input_data.remove_label_ids = []

        try:
            result = (
                self.client.users()
                .messages()
                .modify(
                    userId="me",
                    id=input_data.message_id,
                    body={"addLabelIds": input_data.add_label_ids, "removeLabelIds": input_data.remove_label_ids},
                )
                .execute()
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}
        except Exception as e:
            logger.error(f"{self.name} ({self.id}) - Failed to modify labels: {e}")
            raise ToolExecutionException(str(e), recoverable=False)

    def close(self):
        self.client.close()
