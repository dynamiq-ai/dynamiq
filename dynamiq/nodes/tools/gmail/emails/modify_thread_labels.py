from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.tools.gmail.emails.utils import GmailLabel
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_MODIFY_THREAD_LABELS = """## Modify Thread Labels Tool
### Description
Modifies Gmail labels on all email within thread using the Gmail API.
You can add or remove Gmail system labels such as INBOX, SPAM, TRASH, UNREAD, STARRED, IMPORTANT,
CATEGORY_PERSONAL, CATEGORY_SOCIAL, CATEGORY_PROMOTIONS, CATEGORY_UPDATES, CATEGORY_FORUMS.

### Parameters
- `thread_id` (str): The ID of the email message to modify.
- `add_label_ids` (list[str], optional): List of GmailLabel enum values to add to the message.
- `remove_label_ids` (list[str], optional): List of GmailLabel enum values to remove from the message.
"""


class ModifyThreadLabelsInputSchema(BaseModel):
    thread_id: str = Field(..., description="The ID of the thread to modify labels for.")
    add_label_ids: list[GmailLabel] | None = Field(None, description="List of label IDs to add to the thread.")
    remove_label_ids: list[GmailLabel] | None = Field(None, description="List of label IDs to remove from the thread.")


class ModifyThreadLabels(ConnectionNode):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "modify-thread-labels"
    description: str = DESCRIPTION_MODIFY_THREAD_LABELS
    input_schema: ClassVar[type[BaseModel]] = ModifyThreadLabelsInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: ModifyThreadLabelsInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Modifies Gmail thread labels by adding or removing specified labels.

        Args:
            input_data (ModifyThreadLabelsInputSchema): Thread ID and label modifications.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: Success status and details of the operation.

        Raises:
            ToolExecutionException: If the label modification fails.
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
                .threads()
                .modify(
                    userId="me",
                    id=input_data.thread_id,
                    body={"addLabelIds": input_data.add_label_ids, "removeLabelIds": input_data.remove_label_ids},
                )
                .execute()
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
