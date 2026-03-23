from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseMailerLite


class AssignSubscriberToGroupInputSchema(BaseModel):
    """
    Input schema for assigning a subscriber to a group.
    """

    subscriber_id: str = Field(..., description="An existing subscriber ID.")
    group_id: str = Field(..., description="An existing group ID.")


class AssignSubscriberToGroup(BaseMailerLite):
    """
    Node to assign (add) a subscriber to a group in .
    """

    name: str = "AssignSubscriberToGroup"
    description: str = "Assign an existing subscriber to a group."
    input_schema: ClassVar[type[AssignSubscriberToGroupInputSchema]] = AssignSubscriberToGroupInputSchema

    def execute(self, input_data: AssignSubscriberToGroupInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: assigning subscriber to group:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + f"subscribers/{input_data.subscriber_id}/groups/{input_data.group_id}"

        try:
            response = self.client.post(api_url, headers=self.connection.headers)
            if response.status_code not in (200, 201):
                response.raise_for_status()

            data = response.json()
        except Exception as e:
            logger.error(f"{self.name} - {self.id}: failed to assign subscriber to group. Error: {e}")
            raise ToolExecutionException(
                f"Failed to assign subscriber "
                f"'{input_data.subscriber_id}' to group '{input_data.group_id}'. Error: {str(e)}.",
                recoverable=True,
            )

        return {"content": data}
