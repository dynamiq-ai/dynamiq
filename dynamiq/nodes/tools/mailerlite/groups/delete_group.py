from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseMailerLite


class DeleteGroupInputSchema(BaseModel):
    """
    Input schema for deleting a group by ID.
    """

    group_id: str = Field(..., description="ID of the group to delete.")


class DeleteGroup(BaseMailerLite):
    """
    Node to delete a group by ID in .
    """

    name: str = "DeleteGroup"
    description: str = "Delete a group by ID in ."
    input_schema: ClassVar[type[DeleteGroupInputSchema]] = DeleteGroupInputSchema

    def execute(self, input_data: DeleteGroupInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: deleting group ID={input_data.group_id}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + f"groups/{input_data.group_id}"

        try:
            response = self.client.delete(api_url, headers=self.connection.headers)
            if response.status_code != 204:
                response.raise_for_status()
            # No JSON on success (204)
        except Exception as e:
            logger.error(f"{self.name} - {self.id}: failed to delete group. Error: {e}")
            raise ToolExecutionException(
                f"Failed to delete group '{input_data.group_id}'. Error: {str(e)}.",
                recoverable=True,
            )

        return {"content": "Group deleted "}
