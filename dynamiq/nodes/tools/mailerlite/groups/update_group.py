from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseMailerLite


class UpdateGroupInputSchema(BaseModel):
    """
    Input schema for updating a group by ID.
    """

    group_id: str = Field(..., description="ID of the group to update.")
    name: str = Field(..., description="New name for the group (<= 255 chars).")


class UpdateGroup(BaseMailerLite):
    """
    Node to update an existing group in .
    """

    name: str = "UpdateGroup"
    description: str = "Update an existing group in ."
    input_schema: ClassVar[type[UpdateGroupInputSchema]] = UpdateGroupInputSchema

    def execute(self, input_data: UpdateGroupInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: updating group with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + f"groups/{input_data.group_id}"
        payload = {"name": input_data.name}

        try:
            response = self.client.put(api_url, headers=self.connection.headers, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"{self.name} - {self.id}: failed to update group. Error: {e}")
            raise ToolExecutionException(
                f"Failed to update group '{input_data.group_id}'. Error: {str(e)}.",
                recoverable=True,
            )

        return {"content": data}
