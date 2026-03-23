from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseMailerLite


class CreateGroupInputSchema(BaseModel):
    """
    Input schema for creating a group.
    """

    name: str = Field(..., description="Name of the new group (<= 255 chars).")


class CreateGroup(BaseMailerLite):
    """
    Node to create a new group in .
    """

    name: str = "CreateGroup"
    description: str = "Create a new group in ."
    input_schema: ClassVar[type[CreateGroupInputSchema]] = CreateGroupInputSchema

    def execute(self, input_data: CreateGroupInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: creating group with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + "groups"
        payload = {"name": input_data.name}

        try:
            response = self.client.post(
                api_url,
                headers=self.connection.headers,
                json=payload,
            )
            if response.status_code != 201:
                response.raise_for_status()

            data = response.json()
        except Exception as e:
            logger.error(f"{self.name} - {self.id}: failed to create group. Error: {e}")
            raise ToolExecutionException(
                f"Failed to create group. Error: {str(e)}.",
                recoverable=True,
            )

        return {"content": data}
