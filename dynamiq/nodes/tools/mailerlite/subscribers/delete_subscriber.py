from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseMailerLite


class DeleteSubscriberInputSchema(BaseModel):
    """
    Input schema for deleting a single subscriber by ID.
    (According to  docs, the endpoint is: DELETE /subscribers/:id)
    """

    subscriber_id: str = Field(..., description="Subscriber ID to delete.")


class DeleteSubscriber(BaseMailerLite):
    """
    Node to delete a subscriber by ID in .
    """

    name: str = "DeleteSubscriber"
    description: str = "Delete a subscriber by ID in ."
    input_schema: ClassVar[type[DeleteSubscriberInputSchema]] = DeleteSubscriberInputSchema

    def execute(self, input_data: DeleteSubscriberInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + f"subscribers/{input_data.subscriber_id}"

        try:
            response = self.client.delete(
                api_url,
                headers=self.connection.headers,
            )
            if response.status_code != 204:
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to delete subscriber. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to delete subscriber. Error: {str(e)}.",
                recoverable=True,
            )

        return {"content": "Subscriber deleted."}
