from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseMailerLite


class GetSubscriberInputSchema(BaseModel):
    """
    Input schema for retrieving a single subscriber by ID or email.
    """

    subscriber_identifier: str = Field(
        ..., description="Subscriber ID or email to fetch. E.g., 'some@example.com' or '31986843064993537'."
    )


class GetSubscriber(BaseMailerLite):
    """
    Node to fetch a single subscriber by ID or email.
    """

    name: str = "GetSubscriber"
    description: str = "Retrieve a subscriber by ID or email from ."
    input_schema: ClassVar[type[GetSubscriberInputSchema]] = GetSubscriberInputSchema

    def execute(self, input_data: GetSubscriberInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + f"subscribers/{input_data.subscriber_identifier}"

        try:
            response = self.client.get(
                api_url,
                headers=self.connection.headers,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get subscriber. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to get subscriber. Error: {str(e)}.",
                recoverable=True,
            )

        return {"content": data}
