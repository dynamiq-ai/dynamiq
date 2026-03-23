from typing import ClassVar

from pydantic import BaseModel, EmailStr, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseMailerLite


class CreateUpsertSubscriberInputSchema(BaseModel):
    """
    Input schema for creating or updating a subscriber.
    If a subscriber with email already exists, it is updated.
    """

    email: EmailStr = Field(..., description="Valid email address for subscriber.")
    # You can store any field: "name", "last_name", etc., or custom fields
    fields: dict = Field(
        default_factory=dict,
        description="Optional dictionary of subscriber fields (e.g., {'name': 'John', 'country': 'USA'}).",
    )
    # The group IDs you'd like to add the subscriber to
    groups: list[str] = Field(
        default_factory=list,
        description="List of existing group IDs to associate this subscriber with.",
    )
    status: str | None = Field(
        default=None,
        description="Optional status: 'active', 'unsubscribed', "
        "'unconfirmed', 'bounced', 'junk'. Defaults to 'active'.",
    )


class CreateUpsertSubscriber(BaseMailerLite):
    """
    Node to create or update (upsert) a subscriber in .
    """

    name: str = "CreateUpsertSubscriber"
    description: str = "Create or update a subscriber (upsert) in ."
    input_schema: ClassVar[type[CreateUpsertSubscriberInputSchema]] = CreateUpsertSubscriberInputSchema

    def execute(self, input_data: CreateUpsertSubscriberInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + "subscribers"

        payload = {
            "email": input_data.email,
            "fields": input_data.fields or {},
            "groups": input_data.groups or [],
        }
        if input_data.status:
            payload["status"] = input_data.status

        try:
            response = self.client.post(
                api_url,
                headers=self.connection.headers,
                json=payload,
            )
            if response.status_code not in (200, 201):
                response.raise_for_status()

            data = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to create/update subscriber. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to create/update subscriber. Error: {str(e)}.",
                recoverable=True,
            )

        return {"content": data}
