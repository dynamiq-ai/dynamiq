from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_calendar.google_calendar_base import GoogleCalendarBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_GET_EVENT = """## Get Calendar Event by ID Tool

Retrieves details of a specific event from Google Calendar by its event ID.

### Parameters
- `calendar_id` (str, optional, default: "primary"): The ID of the calendar where the event will be created.
- `event_id` (str): The unique identifier of the event to retrieve.
"""


class RetrieveEventByIdInputSchema(BaseModel):
    calendar_id: str = Field(default="primary", description="Calendar ID (default: 'primary').")
    event_id: str = Field(..., description="ID of the event to retrieve.")


class RetrieveEventById(GoogleCalendarBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "get-event-by-id"
    description: str = DESCRIPTION_GET_EVENT
    input_schema: ClassVar[type[BaseModel]] = RetrieveEventByIdInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: RetrieveEventByIdInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Retrieves an event from the Google Calendar by its ID.

        Args:
            input_data (RetrieveEventByIdInputSchema): Event identifier.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The event details.

        Raises:
            ToolExecutionException: If event retrieval fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            result = self.client.events().get(calendarId=input_data.calendar_id, eventId=input_data.event_id).execute()

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
