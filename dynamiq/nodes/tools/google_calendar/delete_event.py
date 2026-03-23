from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_calendar.google_calendar_base import GoogleCalendarBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_DELETE_EVENT = """## Delete Calendar Event Tool

Deletes a specific event from Google Calendar using its event ID.

### Parameters
- `calendar_id` (str, optional, default: "primary"): The ID of the calendar where the event will be created.
- `event_id` (str): The unique identifier of the event to delete.
"""


class DeleteEventInputSchema(BaseModel):
    calendar_id: str = Field(default="primary", description="Calendar ID (default: 'primary').")
    event_id: str = Field(..., description="ID of the event to delete.")


class DeleteEvent(GoogleCalendarBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "delete-event"
    description: str = DESCRIPTION_DELETE_EVENT
    input_schema: ClassVar[type[BaseModel]] = DeleteEventInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: DeleteEventInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Deletes an event from the Google Calendar.

        Args:
            input_data (DeleteEventInputSchema): Event identifier.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: Confirmation of the deletion result.

        Raises:
            ToolExecutionException: If event deletion fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            self.client.events().delete(calendarId=input_data.calendar_id, eventId=input_data.event_id).execute()

            result = f"Event {input_data.event_id} deleted successfully."
            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
