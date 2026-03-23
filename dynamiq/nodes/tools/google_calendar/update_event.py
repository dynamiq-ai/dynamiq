from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_calendar.google_calendar_base import GoogleCalendarBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_UPDATE_EVENT = """## Update Calendar Event Tool

Updates details of an existing event in Google Calendar.

### Parameters
- `calendar_id` (str, optional, default: "primary"): The ID of the calendar containing the event.
- `event_id` (str): The unique identifier of the event to update.
- `summary` (str, optional): The new title or summary of the event.
- `description` (str, optional): The new description or details of the event.
- `start` (str, optional): The new start time in ISO 8601 format (e.g., `2025-05-28T10:00:00-07:00`).
- `end` (str, optional): The new end time in ISO 8601 format.
- `location` (str, optional): The new location of the event.
- `attendees` (list[str], optional): The new list of attendee email addresses.
"""


class UpdateEventInputSchema(BaseModel):
    calendar_id: str = Field(default="primary", description="Calendar ID (default: 'primary').")
    event_id: str = Field(..., description="ID of the event to update.")
    summary: str | None = Field(default=None, description="New title of the event.")
    description: str | None = Field(default=None, description="New description.")
    start: str | None = Field(default=None, description="New start time (ISO 8601).")
    end: str | None = Field(default=None, description="New end time (ISO 8601).")
    location: str | None = Field(default=None, description="New location.")
    attendees: list[str] | None = Field(default=None, description="New list of attendees.")


class UpdateEvent(GoogleCalendarBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "update-event"
    description: str = DESCRIPTION_UPDATE_EVENT
    input_schema: ClassVar[type[BaseModel]] = UpdateEventInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: UpdateEventInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Updates an existing event in the Google Calendar.

        Args:
            input_data (UpdateEventInputSchema): Event details to update.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The updated event details.

        Raises:
            ToolExecutionException: If event update fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            event = self.client.events().get(calendarId=input_data.calendar_id, eventId=input_data.event_id).execute()

            if input_data.summary:
                event["summary"] = input_data.summary
            if input_data.description:
                event["description"] = input_data.description
            if input_data.start:
                event["start"] = (
                    {"dateTime": input_data.start} if "T" in input_data.start else {"date": input_data.start}
                )
            if input_data.end:
                event["end"] = {"dateTime": input_data.end} if "T" in input_data.end else {"date": input_data.end}
            if input_data.location:
                event["location"] = input_data.location
            if input_data.attendees:
                event["attendees"] = [{"email": email} for email in input_data.attendees]

            result = (
                self.client.events()
                .update(calendarId=input_data.calendar_id, eventId=input_data.event_id, body=event)
                .execute()
            )

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
