import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_calendar.google_calendar_base import GoogleCalendarBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_CREATE_EVENT = """## Create Calendar Event Tool

Creates a new event in a specified Google Calendar.

### Parameters
- `calendar_id` (str, optional, default: "primary"): The ID of the calendar where the event will be created.
- `title` (str): The title or summary of the event.
- `start` (str): Start time in ISO 8601 format or date in YYYY-MM-DD format for all-day events,
(e.g., `2025-05-28T10:00:00-07:00`, `2025-05-28`) .
- `end` (str): End time in ISO 8601 format or date in YYYY-MM-DD format for all-day events.
- `description` (str, optional): A description or details of the event.
- `location` (str, optional): The location where the event takes place.
- `attendees` (list[str], optional): A list of email addresses of event attendees.
"""


class CreateEventInputSchema(BaseModel):
    calendar_id: str = Field(default="primary", description="Calendar ID (default: 'primary').")
    title: str = Field(..., description="Title of the event.")
    start: str = Field(..., description="Start time (ISO 8601 format) or date (YYYY-MM-DD) for all-day events.")
    end: str = Field(..., description="End time (ISO 8601 format) or date (YYYY-MM-DD) for all-day events.")
    description: str | None = Field(None, description="Description of the event.")
    location: str | None = Field(default=None, description="Location of the event.")
    attendees: list[str] | None = Field(default=None, description="List of attendee email addresses.")


class CreateEvent(GoogleCalendarBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "create-event"
    description: str = DESCRIPTION_CREATE_EVENT
    input_schema: ClassVar[type[BaseModel]] = CreateEventInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    @staticmethod
    def _is_date_only(value: str) -> bool:
        """Check if the string is in date-only format (YYYY-MM-DD)."""
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))

    def execute(
        self, input_data: CreateEventInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Creates an event in the Google Calendar.

        Args:
            input_data (CreateEventInputSchema): Event details.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: The created event details.

        Raises:
            ToolExecutionException: If event creation fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            event_body = {
                "summary": input_data.title,
                "description": input_data.description,
            }

            if self._is_date_only(input_data.start) and self._is_date_only(input_data.end):
                event_body["start"] = {"date": input_data.start}
                event_body["end"] = {"date": input_data.end}
            else:
                event_body["start"] = {"dateTime": input_data.start}
                event_body["end"] = {"dateTime": input_data.end}

            if input_data.location:
                event_body["location"] = input_data.location

            if input_data.attendees:
                event_body["attendees"] = [{"email": email} for email in input_data.attendees]

            result = self.client.events().insert(calendarId=input_data.calendar_id, body=event_body).execute()

            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
