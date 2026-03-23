from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.google_calendar.google_calendar_base import GoogleCalendarBase
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class EventType(str, Enum):
    birthday = "birthday"
    default = "default"
    focusTime = "focusTime"
    fromGmail = "fromGmail"
    outOfOffice = "outOfOffice"
    workingLocation = "workingLocation"


class EventsOrderBy(str, Enum):
    startTime = "startTime"
    updated = "updated"


DESCRIPTION_SEARCH_EVENTS = """## Search Calendar Events Tool
Searches for events in a Google Calendar based on provided filters.

### Parameters
- `calendar_id` (str, optional, default: "primary"): The ID of the calendar to search.
- `query` (str, optional): Free-text search query for matching event fields like summary or description.
- `time_min` (str, optional): Lower bound (inclusive) for event start time in RFC3339 format.
- `time_max` (str, optional): Upper bound (exclusive) for event start time in RFC3339 format.
- `max_results` (int, optional): Maximum number of events to return per page.
- `order_by` (str, optional): Specifies the ordering of events in the result.
  - Possible values:
    - `"startTime"`: Order by event start time (requires `single_events=True`).
    - `"updated"`: Order by last modification time.
- `event_types` (list[str], optional): Filter events by type. Possible values include:
  - `"birthday"`
  - `"default"`
  - `"focusTime"`
  - `"fromGmail"`
  - `"outOfOffice"`
  - `"workingLocation"`
- `max_attendees` (int, optional): Maximum number of attendees to include in the response.
- `private_extended_property` (list[str], optional): Filters by private extended properties, in `key=value` format.
- `shared_extended_property` (list[str], optional): Filters by shared extended properties, in `key=value` format.
- `show_deleted` (bool, optional, default: False): Whether to include deleted events in the results.
- `show_hidden_invitations` (bool, optional, default: False): Whether to include hidden invitations in the results.
- `single_events` (bool, optional, default: False): Whether to expand recurring events into instances.
- `time_zone` (str, optional): The time zone for the response data.
- `updated_min` (str, optional): Lower bound for the last modification time of events (RFC3339 format).

### Notes
- If `order_by` is set to `"startTime"`, `single_events` must be `True`.
"""


class SearchEventsInputSchema(BaseModel):
    calendar_id: str = Field(default="primary", description="Calendar ID (default: 'primary').")
    query: str | None = Field(default=None, description="Free text search query.")
    time_min: str | None = Field(default=None, description="Lower bound for event end time (RFC3339).")
    time_max: str | None = Field(default=None, description="Upper bound for event start time (RFC3339).")
    max_results: int | None = Field(default=None, description="Maximum number of events to return.")
    order_by: EventsOrderBy | None = Field(default=None, description="Order of events.")
    event_types: list[EventType] | None = Field(default=None, description="List of event types to filter.")
    max_attendees: int | None = Field(default=None, description="Max attendees to include.")
    private_extended_property: list[str] | None = Field(
        default=None, description="Filter by private extended property."
    )
    shared_extended_property: list[str] | None = Field(default=None, description="Filter by shared extended property.")
    show_deleted: bool | None = Field(default=False, description="Include deleted events.")
    show_hidden_invitations: bool | None = Field(default=False, description="Include hidden invitations.")
    single_events: bool | None = Field(default=True, description="Expand recurring events.")
    time_zone: str | None = Field(default=None, description="Timezone for response.")
    updated_min: str | None = Field(default=None, description="Lower bound for last modification time (RFC3339).")


class SearchEvents(GoogleCalendarBase):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "search-events"
    description: str = DESCRIPTION_SEARCH_EVENTS
    input_schema: ClassVar[type[BaseModel]] = SearchEventsInputSchema
    connection: GoogleOAuth2 = GoogleOAuth2()

    def execute(
        self, input_data: SearchEventsInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Searches for events in a specified Google Calendar with optional filters.

        Args:
            input_data (SearchEventsInputSchema): Search filters and calendar details.
            config (RunnableConfig | None): Optional runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            dict[str, Any]: A list of matching events.

        Raises:
            ToolExecutionException: If the search fails or invalid parameters are provided.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            if input_data.order_by == EventsOrderBy.startTime and not input_data.single_events:
                raise ToolExecutionException(
                    "Google Calendar API requires 'singleEvents=True' when using 'orderBy=startTime'.",
                    recoverable=True,
                )

            params = {
                "calendarId": input_data.calendar_id,
                "q": input_data.query,
                "timeMin": input_data.time_min,
                "timeMax": input_data.time_max,
                "maxResults": input_data.max_results,
                "orderBy": input_data.order_by.value if input_data.order_by else None,
                "maxAttendees": input_data.max_attendees,
                "showDeleted": input_data.show_deleted,
                "showHiddenInvitations": input_data.show_hidden_invitations,
                "singleEvents": input_data.single_events,
                "timeZone": input_data.time_zone,
                "updatedMin": input_data.updated_min,
            }

            if input_data.event_types:
                params["eventTypes"] = [et.value for et in input_data.event_types]

            if input_data.private_extended_property:
                params["privateExtendedProperty"] = input_data.private_extended_property

            if input_data.shared_extended_property:
                params["sharedExtendedProperty"] = input_data.shared_extended_property

            params = {k: v for k, v in params.items() if v is not None}
            events_result = self.client.events().list(**params).execute()

            result = events_result.get("items", [])
            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
            return {"content": result}

        except Exception as e:
            raise ToolExecutionException(str(e), recoverable=True)

    def close(self):
        self.client.close()
