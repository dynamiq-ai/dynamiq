import os

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes.tools.google_calendar import (
    CreateEvent,
    DeleteEvent,
    ListCalendars,
    RetrieveEventById,
    SearchEvents,
    UpdateEvent,
)


def get_connection():
    """Initialize a Google Calendar API connection with specified credential files."""
    return GoogleOAuth2(
        access_token=os.getenv("GOOGLE_OAUTH2_ACCESS_TOKEN"),
    )


def create_event():
    """Create a Google Calendar event."""
    tool = CreateEvent(id="create-calendar-event", connection=get_connection())

    result = tool.run(
        input_data={
            "title": "Team Meeting",
            "description": "Discuss Q2 Roadmap and goals.",
            "start": "2025-06-01T10:00:00-07:00",
            "end": "2025-06-01T11:00:00-07:00",
            "location": "123 Main St, Conference Room A",
            "attendees": ["example@gmail.com"],
        }
    )
    tool.close()
    print("Created Calendar Event:", result.output)


def delete_event():
    """Delete an event from Google Calendar."""
    tool = DeleteEvent(id="delete-calendar-event", connection=get_connection())

    result = tool.run(input_data={"event_id": "your_event_id"})
    tool.close()
    print("Delete Event Result:", result.output)


def get_event_by_id():
    """Get a specific event by ID from Google Calendar."""
    tool = RetrieveEventById(id="get-calendar-event", connection=get_connection())

    result = tool.run(input_data={"event_id": "your_event_id"})
    tool.close()
    print("Retrieved Calendar Event:", result.output)


def search_events():
    """Search for events in Google Calendar."""
    tool = SearchEvents(id="search-calendar-events", connection=get_connection())

    result = tool.run(
        input_data={
            "calendar_id": "primary",
            "query": "Standup Meeting",
            "time_min": "2025-05-01T00:00:00Z",
            "time_max": "2025-06-30T23:59:59Z",
            "maxAttendees": 10,
            "max_results": 10,
            "show_deleted": True,
        }
    )
    tool.close()
    print("Search Calendar Events:", result.output)


def update_event():
    """Update an event in Google Calendar."""
    tool = UpdateEvent(id="update-calendar-event", connection=get_connection())
    result = tool.run(
        input_data={
            "event_id": "your_event_id",
            "summary": "Updated Team Meeting",
            "description": "Updated description for the event.",
            "location": "456 New Location, Conference Room B",
            "attendees": ["new.attendee@example.com"],
        }
    )
    tool.close()
    print("Updated Calendar Event:", result.output)


def list_calendars():
    """List all calendars in Google Calendar."""
    tool = ListCalendars(id="list-calendars", connection=get_connection())
    result = tool.run(input_data={})
    tool.close()
    print("List Calendars:", result.output)


if __name__ == "__main__":
    create_event()
    delete_event()
    get_event_by_id()
    search_events()
    update_event()
    list_calendars()
