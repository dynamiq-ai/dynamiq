import os

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes.tools.google_calendar import (
    CreateEvent,
    DeleteEvent,
    RetrieveEventById,
    SearchEvents,
    UpdateEvent,
)
from examples.components.tools.extra_utils import setup_llm

AGENT_ROLE = (
    "An AI-powered productivity assistant for managing Gmail: "
    "creating drafts, searching messages, and modifying labels to streamline workflows."
)


def create_gmail_connection() -> GoogleOAuth2:
    """Initialize a Google Calendar API connection with specified credential files."""
    return GoogleOAuth2(
        access_token=os.getenv("GOOGLE_OAUTH2_ACCESS_TOKEN"),
    )


if __name__ == "__main__":
    with get_connection_manager() as cm:
        llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)

        agent = Agent(
            name="CalendarManagerAgent",
            id="calendar-manager",
            llm=llm,
            tools=[
                CreateEvent(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
                DeleteEvent(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
                RetrieveEventById(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
                SearchEvents(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
                UpdateEvent(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
            ],
            role=AGENT_ROLE,
            inference_mode=InferenceMode.XML,
        )

        wf = Workflow(
            flow=Flow(
                connection_manager=cm,
                init_components=True,
                nodes=[agent],
            )
        )

        task = """
        Your task is to manage a user's calendar by completing the following steps sequentially:

        1. Search for any events with the word "Project" in the title that are scheduled within the next 7 days.
           If no events are found, note this in your response.
        2. If there is at least one event found, get the details of the first event by its ID.
        3. Update the title of this event to add "(Updated)" at the end.
        4. After updating, confirm that the event was successfully modified.
        5. Finally, delete the same event and confirm the deletion.
        """

        result = wf.run(input_data={"input": task})

    print(result.output)
