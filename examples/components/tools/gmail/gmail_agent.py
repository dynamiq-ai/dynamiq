import os

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes.tools.gmail import (
    CreateDraft,
    ModifyEmailLabels,
    ReplyToEmail,
    RetrieveEmailsById,
    SearchEmails,
    SendDraft,
    SendEmail,
)
from examples.components.tools.extra_utils import setup_llm

AGENT_ROLE = (
    "An AI-powered productivity assistant for managing Gmail: "
    "creating drafts, searching messages, and modifying labels to streamline workflows."
)


def create_gmail_connection() -> GoogleOAuth2:
    """Initialize a Gmail API connection with specified credential files."""
    return GoogleOAuth2(
        access_token=os.getenv("GOOGLE_OAUTH2_ACCESS_TOKEN"),
    )


if __name__ == "__main__":
    with get_connection_manager() as cm:
        llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)

        agent = Agent(
            name="EmailManagerAgent",
            id="email-manager",
            llm=llm,
            tools=[
                CreateDraft(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
                ModifyEmailLabels(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
                SendDraft(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
                SendEmail(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
                ReplyToEmail(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
                SearchEmails(connection=create_gmail_connection(), is_postponed_component_init=True),
                RetrieveEmailsById(
                    connection=create_gmail_connection(),
                    is_postponed_component_init=True,
                ),
            ],
            role=AGENT_ROLE,
            inference_mode=InferenceMode.XML,
            is_postponed_component_init=True,
        )

        wf = Workflow(
            flow=Flow(
                connection_manager=cm,
                init_components=True,
                nodes=[agent],
            )
        )

        task = """
        Draft a reply to the most recent unread email from example@gmail.com, providing a follow-up.
        Label the email as 'Important', and send it once the draft is ready.
        """

        result = wf.run(input_data={"input": task})

    print(result.output)
