"""Run a Composio action directly and as an agent tool.

The node executes one Composio tool (an action of a toolkit) on behalf of a Composio user.
Unlike the builder, which snapshots the tool's schema when the action is picked, a script has
to supply `input_props` itself - it is the tool's `input_parameters` JSON Schema, available
from `GET /api/v3/tools/{slug}`.
"""

import os

from dynamiq.connections import Composio as ComposioConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools import Composio
from examples.components.tools.extra_utils.utils_llm import setup_llm

# The Composio user the action runs as. On the platform this is always the project id.
USER_ID = os.environ.get("COMPOSIO_USER_ID", "")

# Pins the action to one connected account. Without it Composio resolves an account for USER_ID.
CONNECTED_ACCOUNT_ID = os.environ.get("COMPOSIO_CONNECTED_ACCOUNT_ID") or None

# The tool's `input_parameters` JSON Schema, as returned by GET /api/v3/tools/{slug}.
GMAIL_SEND_EMAIL_PROPS = {
    "type": "object",
    "title": "SendEmailRequest",
    "required": ["recipient_email", "body"],
    "properties": {
        "recipient_email": {"type": "string", "description": "Address of the recipient."},
        "body": {"type": "string", "description": "Body of the email."},
        "subject": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
            "description": "Subject of the email.",
        },
        "is_html": {"type": "boolean", "default": False, "description": "Whether the body is HTML."},
    },
}


def build_tool(**kwargs) -> Composio:
    return Composio(
        connection=ComposioConnection(),
        input_props=GMAIL_SEND_EMAIL_PROPS,
        user_id=USER_ID,
        toolkit_slug="gmail",
        tool_slug="GMAIL_SEND_EMAIL",
        connected_account_id=CONNECTED_ACCOUNT_ID,
        # Pinning the version keeps execution on the schema the arguments were written against.
        tool_version=os.environ.get("COMPOSIO_TOOL_VERSION") or None,
        **kwargs,
    )


def send_email_directly() -> None:
    """Run the action on its own, supplying every argument at call time."""
    result = build_tool().run(
        input_data={
            "recipient_email": "someone@example.com",
            "subject": "Sent from Dynamiq",
            "body": "Hello from the Composio action node.",
        }
    )
    print(result)


def send_email_from_an_agent() -> None:
    """Hand the action to an agent.

    Values in `arguments` are configured up front: they are dropped from the required set, so the
    agent may override them but does not have to supply them. The generated tool description lists
    them, along with the required and optional parameters.
    """
    tool = build_tool(
        # `is_optimized_for_agents` makes the node return the raw response body, which the agent
        # reads directly rather than through a parsed payload.
        is_optimized_for_agents=True,
        arguments={"recipient_email": "someone@example.com"},
    )
    print(f"\nTool description the agent sees:\n{tool.description}\n")

    agent = Agent(name="Mailer", llm=setup_llm(), tools=[tool])
    result = agent.run(input_data={"input": "Send a short note saying the deploy finished."})
    print(result.output)


if __name__ == "__main__":
    if not USER_ID:
        raise SystemExit("Set COMPOSIO_USER_ID (and COMPOSIO_API_KEY) to run this example.")

    send_email_directly()
    send_email_from_an_agent()
