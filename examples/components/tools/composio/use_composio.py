import json
import os
from typing import Iterable

from dynamiq.memory import Memory
from dynamiq.memory.backends import InMemory
from dynamiq.nodes.agents import Agent

from dynamiq.connections import Composio as ComposioConnection
from dynamiq.nodes.tools import Composio
from examples.components.tools.extra_utils import setup_llm


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} must be set for this example.")
    return value


def build_connection() -> ComposioConnection:
    api_key = require_env("COMPOSIO_API_KEY")
    kwargs: dict[str, str] = {"api_key": api_key}

    if connected_account_id := os.environ.get("COMPOSIO_CONNECTED_ACCOUNT_ID"):
        kwargs["connected_account_id"] = connected_account_id
    if entity_id := os.environ.get("COMPOSIO_ENTITY_ID"):
        kwargs["entity_id"] = entity_id

    return ComposioConnection(**kwargs)


def show_session_info(connection: ComposioConnection) -> None:
    session = None
    try:
        session = connection.connect()
        response = session.get(
            f"{connection.url.rstrip('/')}/auth/session/info",
            timeout=connection.timeout,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"Failed to fetch Composio session info: {exc}")
        return
    finally:
        if session is not None:
            session.close()

    print("\n# Get current user session information")
    print(json.dumps(data, indent=2))


def list_linear_tools(connection: ComposioConnection) -> list[str]:
    session = None
    try:
        session = connection.connect()
        response = session.get(
            f"{connection.url.rstrip('/')}/tools/enum",
            timeout=connection.timeout,
        )
        response.raise_for_status()
        slugs = response.json()
    except Exception as exc:
        print(f"Failed to list Composio tools: {exc}")
        return []
    finally:
        if session is not None:
            session.close()

    linear_slugs = sorted({slug for slug in slugs if slug.startswith("LINEAR_")})
    if linear_slugs:
        print("\nAvailable Linear tools:")
        for slug in linear_slugs:
            print(f" - {slug}")
    return linear_slugs


def build_tools(connection: ComposioConnection, wanted: Iterable[str]) -> list[Composio]:
    tools: list[Composio] = []
    for slug in wanted:
        tool = Composio(tool_slug=slug, connection=connection)
        tools.append(tool)
        print(f"\nLoaded tool '{tool.name}':\n{tool.description}")
    return tools


def main() -> None:
    connection = build_connection()
    show_session_info(connection)

    if not connection.connected_account_id:
        print("\nSet COMPOSIO_CONNECTED_ACCOUNT_ID to execute Linear project tools.")
        return

    available_slugs = list_linear_tools(connection)

    required_slugs = (
        "LINEAR_LIST_LINEAR_PROJECTS",
        "LINEAR_LIST_LINEAR_TEAMS",
        "LINEAR_CREATE_LINEAR_ISSUE",
    )

    missing = [slug for slug in required_slugs if slug not in available_slugs]
    if missing:
        print("\nMissing required Linear tools:")
        for slug in missing:
            print(f" - {slug}")
        return

    tools = build_tools(connection, required_slugs)

    llm = setup_llm(model_provider="claude", model_name="claude-3-5-sonnet-20240620")
    memory = Memory(backend=InMemory())

    agent = Agent(
        name="Linear Helper Agent",
        llm=llm,
        tools=tools,
        memory=memory,
    )

    result = agent.run(
        input_data={
            "input": (
                "Show me the project and team list. "
                "Create one task with a simple description in any available project."
            )
        },
        config=None,
    )

    print("\n# Agent Output")
    print(result.output)


if __name__ == "__main__":
    main()
