"""Share ONE live browser session across an agent and its subagents.

This example demonstrates ``share_browser_session_with_subagents``: an owner
agent orchestrates two initialized subagents — a "navigator" that opens a URL and
a "reader" that reports the current page — and both drive the SAME live
Browserbase session (one logged-in page carried across the handoff), one agent at
a time via an exclusive per-agent-run lease.

The owner holds NO browser tool of its own (it only delegates), which respects
the Model A deadlock invariant. Because the subagents are initialized
(``SubAgentTool(agent=...)``), we keep references to each one's Stagehand tool and
print both ``_session_id`` values after the run — they are identical, proving the
session was shared. We also print the surfaced ``live_view_url`` and the reader's
final content.

Requires OPENAI_API_KEY, BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID.
"""

import os

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Stagehand as StagehandConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import Stagehand
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.types import InferenceMode

TARGET_URL = "https://example.com/"


def _llm() -> OpenAI:
    return OpenAI(
        connection=OpenAIConnection(),
        model="gpt-4o",
        temperature=0.0,
        max_tokens=800,
        is_postponed_component_init=True,
    )


def _stagehand_tool(name: str) -> Stagehand:
    return Stagehand(
        name=name,
        connection=StagehandConnection(model_api_key=os.getenv("OPENAI_API_KEY")),
        model_name="gpt-4o",
        is_return_live_view_url_enabled=True,
        is_postponed_component_init=True,
    )


def build_owner():
    """Wire an owner + navigator/reader subagents that share one browser session.

    Returns the owner plus the two Stagehand tool references so the caller can
    inspect their session ids after the run.
    """
    navigator_tool = _stagehand_tool("navigator-browser")
    reader_tool = _stagehand_tool("reader-browser")

    navigator = Agent(
        name="navigator",
        role=(
            "Your ONLY job is to use your Stagehand browser tool to navigate the browser to the "
            "EXACT URL you are given, then confirm the navigation succeeded. Do nothing else."
        ),
        description="Navigates the shared browser to a given URL using its Stagehand tool.",
        llm=_llm(),
        tools=[navigator_tool],
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        max_loops=4,
        is_postponed_component_init=True,
    )
    reader = Agent(
        name="reader",
        role=(
            "Your ONLY job is to use your Stagehand browser tool to extract and report the CURRENT "
            "page's URL and title. Do NOT navigate anywhere; just read the page you are already on."
        ),
        description="Reports the current page's URL/title using its Stagehand tool.",
        llm=_llm(),
        tools=[reader_tool],
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        max_loops=4,
        is_postponed_component_init=True,
    )

    owner = Agent(
        id="owner",
        name="owner",
        role=(
            "You coordinate two sub-agents that share ONE live browser session. You have NO browser "
            "tool of your own — you only delegate. Always invoke a tool as {'input': '<task>'}. "
            "First delegate to the 'navigator' sub-agent to open the exact URL you are given. "
            "AFTER navigator confirms, delegate to the 'reader' sub-agent to report the current "
            "page's URL and title. Then give a short final answer."
        ),
        description="Owner agent that shares one browser session across its subagents.",
        llm=_llm(),
        tools=[
            SubAgentTool(
                name="navigator",
                description="Delegate a browser navigation task. Pass {'input': '<task>'}.",
                agent=navigator,
            ),
            SubAgentTool(
                name="reader",
                description="Delegate reading the current page. Pass {'input': '<task>'}.",
                agent=reader,
            ),
        ],
        share_browser_session_with_subagents=True,
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        max_loops=8,
        is_postponed_component_init=True,
    )
    return owner, navigator_tool, reader_tool


def main():
    missing = [
        name for name in ("OPENAI_API_KEY", "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID") if not os.getenv(name)
    ]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}. Set them and re-run.")
        return

    owner, navigator_tool, reader_tool = build_owner()

    with get_connection_manager() as cm:
        wf = Workflow(flow=Flow(connection_manager=cm, init_components=True, nodes=[owner]))
        try:
            result = wf.run(
                input_data={
                    "input": (
                        f"First have the navigator open the URL {TARGET_URL}. After it confirms, have the "
                        "reader report the current page's URL and title."
                    )
                }
            )

            owner_output = result.output.get("owner", {}).get("output", {})
            print("Status:", result.status)
            print("Navigator tool session_id:", navigator_tool._session_id)
            print("Reader tool session_id:   ", reader_tool._session_id)
            print(
                "Shared ONE session:",
                navigator_tool._session_id is not None and navigator_tool._session_id == reader_tool._session_id,
            )
            print("Surfaced live_view_url:", owner_output.get("live_view_url"))
            print("Reader final content:", owner_output.get("content"))
        finally:
            for tool in (navigator_tool, reader_tool):
                try:
                    tool.close()
                except Exception as exc:  # defensive: don't let cleanup mask the run outcome
                    print(f"Warning: closing {tool.name} failed: {exc}")


if __name__ == "__main__":
    main()
