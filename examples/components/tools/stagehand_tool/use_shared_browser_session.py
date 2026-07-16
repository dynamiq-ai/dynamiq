"""Share ONE live browser session between an agent and its subagent.

This example demonstrates ``share_browser_session_with_subagents`` with the owner
CO-DRIVING the browser: the owner agent has its OWN Stagehand tool, navigates the
browser to a URL itself, and THEN delegates to a "reader" subagent to report the
current page. Both drive the SAME live Browserbase session (one logged-in page
carried across the handoff), one agent at a time.

The browser lease is reentrant down the ancestor chain, so the owner may hold the
lease (from its own navigation) AND await the subagent without deadlock — the
subagent borrows the lease its blocked ancestor holds. Genuinely-parallel
subagents would still serialize (one driver at a time).

Because the reader subagent is initialized (``SubAgentTool(agent=...)``), we keep
references to both Stagehand tools and print their ``_session_id`` values after
the run — they are identical, proving the session was shared. We also print the
surfaced ``live_view_url`` and the reader's final content.

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
    """Wire an owner that co-drives the browser and delegates to a reader subagent.

    Returns the owner plus the owner's and the subagent's Stagehand tool references
    so the caller can inspect their session ids after the run.
    """
    owner_tool = _stagehand_tool("owner-browser")
    reader_tool = _stagehand_tool("reader-browser")

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
            "You share ONE live browser session with your sub-agent, and you have your OWN Stagehand "
            "browser tool. Always invoke a tool as {'input': '<task>'}. FIRST use your OWN Stagehand "
            "tool to navigate the browser to the exact URL you are given. THEN delegate to the "
            "'reader' sub-agent to report the current page's URL and title. Then give a short answer."
        ),
        description="Owner agent that drives the browser itself and also delegates on one session.",
        llm=_llm(),
        tools=[
            owner_tool,
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
    return owner, owner_tool, reader_tool


def main():
    missing = [
        name for name in ("OPENAI_API_KEY", "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID") if not os.getenv(name)
    ]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}. Set them and re-run.")
        return

    owner, owner_tool, reader_tool = build_owner()

    with get_connection_manager() as cm:
        wf = Workflow(flow=Flow(connection_manager=cm, init_components=True, nodes=[owner]))
        try:
            result = wf.run(
                input_data={
                    "input": (
                        f"First navigate the browser to {TARGET_URL} using your own Stagehand tool. After it "
                        "loads, delegate to the reader sub-agent to report the current page's URL and title."
                    )
                }
            )

            owner_output = result.output.get("owner", {}).get("output", {})
            print("Status:", result.status)
            print("Owner tool session_id: ", owner_tool._session_id)
            print("Reader tool session_id:", reader_tool._session_id)
            print(
                "Shared ONE session:",
                owner_tool._session_id is not None and owner_tool._session_id == reader_tool._session_id,
            )
            print("Surfaced live_view_url:", owner_output.get("live_view_url"))
            print("Reader final content:", owner_output.get("content"))
        finally:
            for tool in (owner_tool, reader_tool):
                try:
                    tool.close()
                except Exception as exc:  # defensive: don't let cleanup mask the run outcome
                    print(f"Warning: closing {tool.name} failed: {exc}")


if __name__ == "__main__":
    main()
