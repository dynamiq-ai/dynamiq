"""Live shared-browser test: an owner agent and its two subagents share ONE
live Browserbase session across a delegated handoff (requires OPENAI_API_KEY,
BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID).

The owner orchestrates only (it holds NO Stagehand tool of its own, respecting
the Model A deadlock invariant): it first delegates to a "navigator" subagent to
open a URL, then to a "reader" subagent to report the CURRENT page. Both
subagents are INITIALIZED subagents (``SubAgentTool(agent=...)``) so the test can
hold references to each one's Stagehand tool instance.

Sharing-assertion mechanism: TOOL REFERENCES. Initialized subagents run in place
(``_run_tool`` resolves ``tool.get_or_create_agent()`` and only clones for
factory-mode / parallel non-child tools), and with
``parallel_tool_calls_enabled=False`` the Stagehand tool is not cloned either, so
the held ``navigator_tool`` / ``reader_tool`` references reflect the real
``_session_id`` set at action time. If the two distinct tool instances end up on
the SAME ``_session_id``, they drove one shared+persisted live browser.

Expensive and LLM-nondeterministic; skipped without all three credentials.
"""

import os

import pytest

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
from dynamiq.runnables import RunnableConfig, RunnableStatus

TARGET_URL = "https://example.com/"


def _openai_llm() -> OpenAI:
    return OpenAI(
        connection=OpenAIConnection(api_key=os.getenv("OPENAI_API_KEY")),
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


@pytest.mark.integration
def test_two_subagents_share_one_browserbase_session():
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("BROWSERBASE_API_KEY") and os.getenv("BROWSERBASE_PROJECT_ID")):
        pytest.skip("OPENAI_API_KEY, BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID are required for this test.")

    # Two Stagehand tools we keep references to — one per subagent.
    navigator_tool = _stagehand_tool("navigator-browser")
    reader_tool = _stagehand_tool("reader-browser")

    navigator = Agent(
        name="navigator",
        role=(
            "Your ONLY job is to use your Stagehand browser tool to navigate the browser to the "
            "EXACT URL you are given, then confirm the navigation succeeded. Do nothing else."
        ),
        description="Navigates the shared browser to a given URL using its Stagehand tool.",
        llm=_openai_llm(),
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
        llm=_openai_llm(),
        tools=[reader_tool],
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        max_loops=4,
        is_postponed_component_init=True,
    )

    navigator_subagent = SubAgentTool(
        name="navigator",
        description="Delegate a browser navigation task. Pass {'input': '<task>'}.",
        agent=navigator,
    )
    reader_subagent = SubAgentTool(
        name="reader",
        description="Delegate reading the current page. Pass {'input': '<task>'}.",
        agent=reader,
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
        llm=_openai_llm(),
        tools=[navigator_subagent, reader_subagent],
        share_browser_session_with_subagents=True,
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        max_loops=8,
        is_postponed_component_init=True,
    )

    with get_connection_manager() as cm:
        wf = Workflow(flow=Flow(connection_manager=cm, init_components=True, nodes=[owner]))
        try:
            result = wf.run(
                input_data={
                    "input": (
                        f"First have the navigator open the URL {TARGET_URL}. After it confirms, have the "
                        "reader report the current page's URL and title."
                    )
                },
                config=RunnableConfig(),
            )

            assert result.status == RunnableStatus.SUCCESS

            # HARD proof of a shared+persisted session: two DISTINCT Stagehand tool instances
            # ended up driving the SAME Browserbase session_id (carried across the handoff).
            assert navigator_tool._session_id is not None, "navigator tool never opened a session"
            assert reader_tool._session_id is not None, "reader tool never opened a session"
            assert navigator_tool._session_id == reader_tool._session_id, (
                "subagents did not share ONE browser session: "
                f"{navigator_tool._session_id!r} != {reader_tool._session_id!r}"
            )

            # The owner surfaces the shared browser's live-view URL on its run result.
            owner_output = result.output[owner.id]["output"]
            assert owner_output.get("live_view_url"), "owner run result did not surface a shared live_view_url"

            # Secondary, best-effort (LLM phrasing varies): the reader's content should reflect
            # the navigated page — log rather than hard-fail on continuity.
            reader_content = str(owner_output.get("content") or "")
            if "example" not in reader_content.lower():
                print(
                    "NOTE: reader content did not obviously mention the navigated page; "
                    f"content was: {reader_content[:200]!r}"
                )
        finally:
            # Owner closes the shared session at run end; close defensively in case the run raised
            # before teardown, so the live Browserbase session does not leak.
            for tool in (navigator_tool, reader_tool):
                try:
                    tool.close()
                except Exception:
                    pass
