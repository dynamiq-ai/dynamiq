"""Share ONE live browser between an agent and its subagents.

This example demonstrates ``share_browser_session_with_subagents``. Every agent in
the run drives ONE live Browserbase session, so cookies and logins are visible to
each other immediately — nothing has to close for state to cross.

What that means in practice:

- State crosses **instantly**, including the current page: a subagent picks up
  exactly where the previous one left off, already logged in.
- Only one agent may drive the page at a time (they share it), so agents take turns.
  An agent that browses and then delegates hands the page over for the duration of
  the call and takes it back afterwards — nothing closes, nothing is lost.
- The one shape that does NOT work is using the browser and delegating browser work
  in the same *parallel* batch: both would drive the same page at once. That fails
  with an explicit error rather than hanging.
- The session is ended once, at the owner's teardown, which is also what persists its
  Browserbase Context. That Context is how state reaches a LATER run (say, the next
  turn of a user's conversation) — a separate axis from agent-to-agent sharing.

The owner below is a pure coordinator with no browser tool of its own. That is the
shape to reach for by default: it keeps the handoffs obvious and sidesteps the
parallel case entirely.

The run has a "setter" subagent log a cookie, then a separate "checker" subagent read
it back. We print the session ids afterwards: all agents on ONE session.

Requires OPENAI_API_KEY, BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID.
"""

import io
import os
from collections import defaultdict

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

COOKIE_SET_URL = "https://httpbin.org/cookies/set/dynamiq_demo/shared-ok"
COOKIE_READ_URL = "https://httpbin.org/cookies"

# tool name -> [{"session_id": ..., "context_id": ...}] for every session that tool opened
SESSION_LOG: dict[str, list[dict]] = defaultdict(list)


class RecordingStagehand(Stagehand):
    """Stagehand that logs the session id it drove and the Context that session carries.

    Only needed to *show* what happened: the tool clears ``_session_id`` when it closes, so it
    cannot be read afterwards.
    """

    async def _init_client(
        self,
        files: list[io.BytesIO],
        shared_context_id: str | None = None,
        create_shared_session: bool = False,
    ):
        await super()._init_client(files, shared_context_id, create_shared_session)
        record = {"session_id": self._session_id, "context_id": shared_context_id}
        if record not in SESSION_LOG[self.name]:
            SESSION_LOG[self.name].append(record)


def _llm() -> OpenAI:
    return OpenAI(
        connection=OpenAIConnection(),
        model="gpt-4o",
        temperature=0.0,
        max_tokens=800,
        is_postponed_component_init=True,
    )


def _stagehand_tool(name: str) -> RecordingStagehand:
    return RecordingStagehand(
        name=name,
        connection=StagehandConnection(model_api_key=os.getenv("OPENAI_API_KEY")),
        model_name="gpt-4o",
        is_return_live_view_url_enabled=True,
        is_postponed_component_init=True,
    )


def _browsing_agent(name: str, role: str, tool: Stagehand) -> Agent:
    return Agent(
        name=name,
        role=role,
        description=f"{name}: browses using its own Stagehand tool.",
        llm=_llm(),
        tools=[tool],
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        max_loops=4,
        is_postponed_component_init=True,
    )


def build_owner():
    """Wire a coordinator owner over two subagents that browse in turn.

    Returns the owner plus both Stagehand tool references so the caller can close them.
    """
    setter_tool = _stagehand_tool("setter-browser")
    checker_tool = _stagehand_tool("checker-browser")

    setter = _browsing_agent(
        "setter",
        (
            "Your ONLY job is to use your Stagehand browser tool to navigate to the EXACT URL you "
            "are given, then confirm the page loaded. Do nothing else."
        ),
        setter_tool,
    )
    checker = _browsing_agent(
        "checker",
        (
            "Your ONLY job is to use your Stagehand browser tool to navigate to the EXACT URL you "
            "are given and report the page's text content verbatim. Do not navigate anywhere else."
        ),
        checker_tool,
    )

    owner = Agent(
        id="owner",
        name="owner",
        role=(
            "You coordinate two sub-agents that share ONE browser profile, and only one of them may "
            "use the browser at a time. You have NO browser tool of your own — you only delegate. "
            "Always invoke a tool as {'input': '<task>'}. First delegate to 'setter'; ONLY AFTER it "
            "answers, delegate to 'checker'. Then report what the checker saw."
        ),
        description="Owner agent that shares one browser profile across its subagents.",
        llm=_llm(),
        tools=[
            SubAgentTool(
                name="setter",
                description="Delegate opening a URL. Pass {'input': '<task>'}.",
                agent=setter,
            ),
            SubAgentTool(
                name="checker",
                description="Delegate reading a page. Pass {'input': '<task>'}.",
                agent=checker,
            ),
        ],
        share_browser_session_with_subagents=True,
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        max_loops=8,
        is_postponed_component_init=True,
    )
    return owner, setter_tool, checker_tool


def _print_sharing_report():
    records = [(name, r) for name, rs in SESSION_LOG.items() for r in rs]
    for name, record in records:
        print(f"  {name}: session={record['session_id']} context={record['context_id']}")

    session_ids = {r["session_id"] for _, r in records}
    context_ids = {r["context_id"] for _, r in records if r["context_id"]}
    print("Shared ONE live session:", len(session_ids) == 1 and all(session_ids))
    print("Session carries a Context (cross-run persistence):", bool(context_ids))


def main():
    missing = [
        name for name in ("OPENAI_API_KEY", "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID") if not os.getenv(name)
    ]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}. Set them and re-run.")
        return

    owner, setter_tool, checker_tool = build_owner()

    with get_connection_manager() as cm:
        wf = Workflow(flow=Flow(connection_manager=cm, init_components=True, nodes=[owner]))
        try:
            result = wf.run(
                input_data={
                    "input": (
                        f"First have the setter open {COOKIE_SET_URL} (this stores a cookie in the "
                        f"shared browser profile). After it confirms, have the checker open "
                        f"{COOKIE_READ_URL} and report the cookies the page lists."
                    )
                }
            )

            print("Status:", result.status)
            if not result.output:
                print("Run failed; see the errors above.")
                return
            owner_output = result.output.get("owner", {}).get("output", {})
            _print_sharing_report()
            print("Surfaced live_view_url:", owner_output.get("live_view_url"))
            # The cookie set by the setter should appear here, proving state crossed the handoff.
            print("Checker final content:", owner_output.get("content"))
        finally:
            for tool in (setter_tool, checker_tool):
                try:
                    tool.close()
                except Exception as exc:  # defensive: don't let cleanup mask the run outcome
                    print(f"Warning: closing {tool.name} failed: {exc}")


if __name__ == "__main__":
    main()
