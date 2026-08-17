"""Agent-level end-to-end test: the Stagehand tool driven by a dynamiq Agent.

Exercises the full production path — Agent (Anthropic LLM) -> Stagehand tool -> browser —
rather than calling the tool directly. Requires outbound access to the browser provider.

Keys via environment variables:

    ANTHROPIC_API_KEY                        agent LLM + Stagehand model
    BROWSERBASE_API_KEY / BROWSERBASE_PROJECT_ID   for the Browserbase path
    STEEL_API_KEY                            for the Steel path
    ANTHROPIC_BASE_URL                       leave unset for the Steel path (defaults correctly);
                                             if set, it must include the /v1 suffix

Run:

    python e2e_agent.py browserbase   # agent + Browserbase-backed browser (default)
    python e2e_agent.py steel         # agent + Steel-backed browser
"""

import os
import sys

from dynamiq import Workflow
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import Browserbase as BrowserbaseConnection
from dynamiq.connections import SteelBrowser
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic
from dynamiq.nodes.tools import Stagehand
from dynamiq.runnables import RunnableConfig, RunnableStatus

# Deliberately NOT claude-sonnet-5. Its observe fails on the HOSTED Browserbase v3 server, not just
# the bundled local one: 0/7 on api.stagehand.browserbase.com, always a bare 502. act, extract and
# go_back are fine — observe alone is broken. The Model Gateway refuses sonnet-5 outright (400,
# unsupported model), so no key arrangement makes it usable there. Revisit once Browserbase ships a
# fix; `fallback_model_name` also covers it now, at the cost of a failed round-trip per observe.
MODEL = "anthropic/claude-sonnet-4-6"
TASK = (
    "Open https://example.com, extract the page heading and the first paragraph, "
    "then follow the 'More information...' link and report the heading of the page you land on. "
    "Answer with both headings."
)


def build_tool(target: str) -> Stagehand:
    if target == "steel":
        connection = SteelBrowser(model_api_key=os.environ["ANTHROPIC_API_KEY"])
    else:
        connection = BrowserbaseConnection(model_api_key=os.environ["ANTHROPIC_API_KEY"])
    return Stagehand(
        connection=connection,
        model_name=MODEL,
        is_return_live_view_url_enabled=True,
        is_postponed_component_init=True,
    )


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "browserbase"
    agent = Agent(
        id="browser-agent",
        name="browser-agent",
        llm=Anthropic(
            id="anthropic-llm",
            connection=AnthropicConnection(),
            model="claude-sonnet-5",
            temperature=0.0,
            max_tokens=2000,
            is_postponed_component_init=True,
        ),
        tools=[build_tool(target)],
        role="You browse the web with your Stagehand tool and answer precisely.",
        max_loops=12,
        is_postponed_component_init=True,
    )

    with get_connection_manager() as cm:
        wf = Workflow(flow=Flow(connection_manager=cm, init_components=True, nodes=[agent]))
        result = wf.run(input_data={"input": TASK}, config=RunnableConfig())

    output = result.output.get(agent.id, {})
    print("status:", result.status)
    print("answer:", str(output.get("output", {}).get("content"))[:1000])

    ok = result.status == RunnableStatus.SUCCESS
    answer = str(output.get("output", {}).get("content", "")).lower()
    if ok and "example domain" not in answer:
        print("WARNING: answer does not mention the example.com heading — inspect the transcript")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
