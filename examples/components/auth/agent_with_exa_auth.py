"""
Minimal example showing how an agent surfaces Exa tool authentication requests
and how to satisfy them with structured `AuthConfig` data.
"""

from __future__ import annotations

import os

from dynamiq.auth import AuthConfig, AuthCredential, AuthCredentialType, AuthScheme, AuthSchemeLocation, AuthSchemeType
from dynamiq.callbacks import AuthRequestLoggingCallback
from dynamiq.connections import Exa
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.base import ToolAuthConfig
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.runnables import RunnableConfig
from dynamiq.workflow import Workflow
from examples.llm_setup import setup_llm


def build_agent() -> Agent:
    """Create an agent that uses the Exa search tool without pre-configured credentials."""
    llm = setup_llm()
    exa_tool = ExaTool(connection=Exa(api_key=""))

    return Agent(
        name="Research Agent",
        description="Finds the latest articles via Exa search.",
        role="Use the Exa tool when you need fresh information.",
        llm=llm,
        tools=[exa_tool],
        max_loops=2,
    )


def build_workflow(agent: Agent) -> Workflow:
    """Wrap the agent in a Flow/Workflow pair for consistency with other examples."""
    flow = Flow(nodes=[agent])
    return Workflow(flow=flow)


def extract_agent_output(workflow_result, agent: Agent) -> dict:
    agent_result = workflow_result.output.get(agent.id, {})
    return agent_result.get("output", {})


def run_without_credentials(workflow: Workflow, agent: Agent):
    """Invoke the agent without any auth data. The tool reports a missing API key."""
    result = workflow.run(
        input_data={"input": "What is new in reinforcement learning?"},
        config=RunnableConfig(callbacks=[AuthRequestLoggingCallback()]),
    )
    payload = extract_agent_output(result, agent)
    print("--- First Run (no credentials) ---")
    print("status:", payload.get("status"))
    print("content:", payload.get("content"))
    auth_requests = payload.get("auth_requests")
    print("auth requests:", auth_requests)
    if auth_requests:
        for req in auth_requests:
            print("  -> pending auth for tool", req.get("tool_name"), "scheme:", req.get("required", {}).get("scheme"))
    else:
        print("No authentication required; skipping second run.")
    return payload


def run_with_credentials(workflow: Workflow, agent: Agent, api_key: str):
    """
    Re-run the agent with `tool_auth` populated by an AuthConfig object.
    The tool now receives the key via headers and can complete the search.
    """
    auth_config = AuthConfig(
        scheme=AuthScheme(
            type=AuthSchemeType.API_KEY,
            name="x-api-key",
            location=AuthSchemeLocation.HEADER,
            description="Exa API key required for authenticated requests.",
        ),
        credential=AuthCredential(type=AuthCredentialType.API_KEY, api_key=api_key),
        metadata={"provider": "exa"},
    )

    tool_auth = ToolAuthConfig.model_validate({"by_name": {"Exa Search Tool": auth_config}})

    result = workflow.run(
        input_data={
            "input": "What is new in reinforcement learning?",
            "tool_auth": tool_auth,
        },
        config=RunnableConfig(callbacks=[AuthRequestLoggingCallback()]),
    )
    payload = extract_agent_output(result, agent)
    print("--- Second Run (with credentials) ---")
    print("status:", payload.get("status"))
    print("content:", payload.get("content"))
    return payload


if __name__ == "__main__":
    agent = build_agent()
    workflow = build_workflow(agent)
    first = run_without_credentials(workflow, agent)

if first.get("status") == "auth_required":
    api_key = os.environ["EXA_API_KEY"]
    second = run_with_credentials(workflow, agent, api_key)
    print("final content:", second.get("content"))
else:
    print("Workflow succeeded without additional authentication.")
