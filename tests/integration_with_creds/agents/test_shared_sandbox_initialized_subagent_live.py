"""Live e2e: an owner agent and an *initialized* (agent=<instance>) subagent share one
E2B sandbox (requires OPENAI_API_KEY and E2B_API_KEY).

This is the P1.5 case P1 could not do: the researcher is a pre-built Agent instance
(constructed before the run, with no sandbox of its own), wired via
``SubAgentTool(agent=researcher)``. Run-time resolution routes it onto the owner's
shared sandbox. Verified by inspecting the owner's live backend after the run.
"""

import os
import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.connections import E2B as E2BConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox


def _openai_llm() -> OpenAI:
    return OpenAI(model="gpt-4o-mini", connection=connections.OpenAI())


@pytest.mark.integration
def test_owner_and_initialized_subagent_share_sandbox_live():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("E2B_API_KEY"):
        pytest.skip("OPENAI_API_KEY and E2B_API_KEY are required for this test.")

    marker = uuid.uuid4().hex[:12]
    proof_name = f"proof-{marker}.txt"

    owner_backend = E2BSandbox(connection=E2BConnection())

    # Pre-built (initialized) subagent with NO sandbox of its own — the P1.5 case.
    researcher = Agent(
        name="researcher",
        role=(
            "You are a Researcher sub-agent with a shell tool in a Linux sandbox.\n"
            "When asked to write a file, use your shell tool to create it in your working "
            "directory, then confirm it is written."
        ),
        description="Writes a file using its shell tool.",
        llm=_openai_llm(),
        tools=[],
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        max_loops=6,
    )
    researcher_tool = SubAgentTool(
        name="researcher",
        description="Delegate a shell/file task to the researcher sub-agent. Pass {'input': '<task>'}.",
        agent=researcher,
    )

    owner = Agent(
        name="owner",
        role=(
            "You coordinate a Researcher sub-agent that shares your sandbox filesystem.\n"
            "Always invoke the tool as {'input': '<task>'}. Delegate file writes to the researcher."
        ),
        description="Owner agent that shares its sandbox with its subagents.",
        llm=_openai_llm(),
        tools=[researcher_tool],
        sandbox=SandboxConfig(enabled=True, backend=owner_backend),
        share_sandbox_with_subagents=True,  # sandbox_sharing_scope defaults to ALL
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        max_loops=8,
    )

    wf = Workflow(flow=Flow(nodes=[owner]))
    try:
        result = wf.run(
            input_data={
                "input": (
                    "Ask the researcher sub-agent to write the exact text "
                    f"'{marker}' into a file named {proof_name} in its working directory "
                    "using its shell tool, then confirm it is done."
                ),
            },
            config=RunnableConfig(),
        )

        assert result.status == RunnableStatus.SUCCESS

        # The subagent's write must be visible on the OWNER's live backend - only possible if the
        # subagent ran on a view of the owner's shared sandbox. Searched for by name: the subagent
        # is told to keep its files under its own working directory, so the exact path is its choice.
        found = owner_backend.run_command_shell(f"grep -rl --include={proof_name} {marker} /home/user")
        assert proof_name in found.stdout, f"{proof_name} with the marker is not on the owner's sandbox"
    finally:
        owner_backend.close(kill=True)
