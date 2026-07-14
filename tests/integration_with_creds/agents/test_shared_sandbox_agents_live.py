"""Live end-to-end test: an owner agent and a factory-built subagent share one
E2B sandbox (requires OPENAI_API_KEY and E2B_API_KEY).

Only *factory-mode* subagents share the owner's sandbox: they are constructed
during the owner's ``execute()`` while the shared-session ContextVar is set, so
their tools resolve onto a view of the owner's sandbox. This test therefore
wires the subagent via ``SubAgentTool(agent_factory=...)`` (an initialized
``agent=`` subagent would NOT share — see spec 8.4).

It drives the whole stack once and verifies sharing by inspecting the owner's
live sandbox after the run: a file the subagent wrote must be visible on the
owner's backend, which is only possible if they share one sandbox.

Expensive and LLM-nondeterministic; skipped without both credentials.
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
def test_owner_and_factory_subagent_share_sandbox_live():
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("E2B_API_KEY"):
        pytest.skip("OPENAI_API_KEY and E2B_API_KEY are required for this test.")

    marker = uuid.uuid4().hex[:12]
    proof_path = f"/home/user/proof-{marker}.txt"

    owner_backend = E2BSandbox(connection=E2BConnection())

    def researcher_factory() -> Agent:
        # Nested BaseModels (the llm) MUST be constructed inside the factory so
        # they are not shared/mutated across invocations.
        return Agent(
            name="researcher",
            role=(
                "You are a Researcher sub-agent with a shell tool in a Linux sandbox.\n"
                "When asked to write a file, use your shell tool to create it exactly "
                "at the absolute path you are given, then confirm it is written."
            ),
            description="Writes a file to an absolute path using its shell tool.",
            llm=_openai_llm(),
            tools=[],
            inference_mode=InferenceMode.XML,
            parallel_tool_calls_enabled=False,
            max_loops=6,
        )

    researcher_tool = SubAgentTool(
        name="researcher",
        description="Delegate a shell/file task to the researcher sub-agent. Pass {'input': '<task>'}.",
        agent_factory=researcher_factory,
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
        share_sandbox_with_subagents=True,
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
                    f"'{marker}' into the file at the absolute path {proof_path} "
                    "using its shell tool, then confirm it is done."
                ),
            },
            config=RunnableConfig(),
        )

        assert result.status == RunnableStatus.SUCCESS

        # Robust check, independent of the LLM's prose: the subagent's write must
        # be visible on the OWNER's live backend — only possible if the factory
        # subagent ran on a view of the owner's shared sandbox.
        assert owner_backend.exists(proof_path) is True
        assert marker.encode() in owner_backend.retrieve(proof_path)
    finally:
        owner_backend.close(kill=True)
