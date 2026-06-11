"""End-to-end "hard workflow" test across the provider x inference-mode matrix.

A single agent researches on the web (Exa), builds a small website + Flask backend, and
verifies it in an E2B sandbox. We assert the task finishes WITHOUT triggering recovery
(the agent loop's correction on a parse/tool-call failure), detected two ways: scanning
``agent._prompt.messages`` for recovery markers, and checking the ALL-mode stream for a
``tool_input_error`` event.

Per-provider creds (OPENAI/ANTHROPIC/GEMINI/AWS) plus EXA_API_KEY and E2B_API_KEY are
required; combos with missing creds are skipped.
"""

import os

import pytest

from dynamiq import Workflow
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import Exa as ExaConnection
from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic, Bedrock, Gemini, OpenAI
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from dynamiq.utils.logger import logger

from .streaming_assertions import collect_streaming_events

# Provider matrix -- one representative model each (full model sweeps live in test_agent_llms.py).
COMMON_LLM_KWARGS = dict(max_tokens=20000, temperature=0)


def _openai_llm():
    return OpenAI(connection=OpenAIConnection(), model="gpt-4.1", **COMMON_LLM_KWARGS)


def _anthropic_llm():
    return Anthropic(connection=AnthropicConnection(), model="claude-sonnet-4-5", **COMMON_LLM_KWARGS)


def _gemini_llm():
    return Gemini(connection=GeminiConnection(), model="gemini-2.5-pro", **COMMON_LLM_KWARGS)


def _bedrock_llm():
    # Cross-region inference profile; switch the us./eu./apac. prefix to one enabled on the account.
    return Bedrock(
        connection=AWSConnection(),
        model="bedrock/us.anthropic.claude-sonnet-4-6",
        **COMMON_LLM_KWARGS,
    )


PROVIDERS = {
    "openai": (["OPENAI_API_KEY"], _openai_llm),
    "anthropic": (["ANTHROPIC_API_KEY"], _anthropic_llm),
    "gemini": (["GEMINI_API_KEY"], _gemini_llm),
    "bedrock": (["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"], _bedrock_llm),
}

INFERENCE_MODES = [
    InferenceMode.XML,
    InferenceMode.STRUCTURED_OUTPUT,
    InferenceMode.FUNCTION_CALLING
]
INFERENCE_MODE_IDS = ["xml", "structured_output", "function_calling"]

# Required for every combo regardless of provider.
TOOL_ENV_KEYS = ["EXA_API_KEY", "E2B_API_KEY"]

HARD_TASK = (
    "First, research the current stable major version of the Flask Python web framework and one "
    "recommended way to structure a minimal Flask app. Then build a minimal website: create an "
    "`index.html` page and a Flask backend `app.py` that serves that page on `/` and exposes a "
    "`/api/health` endpoint returning JSON `{\"status\": \"ok\"}`. Save BOTH files to "
    "`/home/user/output` so they are returned. Finally, verify the backend imports cleanly by "
    "running `python -c \"import app\"` from `/home/user/output` in the sandbox; if it fails, fix "
    "the code and re-verify. Report the Flask version you used in your final answer."
)

AGENT_ROLE = (
    "You are a senior full-stack engineer. You research before you build, write real files to the "
    "sandbox filesystem, and verify your code runs before reporting success. Save all deliverable "
    "files to /home/user/output. Write Python to script files and execute them rather than using "
    "fragile one-liners."
)

EXPECTED_FILES = ["app.py", "index.html"]


@pytest.fixture(scope="module")
def run_config():
    # Research + codegen + sandbox execution; matches the 150s per-case timeout cap.
    return RunnableConfig(request_timeout=150)


def find_recovery_events(agent: Agent) -> list[tuple[str, str]]:
    """Scan the agent's prompt for recovery corrections appended by the loop.

    Mirrors the marker-scan in test_agent_cancellation_fc_memory.py. Two markers exist:
    - FUNCTION_CALLING mode: a ``tool``-role "Tool call failed: ..." reply.
    - DEFAULT/XML/STRUCTURED_OUTPUT: a ``user``-role "Correction Instruction: ..." message.
    """
    events: list[tuple[str, str]] = []
    for message in agent._prompt.messages:
        content = message.content or ""
        if message.role == MessageRole.TOOL and "Tool call failed: the previous call could not be processed" in content:
            events.append(("function_calling", content))
        elif message.role == MessageRole.USER and content.startswith("Correction Instruction:"):
            events.append(("correction", content))
    return events


@pytest.mark.smoke  # gated behind the `run-smoke-tests` PR label; excluded from default suites
@pytest.mark.integration
@pytest.mark.timeout(150)  # hard per-case wall-clock cap (requires pytest-timeout)
@pytest.mark.parametrize("provider", list(PROVIDERS), ids=list(PROVIDERS))
@pytest.mark.parametrize("inference_mode", INFERENCE_MODES, ids=INFERENCE_MODE_IDS)
def test_e2e_hard_workflow_no_recovery(provider, inference_mode, run_config):
    """Run the hard research+build task and assert success, returned files, valid streaming,
    and ZERO recovery events -- across every provider x inference-mode combination."""
    required_env_keys, llm_factory = PROVIDERS[provider]
    missing = [key for key in (*required_env_keys, *TOOL_ENV_KEYS) if not os.getenv(key)]
    if missing:
        pytest.skip(f"Missing credentials for {provider}/{inference_mode.value}: {missing}")

    llm = llm_factory()
    sandbox_backend = E2BSandbox(connection=E2BConnection())
    try:
        agent = Agent(
            name=f"E2EHardWorkflowAgent_{provider}_{inference_mode.value}",
            id=f"e2e_hard_workflow_{provider}_{inference_mode.value}",
            llm=llm,
            role=AGENT_ROLE,
            tools=[ExaTool(connection=ExaConnection())],
            sandbox=SandboxConfig(enabled=True, backend=sandbox_backend),
            inference_mode=inference_mode,
            max_loops=25,
            parallel_tool_calls_enabled=True,
            verbose=True,
            streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
        )

        streaming = StreamingIteratorCallbackHandler()
        workflow = Workflow(flow=Flow(nodes=[agent]))
        result = workflow.run(
            input_data={"input": HARD_TASK},
            config=run_config.model_copy(update={"callbacks": [streaming]}),
        )

        assert (
            result.status == RunnableStatus.SUCCESS
        ), f"[{provider}/{inference_mode.value}] run failed: {result.output}"

        agent_output = result.output[agent.id]["output"]
        content = agent_output["content"]
        assert isinstance(content, str) and content, "Agent final content should be a non-empty string"

        # Both deliverable files came back from the sandbox.
        returned_files = agent_output.get("files") or []
        returned_names = {f.name for f in returned_files}
        for name in EXPECTED_FILES:
            assert name in returned_names, (
                f"[{provider}/{inference_mode.value}] expected file '{name}' missing "
                f"from returned files: {returned_names}"
            )

        ordered_events = collect_streaming_events(streaming, agent.id)

        # The agent actually researched -- the Exa tool ran at least once.
        exa_tool_runs = [
            content
            for step, content in ordered_events
            if step == "tool" and isinstance(content, dict) and content.get("name") == "exa-search"
        ]
        assert exa_tool_runs, f"[{provider}/{inference_mode.value}] Exa research tool was never invoked"

        # No recovery occurred -- checked on both the stream and the prompt.
        assert not any(
            step == "tool_input_error" for step, _ in ordered_events
        ), f"[{provider}/{inference_mode.value}] streamed a tool_input_error (recovery occurred)"

        recovery_events = find_recovery_events(agent)
        assert not recovery_events, (
            f"[{provider}/{inference_mode.value}] agent triggered {len(recovery_events)} recovery "
            f"event(s): {[kind for kind, _ in recovery_events]}"
        )

        logger.info(f"--- E2E hard workflow passed clean (no recovery) for {provider}/{inference_mode.value} ---")
    finally:
        sandbox_backend.close()
