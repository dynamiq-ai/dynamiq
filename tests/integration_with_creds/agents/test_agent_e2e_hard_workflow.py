"""End-to-end "hard workflow" test across the provider x inference-mode matrix.

A single agent researches on the web (Exa), builds a small website + Flask backend, and
verifies it in an E2B sandbox. We assert the task finishes WITHOUT triggering recovery
(the agent loop's correction on a parse/tool-call failure), detected two ways: scanning
``agent._prompt.messages`` for recovery markers, and checking the ALL-mode stream for a
``tool_input_error`` event.

The matrix spans every LLM provider exposed by ``dynamiq.nodes.llms``. Per-provider creds
plus EXA_API_KEY and E2B_API_KEY are required; combos with missing creds are skipped.
"""

import os

import pytest

from dynamiq import Workflow
from dynamiq import connections as conn
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
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


def _llm_factory(node_cls, connection_cls, model, **connection_kwargs):
    """Build a zero-arg LLM factory; connection construction is deferred so combos with
    missing creds skip before any env var is read (connections raise on absent env keys)."""

    def factory():
        return node_cls(connection=connection_cls(**connection_kwargs), model=model, **COMMON_LLM_KWARGS)

    return factory


def _ollama_llm():
    # Opt-in: only runs when OLLAMA_URL points at a reachable server (no API key concept).
    return llms.Ollama(connection=conn.Ollama(url=os.environ["OLLAMA_URL"]), model="llama3.1:8b", **COMMON_LLM_KWARGS)


# GCP service-account fields are all required by the VertexAI connection, so every env var
# must be present for the combo to run.
VERTEXAI_ENV_KEYS = [
    "VERTEXAI_PROJECT_ID",
    "VERTEXAI_PROJECT_LOCATION",
    "GOOGLE_CLOUD_PROJECT_ID",
    "GOOGLE_CLOUD_PRIVATE_KEY_ID",
    "GOOGLE_CLOUD_PRIVATE_KEY",
    "GOOGLE_CLOUD_CLIENT_EMAIL",
    "GOOGLE_CLOUD_CLIENT_ID",
    "GOOGLE_CLOUD_AUTH_URI",
    "GOOGLE_CLOUD_TOKEN_URI",
    "GOOGLE_CLOUD_AUTH_PROVIDER_X509_CERT_URL",
    "GOOGLE_CLOUD_CLIENT_X509_CERT_URL",
    "GOOGLE_CLOUD_UNIVERSE_DOMAIN",
]

# provider id -> (required env keys, LLM factory). Model choices favor each provider's
# strongest tool-calling model; MODEL_PREFIX is auto-prepended by each node class.
PROVIDERS = {
    "openai": (["OPENAI_API_KEY"], _llm_factory(llms.OpenAI, conn.OpenAI, "gpt-4.1")),
    "anthropic": (["ANTHROPIC_API_KEY"], _llm_factory(llms.Anthropic, conn.Anthropic, "claude-sonnet-4-5")),
    "gemini": (["GEMINI_API_KEY"], _llm_factory(llms.Gemini, conn.Gemini, "gemini-2.5-pro")),
    "bedrock": (
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
        # Cross-region inference profile; switch the us./eu./apac. prefix to one enabled on the account.
        _llm_factory(llms.Bedrock, conn.AWS, "bedrock/us.anthropic.claude-sonnet-4-6"),
    ),
    "ai21": (["AI21_API_KEY"], _llm_factory(llms.AI21, conn.AI21, "jamba-large")),
    "anyscale": (
        ["ANYSCALE_API_KEY"],
        _llm_factory(llms.Anyscale, conn.Anyscale, "meta-llama/Meta-Llama-3-70B-Instruct"),
    ),
    "azureai": (
        # Model is the Azure deployment name -- adjust to one provisioned on the resource.
        ["AZURE_API_KEY", "AZURE_URL", "AZURE_API_VERSION"],
        _llm_factory(llms.AzureAI, conn.AzureAI, "gpt-4.1"),
    ),
    "cerebras": (["CEREBRAS_API_KEY"], _llm_factory(llms.Cerebras, conn.Cerebras, "llama-3.3-70b")),
    "cohere": (["COHERE_API_KEY"], _llm_factory(llms.Cohere, conn.Cohere, "command-a-03-2025")),
    "databricks": (
        ["DATABRICKS_API_KEY", "DATABRICKS_API_BASE"],
        _llm_factory(llms.Databricks, conn.Databricks, "databricks-meta-llama-3-3-70b-instruct"),
    ),
    "deepinfra": (
        ["DEEPINFRA_API_KEY"],
        _llm_factory(llms.DeepInfra, conn.DeepInfra, "meta-llama/Llama-3.3-70B-Instruct"),
    ),
    "deepseek": (["DEEPSEEK_API_KEY"], _llm_factory(llms.DeepSeek, conn.DeepSeek, "deepseek-chat")),
    "fireworksai": (
        ["FIREWORKS_AI_API_KEY"],
        _llm_factory(llms.FireworksAI, conn.FireworksAI, "accounts/fireworks/models/llama-v3p3-70b-instruct"),
    ),
    "groq": (["GROQ_API_KEY"], _llm_factory(llms.Groq, conn.Groq, "llama-3.3-70b-versatile")),
    "huggingface": (
        ["HUGGINGFACE_API_KEY"],
        _llm_factory(llms.HuggingFace, conn.HuggingFace, "meta-llama/Meta-Llama-3.1-70B-Instruct"),
    ),
    "mistral": (["MISTRAL_API_KEY"], _llm_factory(llms.Mistral, conn.Mistral, "mistral-large-latest")),
    "nvidia_nim": (
        ["NVIDIA_NIM_API_KEY", "NVIDIA_NIM_URL"],
        _llm_factory(llms.NvidiaNIM, conn.NvidiaNIM, "meta/llama-3.3-70b-instruct"),
    ),
    "ollama": (["OLLAMA_URL"], _ollama_llm),
    "openrouter": (["OPENROUTER_API_KEY"], _llm_factory(llms.OpenRouter, conn.OpenRouter, "openai/gpt-4.1")),
    "perplexity": (["PERPLEXITYAI_API_KEY"], _llm_factory(llms.Perplexity, conn.Perplexity, "sonar-pro")),
    "replicate": (
        ["REPLICATE_API_KEY"],
        _llm_factory(llms.Replicate, conn.Replicate, "meta/meta-llama-3-70b-instruct"),
    ),
    "sambanova": (
        ["SAMBANOVA_API_KEY"],
        _llm_factory(llms.SambaNova, conn.SambaNova, "Meta-Llama-3.3-70B-Instruct"),
    ),
    "togetherai_kimi": (
        ["TOGETHER_API_KEY"],
        _llm_factory(llms.TogetherAI, conn.TogetherAI, "moonshotai/kimi-k2.6"),
    ),
    "vertexai": (VERTEXAI_ENV_KEYS, _llm_factory(llms.VertexAI, conn.VertexAI, "gemini-2.5-pro")),
    "watsonx": (
        ["WATSONX_API_KEY", "WATSONX_PROJECT_ID", "WATSONX_URL"],
        _llm_factory(llms.WatsonX, conn.WatsonX, "meta-llama/llama-3-3-70b-instruct"),
    ),
    "xai": (["XAI_API_KEY"], _llm_factory(llms.xAI, conn.xAI, "grok-3")),
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
    sandbox_backend = E2BSandbox(connection=conn.E2B())
    try:
        agent = Agent(
            name=f"E2EHardWorkflowAgent_{provider}_{inference_mode.value}",
            id=f"e2e_hard_workflow_{provider}_{inference_mode.value}",
            llm=llm,
            role=AGENT_ROLE,
            tools=[ExaTool(connection=conn.Exa(), include_full_content=True)],
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
        tool_input_errors = [content for step, content in ordered_events if step == "tool_input_error"]
        assert not tool_input_errors, (
            f"[{provider}/{inference_mode.value}] streamed {len(tool_input_errors)} tool_input_error "
            f"event(s) (recovery occurred):\n"
            + "\n".join(
                f"- loop {e.get('loop_num')}: {e.get('error')}" if isinstance(e, dict) else f"- {e}"
                for e in tool_input_errors
            )
        )

        recovery_events = find_recovery_events(agent)
        assert not recovery_events, (
            f"[{provider}/{inference_mode.value}] agent triggered {len(recovery_events)} recovery "
            f"event(s):\n" + "\n".join(f"- [{kind}] {content}" for kind, content in recovery_events)
        )

        logger.info(f"--- E2E hard workflow passed clean (no recovery) for {provider}/{inference_mode.value} ---")
    finally:
        # kill=True terminates the remote sandbox; close() alone only disconnects and would
        # leave it alive (1h timeout), piling up live sandboxes across the parallel matrix.
        sandbox_backend.close(kill=True)
