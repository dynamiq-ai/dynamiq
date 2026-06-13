"""Strict-tools smoke test for Together AI (per-tool ``strict`` via OpenAIStrictToolsMixin).

The harness registers ``together_ai/zai-org/GLM-5`` with LiteLLM so its function-calling
support isn't stripped from the request.
"""

import os

import pytest

from dynamiq.connections import TogetherAI as TogetherAIConnection
from dynamiq.nodes.llms import TogetherAI
from dynamiq.nodes.types import InferenceMode

from .harness import assert_strict_call_is_clean, run_route_agent

REQUIRED_ENV = ["TOGETHER_API_KEY"]
PROVIDER = "togetherai"

pytestmark = [pytest.mark.smoke, pytest.mark.integration, pytest.mark.flaky(reruns=3)]


def _llm():
    return TogetherAI(
        connection=TogetherAIConnection(), model="together_ai/zai-org/GLM-5", max_tokens=2048, temperature=1
    )


def _skip_if_no_creds():
    missing = [key for key in REQUIRED_ENV if not os.getenv(key)]
    if missing:
        pytest.skip(f"[{PROVIDER}] missing credentials: {missing}")


@pytest.mark.timeout(150)
def test_strict_tool_call_is_clean():
    """strict_tools=True: the agent's first tool call is enum-valid with no extra keys and no recovery."""
    _skip_if_no_creds()
    run = run_route_agent(_llm(), strict_tools=True, inference_mode=InferenceMode.FUNCTION_CALLING)
    assert_strict_call_is_clean(run, label=f"{PROVIDER}-fc-strict")
