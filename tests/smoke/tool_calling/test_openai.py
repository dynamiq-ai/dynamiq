"""Strict-tools smoke test for OpenAI (per-tool ``strict`` via OpenAIStrictToolsMixin)."""

import os

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode

from .harness import assert_strict_call_is_clean, run_route_agent

REQUIRED_ENV = ["OPENAI_API_KEY"]
PROVIDER = "openai"

pytestmark = [pytest.mark.smoke, pytest.mark.integration]


def _llm():
    return OpenAI(connection=OpenAIConnection(), model="gpt-4.1", max_tokens=4096, temperature=1)


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
