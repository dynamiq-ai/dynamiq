"""Strict-tools smoke test for Azure OpenAI (per-tool ``strict`` via OpenAIStrictToolsMixin)."""

import os

import pytest

from dynamiq.connections import AzureAI as AzureAIConnection
from dynamiq.nodes.llms import AzureAI
from dynamiq.nodes.types import InferenceMode

from .harness import assert_strict_call_is_clean, run_route_agent

REQUIRED_ENV = ["AZURE_API_KEY", "AZURE_URL", "AZURE_API_VERSION"]
PROVIDER = "azureai"

pytestmark = [pytest.mark.smoke, pytest.mark.integration, pytest.mark.flaky(reruns=3)]


def _llm():
    return AzureAI(connection=AzureAIConnection(), model="azure/gpt-4.1", max_tokens=2048, temperature=1)


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
