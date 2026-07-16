from unittest.mock import MagicMock

import pytest

from dynamiq.connections import E2B
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.sandboxes.base import SandboxConfig, SandboxLifecyclePolicy
from dynamiq.sandboxes.e2b import E2BSandbox


@pytest.fixture
def llm():
    return OpenAI(connection=OpenAIConnection(api_key="k"), model="gpt-4o", max_tokens=10)


def _owner(llm, policy):
    return Agent(
        name="Owner", llm=llm, role="r", tools=[],
        sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx")),
        share_sandbox_with_subagents=True,
        sandbox_on_run_end=policy,
    )


def test_default_policy_is_kill(llm):
    assert _owner(llm, SandboxLifecyclePolicy.KILL).sandbox_on_run_end == SandboxLifecyclePolicy.KILL
    # and the field default itself:
    a = Agent(name="A", llm=llm, role="r", tools=[])
    assert a.sandbox_on_run_end == SandboxLifecyclePolicy.KILL


def test_kill_policy_kills(llm):
    a = _owner(llm, SandboxLifecyclePolicy.KILL)
    a.sandbox.backend = MagicMock(supports_pause=True)
    a._apply_sandbox_on_run_end()
    a.sandbox.backend.close.assert_called_once_with(kill=True)
    a.sandbox.backend.pause.assert_not_called()


def test_pause_policy_pauses_when_supported(llm):
    a = _owner(llm, SandboxLifecyclePolicy.PAUSE)
    a.sandbox.backend = MagicMock(supports_pause=True)
    a._apply_sandbox_on_run_end()
    a.sandbox.backend.pause.assert_called_once()
    a.sandbox.backend.close.assert_not_called()


def test_pause_policy_falls_back_to_disconnect_when_unsupported(llm):
    a = _owner(llm, SandboxLifecyclePolicy.PAUSE)
    a.sandbox.backend = MagicMock(supports_pause=False)
    a._apply_sandbox_on_run_end()
    a.sandbox.backend.pause.assert_not_called()
    a.sandbox.backend.close.assert_called_once_with(kill=False)


def test_disconnect_policy_disconnects(llm):
    a = _owner(llm, SandboxLifecyclePolicy.DISCONNECT)
    a.sandbox.backend = MagicMock(supports_pause=True)
    a._apply_sandbox_on_run_end()
    a.sandbox.backend.close.assert_called_once_with(kill=False)


def test_run_result_surfaces_sandbox_id_when_pausing(llm):
    """When the owner persists (pause) an E2B sandbox, the run result carries sandbox_id
    so a caller can resume it later."""
    a = _owner(llm, SandboxLifecyclePolicy.PAUSE)
    # Simulate the owner branch: a live shared session token + a pausable backend with an id.
    a.sandbox.backend = MagicMock(supports_pause=True, current_sandbox_id="sbx-resume-1")
    result = {"content": "done"}
    a._maybe_surface_sandbox_id(result, shared_session_token=object())
    assert result["sandbox_id"] == "sbx-resume-1"


def test_run_result_omits_sandbox_id_when_killing(llm):
    a = _owner(llm, SandboxLifecyclePolicy.KILL)
    a.sandbox.backend = MagicMock(supports_pause=True, current_sandbox_id="sbx-x")
    result = {"content": "done"}
    a._maybe_surface_sandbox_id(result, shared_session_token=object())
    assert "sandbox_id" not in result


def test_run_result_omits_sandbox_id_when_not_owner(llm):
    a = _owner(llm, SandboxLifecyclePolicy.PAUSE)
    a.sandbox.backend = MagicMock(supports_pause=True, current_sandbox_id="sbx-x")
    result = {"content": "done"}
    a._maybe_surface_sandbox_id(result, shared_session_token=None)  # inherited/non-owner
    assert "sandbox_id" not in result
