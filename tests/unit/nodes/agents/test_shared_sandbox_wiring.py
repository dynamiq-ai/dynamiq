import pytest

from dynamiq.connections import E2B
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.shared_session import _shared_session
from dynamiq.nodes.llms import OpenAI
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox


@pytest.fixture
def test_llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


def _sandboxed_agent(llm, **kwargs):
    return Agent(
        name=kwargs.pop("name", "Owner"),
        llm=llm,
        role="r",
        tools=[],
        sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared")),
        **kwargs,
    )


def test_owner_enters_session_when_flag_and_sandbox(test_llm):
    agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=True)
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    try:
        ss = _shared_session.get()
        assert ss is not None
        assert ss.share_sandbox is True
        assert ss.get_sandbox() is agent.sandbox_backend
    finally:
        agent._exit_shared_session(token)
    assert _shared_session.get() is None


def test_no_session_when_flag_off(test_llm):
    agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=False)
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    assert token is None
    assert _shared_session.get() is None


def test_no_session_when_no_sandbox(test_llm):
    agent = Agent(name="Owner", llm=test_llm, role="r", tools=[], share_sandbox_with_subagents=True)
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    assert token is None
    assert _shared_session.get() is None


def test_inherits_existing_session(test_llm):
    from dynamiq.nodes.agents.shared_session import SharedSession

    outer = SharedSession(sandbox=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="outer"), share_sandbox=True)
    outer_token = _shared_session.set(outer)
    try:
        agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=True)
        token = agent._maybe_enter_shared_session({"run_id": "run-2"})
        assert token is None  # inherited, did not create a new session
        assert _shared_session.get() is outer
    finally:
        _shared_session.reset(outer_token)


def test_cleanup_skips_shared_sandbox():
    from unittest.mock import MagicMock

    from dynamiq.nodes.tools.agent_tool import SubAgentTool

    agent = MagicMock()
    agent._sandbox_is_shared = True
    SubAgentTool.cleanup_factory_agent(agent)
    agent.sandbox_backend.close.assert_not_called()


def test_cleanup_kills_owned_sandbox():
    from unittest.mock import MagicMock

    from dynamiq.nodes.tools.agent_tool import SubAgentTool

    agent = MagicMock()
    agent._sandbox_is_shared = False
    SubAgentTool.cleanup_factory_agent(agent)
    agent.sandbox_backend.close.assert_called_once_with(kill=True)
