from unittest.mock import MagicMock

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.shared_session import _current_agent_run, _shared_session
from dynamiq.nodes.llms import OpenAI


@pytest.fixture
def llm():
    return OpenAI(connection=OpenAIConnection(api_key="k"), model="gpt-4o", max_tokens=10)


def test_browser_flag_enters_session_without_sandbox(llm):
    agent = Agent(name="Owner", llm=llm, role="r", tools=[], share_browser_session_with_subagents=True)
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    try:
        ss = _shared_session.get()
        assert ss is not None
        assert ss.share_browser is True
        assert ss.share_sandbox is False  # no sandbox configured
    finally:
        agent._exit_shared_session(token)
    assert _shared_session.get() is None


def test_no_session_when_both_flags_off(llm):
    agent = Agent(name="Owner", llm=llm, role="r", tools=[])
    assert agent._maybe_enter_shared_session({"run_id": "run-1"}) is None
    assert _shared_session.get() is None


def test_finally_releases_lease_and_owner_closes_browser(llm):
    """The owner's teardown releases the current run's lease and closes the shared browser."""
    agent = Agent(name="Owner", llm=llm, role="r", tools=[], share_browser_session_with_subagents=True)
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    ss = _shared_session.get()
    ss.close_browser = MagicMock()
    ss.release_browser = MagicMock()
    run_token = _current_agent_run.set("owner-key")
    try:
        agent._teardown_shared_browser(shared_session_token=token)
    finally:
        _current_agent_run.reset(run_token)
        agent._exit_shared_session(token)
    ss.release_browser.assert_called_once_with("owner-key")
    ss.close_browser.assert_called_once()


def test_non_owner_releases_lease_but_does_not_close(llm):
    agent = Agent(name="Sub", llm=llm, role="r", tools=[])
    # simulate an inherited (non-owner) session
    from dynamiq.nodes.agents.shared_session import SharedSession

    ss = SharedSession(share_browser=True)
    outer = _shared_session.set(ss)
    ss.close_browser = MagicMock()
    ss.release_browser = MagicMock()
    run_token = _current_agent_run.set("sub-key")
    try:
        agent._teardown_shared_browser(shared_session_token=None)  # non-owner: token is None
    finally:
        _current_agent_run.reset(run_token)
        _shared_session.reset(outer)
    ss.release_browser.assert_called_once_with("sub-key")
    ss.close_browser.assert_not_called()
