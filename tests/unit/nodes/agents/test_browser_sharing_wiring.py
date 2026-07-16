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


def test_stagehand_attaches_and_records_under_shared_session(monkeypatch, llm):
    from dynamiq.nodes.agents.shared_session import SharedSession
    from dynamiq.nodes.tools.stagehand import Stagehand

    tool = Stagehand.__new__(Stagehand)  # bypass connection init for a pure-logic unit test
    tool._session_id = None
    tool._live_view_url = "https://lv/x"
    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)

    ss = SharedSession(share_browser=True)
    session_token = _shared_session.set(ss)
    run_token = _current_agent_run.set("runA")
    try:
        # first tool: no shared id yet -> becomes the creator, attach is a no-op
        shared = tool._attach_shared_browser_before_init()
        assert shared is True
        assert ss._lease_owner == "runA"       # lease acquired
        assert tool._session_id is None         # nothing to attach to yet

        # simulate _init_client having created the session
        tool._session_id = "created-sid"
        tool._record_shared_browser_after_init()
        assert ss.browser_session_id() == "created-sid"
        assert ss.browser_live_view_url() == "https://lv/x"

        # second tool under the same session attaches to the recorded id
        tool2 = Stagehand.__new__(Stagehand)
        tool2._session_id = None
        tool2._live_view_url = None
        monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
        run_token2 = _current_agent_run.set("runA")  # same run -> reentrant lease
        try:
            assert tool2._attach_shared_browser_before_init() is True
            assert tool2._session_id == "created-sid"  # attached
        finally:
            _current_agent_run.reset(run_token2)
    finally:
        _current_agent_run.reset(run_token)
        _shared_session.reset(session_token)


def test_stagehand_no_shared_session_is_passthrough(monkeypatch):
    from dynamiq.nodes.tools.stagehand import Stagehand

    tool = Stagehand.__new__(Stagehand)
    tool._session_id = None
    assert tool._attach_shared_browser_before_init() is False  # no session set
    assert tool._session_id is None
