import asyncio
from unittest.mock import MagicMock

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.shared_session import SharedSession, _current_agent_run, _shared_session
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.stagehand import Stagehand


@pytest.fixture
def llm():
    return OpenAI(connection=OpenAIConnection(api_key="k"), model="gpt-4o", max_tokens=10)


def bare_stagehand(**attrs) -> Stagehand:
    """A Stagehand instance without connection/pydantic init, for pure-logic unit tests.

    Pydantic's internals have to be stubbed too, or any attribute the test does not set falls
    through to ``__getattr__`` and raises — including during ``__del__`` at collection.
    """
    tool = Stagehand.__new__(Stagehand)
    object.__setattr__(tool, "__pydantic_fields_set__", set())
    object.__setattr__(tool, "__pydantic_extra__", None)
    defaults = {
        "name": "T",
        "id": "id-1",
        "client": None,
        "_session_id": None,
        "_live_view_url": None,
        "_shares_browser_session": False,
        "_browserbase_client": None,
        "_steel_client": None,
        "_steel_browser_session": None,
        "_loop": None,
        "_loop_thread": None,
        "browser_context_id": None,
        "shared_browser_session_timeout": 3600,
    }
    merged = {**defaults, **attrs}
    # Private attrs must live in __pydantic_private__, not the instance __dict__: an entry in
    # __dict__ shadows it, so the code's own writes would become invisible to reads.
    object.__setattr__(tool, "__pydantic_private__", {k: v for k, v in merged.items() if k.startswith("_")})
    for key, value in merged.items():
        if not key.startswith("_"):
            object.__setattr__(tool, key, value)
    return tool


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


def test_finally_releases_page_control(llm):
    """Every agent hands the page back at turn end, under its own run key."""
    agent = Agent(name="Owner", llm=llm, role="r", tools=[], share_browser_session_with_subagents=True)
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    ss = _shared_session.get()
    ss.release_page_control = MagicMock()
    ss.end_browser_session = MagicMock()
    run_token = _current_agent_run.set("owner-key")
    try:
        agent._teardown_shared_browser(token)
    finally:
        _current_agent_run.reset(run_token)
        agent._exit_shared_session(token)
    ss.release_page_control.assert_called_once_with("owner-key")
    ss.end_browser_session.assert_called_once()  # owner ends the run's session


def test_subagent_teardown_releases_but_does_not_end_the_session(llm):
    """A subagent must never end the session other agents are still using."""
    agent = Agent(name="Sub", llm=llm, role="r", tools=[])
    ss = SharedSession(share_browser=True)
    ss.release_page_control = MagicMock()
    ss.end_browser_session = MagicMock()
    outer = _shared_session.set(ss)
    run_token = _current_agent_run.set("sub-key")
    try:
        agent._teardown_shared_browser(None)  # non-owner: token is None
    finally:
        _current_agent_run.reset(run_token)
        _shared_session.reset(outer)
    ss.release_page_control.assert_called_once_with("sub-key")
    ss.end_browser_session.assert_not_called()


def test_agent_releases_the_page_around_a_delegate_call(llm):
    agent = Agent(name="Owner", llm=llm, role="r", tools=[], share_browser_session_with_subagents=True)
    ss = SharedSession(share_browser=True)
    ss.acquire_page_control("owner-key")
    session_token = _shared_session.set(ss)
    run_token = _current_agent_run.set("owner-key")
    try:
        agent._release_shared_browser_for_delegate(True)
        assert ss._page_control_key is None  # subagent can take it immediately
    finally:
        _current_agent_run.reset(run_token)
        _shared_session.reset(session_token)


def test_agent_keeps_the_page_for_parallel_or_non_delegate_calls(llm):
    """A sibling browser call of ours may be mid-command on that same page."""
    agent = Agent(name="Owner", llm=llm, role="r", tools=[], share_browser_session_with_subagents=True)
    ss = SharedSession(share_browser=True)
    ss.acquire_page_control("owner-key")
    session_token = _shared_session.set(ss)
    run_token = _current_agent_run.set("owner-key")
    try:
        agent._release_shared_browser_for_delegate(False)
        assert ss._page_control_key == "owner-key"
    finally:
        _current_agent_run.reset(run_token)
        _shared_session.reset(session_token)


def test_stagehand_takes_page_control(monkeypatch):
    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
    tool = bare_stagehand()

    ss = SharedSession(share_browser=True)
    session_token = _shared_session.set(ss)
    run_token = _current_agent_run.set("runA")
    try:
        assert asyncio.run(tool._acquire_shared_browser()) is ss
        assert ss._page_control_key == "runA"
    finally:
        _current_agent_run.reset(run_token)
        _shared_session.reset(session_token)


def test_stagehand_no_shared_session_is_passthrough(monkeypatch):
    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
    tool = bare_stagehand()
    assert asyncio.run(tool._acquire_shared_browser()) is None


def test_stagehand_skips_sharing_when_no_current_agent_run(monkeypatch, caplog):
    """Without a run key the agent's finally could never release, locking the page for the run."""
    import logging

    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
    tool = bare_stagehand()

    ss = SharedSession(share_browser=True)
    session_token = _shared_session.set(ss)
    assert _current_agent_run.get() is None
    try:
        with caplog.at_level(logging.WARNING):
            assert asyncio.run(tool._acquire_shared_browser()) is None
        assert any("no _current_agent_run set" in r.getMessage() for r in caplog.records)
        assert ss._page_control_key is None
    finally:
        _shared_session.reset(session_token)


def test_second_agent_attaches_to_the_shared_session(monkeypatch):
    """The whole point: later agents join the LIVE session rather than creating their own."""
    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
    tool = bare_stagehand(connection=MagicMock(browserbase_project_id="proj-1"))

    ss = SharedSession(share_browser=True)
    ss.adopt_browser_session_id("sess-shared")
    asyncio.run(tool._join_shared_browser(ss))

    assert tool._session_id == "sess-shared"  # resumes it via _init_client
    assert tool._shares_browser_session is True  # so close() detaches instead of ending


def test_first_agent_creates_the_context_then_publishes_its_session(monkeypatch):
    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
    creates = []

    class FakeContexts:
        async def create(self, project_id):
            creates.append(project_id)
            return MagicMock(id="ctx-1")

    connection = MagicMock(browserbase_project_id="proj-1", browserbase_api_key="k")
    tool = bare_stagehand(_browserbase_client=MagicMock(contexts=FakeContexts()), connection=connection)

    ss = SharedSession(share_browser=True)
    asyncio.run(tool._join_shared_browser(ss))
    assert creates == ["proj-1"]
    assert ss.browser_context_id() == "ctx-1"
    assert tool._session_id is None  # nothing to attach to: _init_client will create the session

    tool._session_id = "sess-new"
    tool._record_shared_browser_session(ss)
    assert ss.browser_session_id() == "sess-new"  # later agents will attach to this
    assert ss._browser_end_fn is not None  # and the owner can end it without this tool


def test_configured_context_id_is_reused_instead_of_creating_one(monkeypatch):
    """Cross-run persistence: a caller supplies a stable Context (e.g. per end user)."""
    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
    created = []

    class FakeContexts:
        async def create(self, project_id):
            created.append(project_id)
            return MagicMock(id="ctx-new")

    tool = bare_stagehand(
        browser_context_id="ctx-existing",
        _browserbase_client=MagicMock(contexts=FakeContexts()),
        connection=MagicMock(browserbase_project_id="proj-1", browserbase_api_key="k"),
    )

    ss = SharedSession(share_browser=True)
    asyncio.run(tool._join_shared_browser(ss))

    assert ss.browser_context_id() == "ctx-existing"
    assert created == []  # no throwaway Context was made


def test_losing_the_session_creation_race_ends_the_extra_session(monkeypatch):
    """Two agents creating at once must not leak a second live session."""
    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
    ended = []
    monkeypatch.setattr(
        "dynamiq.nodes.tools.stagehand.end_browserbase_session",
        lambda api_key, project_id, session_id: ended.append(session_id),
    )
    tool = bare_stagehand(connection=MagicMock(browserbase_project_id="proj-1", browserbase_api_key="k"))

    ss = SharedSession(share_browser=True)
    ss.adopt_browser_session_id("sess-winner")

    tool._session_id = "sess-loser"
    tool._record_shared_browser_session(ss)

    assert ended == ["sess-loser"]
    assert tool._session_id == "sess-winner"  # falls in behind the winner
    assert ss.browser_session_id() == "sess-winner"


def test_shared_config_sets_context_keepalive_and_timeout(monkeypatch):
    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
    tool = bare_stagehand(connection=MagicMock(browserbase_project_id="proj-1"), shared_browser_session_timeout=1800)

    config = MagicMock(browserbase_session_create_params=None)
    tool._apply_shared_browser_config(config, "ctx-1")

    params = config.browserbase_session_create_params
    assert params["browser_settings"]["context"] == {"id": "ctx-1", "persist": True}
    assert params["project_id"] == "proj-1"
    # the session now spans the whole run, so it must survive its creator disconnecting
    assert params["keep_alive"] is True
    # "timeout", not "api_timeout": Stagehand camel-cases naively and the API rejects "apiTimeout"
    assert params["timeout"] == 1800
    assert "api_timeout" not in params


def test_shared_config_preserves_user_supplied_params(monkeypatch):
    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
    tool = bare_stagehand(connection=MagicMock(browserbase_project_id="proj-1"))

    config = MagicMock(
        browserbase_session_create_params={
            "project_id": "proj-explicit",
            "keep_alive": False,
            "browser_settings": {"block_ads": True},
        }
    )
    tool._apply_shared_browser_config(config, "ctx-1")

    params = config.browserbase_session_create_params
    assert params["project_id"] == "proj-explicit"
    assert params["keep_alive"] is False  # user's own value wins
    assert params["browser_settings"]["block_ads"] is True
    assert params["browser_settings"]["context"] == {"id": "ctx-1", "persist": True}


def test_close_detaches_instead_of_ending_a_shared_session(monkeypatch):
    """Stagehand's close() always ends the session — fatal when others are still using it."""
    monkeypatch.setattr(Stagehand, "_is_steel_browser_connection", lambda self: False)
    detached = []
    client_closed = []

    tool = bare_stagehand(
        _shares_browser_session=True,
        client=MagicMock(close=lambda: client_closed.append(1)),
        _steel_browser_session=None,
        _steel_client=None,
        _browserbase_client=None,
        _loop=None,
        _loop_thread=None,
    )
    monkeypatch.setattr(Stagehand, "_detach_shared_browser", lambda self: detached.append(1))
    monkeypatch.setattr(Stagehand, "close_loop", lambda self: None)

    tool.close()
    assert detached == [1]
    assert client_closed == []  # the session survives for the other agents


def test_owner_run_result_surfaces_shared_live_view(llm):
    agent = Agent(name="Owner", llm=llm, role="r", tools=[], share_browser_session_with_subagents=True)
    ss = SharedSession(share_browser=True)
    ss.set_browser_live_view_url("https://lv/owner")
    token = _shared_session.set(ss)
    try:
        result = {"content": "done"}
        agent._maybe_surface_live_view(result, shared_session_token=object())
        assert result["live_view_url"] == "https://lv/owner"

        result2 = {"content": "done"}
        agent._maybe_surface_live_view(result2, shared_session_token=None)  # non-owner
        assert "live_view_url" not in result2
    finally:
        _shared_session.reset(token)
