import threading
import time

from dynamiq.nodes.agents.shared_session import SharedSession, _current_agent_run


def test_share_browser_flag():
    assert SharedSession(share_browser=True).share_browser is True
    assert SharedSession().share_browser is False


def test_record_and_read_browser():
    ss = SharedSession(share_browser=True)
    assert ss.browser_session_id() is None
    ss.record_browser(session_id="sess-1", provider="browserbase", live_view_url="https://lv/1")
    assert ss.browser_session_id() == "sess-1"
    assert ss.browser_live_view_url() == "https://lv/1"
    # first-writer-wins: a second record does not overwrite
    ss.record_browser(session_id="sess-2", provider="browserbase")
    assert ss.browser_session_id() == "sess-1"


def test_lease_is_reentrant_for_same_run():
    ss = SharedSession(share_browser=True)
    ss.acquire_browser("runA")
    ss.acquire_browser("runA")  # reentrant no-op, must not block
    ss.release_browser("runA")


def test_lease_blocks_second_run_until_release():
    ss = SharedSession(share_browser=True)
    ss.acquire_browser("runA")
    acquired_b = threading.Event()

    def grab_b():
        ss.acquire_browser("runB")
        acquired_b.set()

    t = threading.Thread(target=grab_b)
    t.start()
    time.sleep(0.1)
    assert not acquired_b.is_set()  # B blocked while A holds it
    ss.release_browser("runA")
    assert acquired_b.wait(timeout=2.0)  # B proceeds after A releases
    ss.release_browser("runB")
    t.join(timeout=2.0)


def test_release_by_non_owner_is_noop():
    ss = SharedSession(share_browser=True)
    ss.acquire_browser("runA")
    ss.release_browser("runB")  # not the owner — must not release A's lease
    assert ss._lease_owner == "runA"
    ss.release_browser("runA")
    assert ss._lease_owner is None


def test_close_browser_invokes_session_close():
    ss = SharedSession(share_browser=True)
    calls = []
    ss.record_browser(session_id="s", provider="browserbase", close_callback=lambda: calls.append(1))
    ss.close_browser()
    assert calls == [1]


def test_current_agent_run_contextvar_defaults_none():
    assert _current_agent_run.get() is None
