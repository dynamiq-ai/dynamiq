import threading
import time

import pytest

from dynamiq.nodes.agents.shared_session import SharedSession, _current_agent_run


def test_share_browser_flag():
    assert SharedSession(share_browser=True).share_browser is True
    assert SharedSession().share_browser is False


def test_session_id_is_adopted_first_writer_wins():
    """Agents share ONE live session — that is what makes their cookies mutually visible."""
    ss = SharedSession(share_browser=True)
    assert ss.browser_session_id() is None
    assert ss.adopt_browser_session_id("sess-1") == "sess-1"
    # a second agent that raced into creating its own gets the established one back
    assert ss.adopt_browser_session_id("sess-2") == "sess-1"
    assert ss.browser_session_id() == "sess-1"


def test_context_id_is_adopted_first_writer_wins():
    """The Context is the CROSS-RUN axis: it carries state to a later run, not between agents."""
    ss = SharedSession(share_browser=True)
    assert ss.browser_context_id() is None
    assert ss.adopt_browser_context_id("ctx-1") == "ctx-1"
    assert ss.adopt_browser_context_id("ctx-2") == "ctx-1"


def test_live_view_url_is_recorded_once():
    ss = SharedSession(share_browser=True)
    assert ss.browser_live_view_url() is None
    ss.set_browser_live_view_url("https://lv/1")
    assert ss.browser_live_view_url() == "https://lv/1"
    # one shared session means one live view; later tools must not clobber it, nor erase it
    ss.set_browser_live_view_url("https://lv/2")
    ss.set_browser_live_view_url(None)
    assert ss.browser_live_view_url() == "https://lv/1"


def test_session_is_ended_once_and_only_by_the_owner():
    ss = SharedSession(share_browser=True)
    ended = []
    ss.register_browser_end(lambda: ended.append(1))
    ss.register_browser_end(lambda: ended.append(2))  # a later agent must not replace the ender

    ss.end_browser_session()
    ss.end_browser_session()  # idempotent: a second teardown must not double-end
    assert ended == [1]


def test_end_is_a_noop_when_nothing_was_registered():
    SharedSession(share_browser=True).end_browser_session()  # must not raise


def test_failing_end_is_swallowed():
    """A failed end must not break the owner's teardown."""
    ss = SharedSession(share_browser=True)

    def boom():
        raise RuntimeError("session already gone")

    ss.register_browser_end(boom)
    ss.end_browser_session()


def test_page_control_is_idempotent_within_a_turn():
    """Control is taken per browser CALL but released once per TURN, so it must not be counted."""
    ss = SharedSession(share_browser=True)
    ss.acquire_page_control("runA")
    ss.acquire_page_control("runA")
    ss.acquire_page_control("runA")
    assert ss._page_control_key == "runA"

    ss.release_page_control("runA")
    assert ss._page_control_key is None
    ss.acquire_page_control("runB", timeout=1.0)  # would time out if still held
    ss.release_page_control("runB")


def test_concurrent_calls_from_one_agent_do_not_block_each_other():
    ss = SharedSession(share_browser=True)
    done = []
    barrier = threading.Barrier(2)

    def call():
        barrier.wait(timeout=2.0)
        ss.acquire_page_control("runA", timeout=2.0)
        done.append(1)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)

    assert done == [1, 1]
    assert ss._page_control_key == "runA"


def test_second_agent_waits_for_the_page():
    ss = SharedSession(share_browser=True)
    ss.acquire_page_control("runA")
    got_b = threading.Event()

    def grab_b():
        ss.acquire_page_control("runB", timeout=3.0)
        got_b.set()

    t = threading.Thread(target=grab_b)
    t.start()
    try:
        time.sleep(0.1)
        assert not got_b.is_set()  # B must not drive the page while A is mid-sequence
        ss.release_page_control("runA")
        assert got_b.wait(timeout=3.0)
    finally:
        t.join(timeout=3.0)
        ss.release_page_control("runB")


def test_release_by_non_holder_is_noop():
    """Every agent releases in its finally, including ones that never browsed."""
    ss = SharedSession(share_browser=True)
    ss.acquire_page_control("runA")
    ss.release_page_control("runB")
    assert ss._page_control_key == "runA"
    ss.release_page_control("runA")
    assert ss._page_control_key is None


def test_delegating_hands_the_page_over_without_closing_anything():
    """The browse-then-delegate case: releasing costs nothing, so the subagent just proceeds."""
    ss = SharedSession(share_browser=True)
    ended = []
    ss.register_browser_end(lambda: ended.append(1))

    ss.acquire_page_control("owner")  # owner browses
    ss.release_page_control("owner")  # owner delegates
    ss.acquire_page_control("sub", timeout=1.0)  # subagent takes over immediately
    ss.release_page_control("sub")  # subagent's turn ends
    ss.acquire_page_control("owner", timeout=1.0)  # owner resumes on the SAME live session

    assert ended == []  # nothing was closed at any point; no state was lost


def test_multiple_waiters_are_served_one_at_a_time():
    ss = SharedSession(share_browser=True)
    order = []
    ss.acquire_page_control("runA")

    def wait_and_take(key):
        ss.acquire_page_control(key, timeout=3.0)
        order.append(key)

    threads = [threading.Thread(target=wait_and_take, args=(k,)) for k in ("runB", "runC")]
    for t in threads:
        t.start()
    time.sleep(0.1)
    assert order == []

    ss.release_page_control("runA")
    time.sleep(0.1)
    assert len(order) == 1  # one waiter got it; the other is still queued

    ss.release_page_control(order[0])
    for t in threads:
        t.join(timeout=3.0)
    assert sorted(order) == ["runB", "runC"]
    ss.release_page_control(order[1])


def test_acquire_times_out_instead_of_hanging():
    ss = SharedSession(share_browser=True)
    ss.acquire_page_control("runA")
    try:
        with pytest.raises(TimeoutError, match="one may drive it at a time"):
            ss.acquire_page_control("runB", timeout=0.1)
    finally:
        ss.release_page_control("runA")


def test_current_agent_run_contextvar_defaults_none():
    assert _current_agent_run.get() is None
