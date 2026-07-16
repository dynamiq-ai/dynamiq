import threading
import time

from dynamiq.nodes.agents.shared_session import SharedSession, _agent_run_chain, _current_agent_run


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
    ss.release_browser("runB")  # not the holder — must not pop A's lease
    assert ss._lease_stack == ["runA"]
    ss.release_browser("runA")
    assert ss._lease_stack == []


def test_reentrant_same_run_no_duplicate_push():
    ss = SharedSession(share_browser=True)
    ss.acquire_browser("runA")
    ss.acquire_browser("runA")  # same run, reentrant -> no duplicate push
    assert len(ss._lease_stack) == 1
    ss.release_browser("runA")  # a single release empties it
    assert ss._lease_stack == []


def test_nested_run_borrows_ancestor_lease():
    """A run whose ancestor currently holds the lease may borrow it (no block)."""
    ss = SharedSession(share_browser=True)
    ss.acquire_browser("owner", ("owner",))
    borrowed = threading.Event()

    def nested():
        # ancestor "owner" is on top of the stack -> borrowable, must NOT block
        ss.acquire_browser("sub", ("owner", "sub"))
        borrowed.set()

    t = threading.Thread(target=nested)
    t.start()
    assert borrowed.wait(timeout=2.0)  # nested proceeds immediately by borrowing
    assert ss._lease_stack == ["owner", "sub"]
    ss.release_browser("sub")
    ss.release_browser("owner")
    assert ss._lease_stack == []
    t.join(timeout=2.0)


def test_parallel_siblings_serialize_under_holding_owner():
    """While the owner holds the lease, two borrowing siblings still serialize."""
    ss = SharedSession(share_browser=True)
    ss.acquire_browser("owner", ("owner",))  # owner acquires and keeps holding

    a_acquired = threading.Event()
    b_acquired = threading.Event()
    release_a = threading.Event()

    def run_a():
        ss.acquire_browser("A", ("owner", "A"))  # borrows from owner
        a_acquired.set()
        release_a.wait(timeout=2.0)
        ss.release_browser("A")

    def run_b():
        ss.acquire_browser("B", ("owner", "B"))
        b_acquired.set()

    ta = threading.Thread(target=run_a)
    tb = threading.Thread(target=run_b)
    ta.start()
    assert a_acquired.wait(timeout=2.0)  # A borrows and proceeds
    tb.start()
    time.sleep(0.1)
    assert not b_acquired.is_set()  # B blocks while A holds the borrowed lease
    release_a.set()
    assert b_acquired.wait(timeout=2.0)  # B proceeds only after A releases
    ss.release_browser("B")
    ss.release_browser("owner")
    assert ss._lease_stack == []
    ta.join(timeout=2.0)
    tb.join(timeout=2.0)


def test_parallel_siblings_serialize_without_owner_driving():
    """Siblings sharing an ancestor that does NOT hold the lease still mutually exclude."""
    ss = SharedSession(share_browser=True)  # owner never acquires -> not on the stack

    a_acquired = threading.Event()
    b_acquired = threading.Event()
    release_a = threading.Event()

    def run_a():
        ss.acquire_browser("A", ("owner", "A"))
        a_acquired.set()
        release_a.wait(timeout=2.0)
        ss.release_browser("A")

    def run_b():
        ss.acquire_browser("B", ("owner", "B"))
        b_acquired.set()

    ta = threading.Thread(target=run_a)
    ta.start()
    assert a_acquired.wait(timeout=2.0)  # A acquires the empty lease
    tb = threading.Thread(target=run_b)
    tb.start()
    time.sleep(0.1)
    # B blocks: A holds it and "A" is not one of B's ancestors, even though they share "owner"
    assert not b_acquired.is_set()
    release_a.set()
    assert b_acquired.wait(timeout=2.0)  # B proceeds after A releases
    ss.release_browser("B")
    assert ss._lease_stack == []
    ta.join(timeout=2.0)
    tb.join(timeout=2.0)


def test_close_browser_invokes_session_close():
    ss = SharedSession(share_browser=True)
    calls = []
    ss.record_browser(session_id="s", provider="browserbase", close_callback=lambda: calls.append(1))
    ss.close_browser()
    assert calls == [1]


def test_current_agent_run_contextvar_defaults_none():
    assert _current_agent_run.get() is None


def test_agent_run_chain_contextvar_defaults_none():
    assert _agent_run_chain.get() is None
