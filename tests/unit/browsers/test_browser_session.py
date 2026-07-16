from dynamiq.browsers.base import BrowserSession


def test_browser_session_holds_identity():
    bs = BrowserSession(provider="browserbase", session_id="sess-1", live_view_url="https://lv/1")
    assert bs.provider == "browserbase"
    assert bs.session_id == "sess-1"
    assert bs.live_view_url == "https://lv/1"


def test_close_invokes_callback_once():
    calls = []
    bs = BrowserSession(provider="browserbase", session_id="sess-1", close_callback=lambda: calls.append(1))
    bs.close()
    bs.close()  # idempotent
    assert calls == [1]


def test_close_without_callback_is_noop():
    BrowserSession(provider="browserbase", session_id="s").close()  # must not raise
