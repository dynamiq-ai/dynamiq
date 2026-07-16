import pytest

from dynamiq.sandboxes.base import Sandbox, SandboxLifecyclePolicy


def test_lifecycle_policy_values():
    assert SandboxLifecyclePolicy.PAUSE == "pause"
    assert SandboxLifecyclePolicy.KILL == "kill"
    assert SandboxLifecyclePolicy.DISCONNECT == "disconnect"
    # string-enum: usable as a plain string
    assert SandboxLifecyclePolicy("pause") is SandboxLifecyclePolicy.PAUSE


def test_base_sandbox_does_not_support_pause():
    # Sandbox is abstract; use a minimal concrete subclass that leaves pause unimplemented.
    class Bare(Sandbox):
        base_path: str = "/home/user"

        def get_tools(self, llm=None):
            return []

    b = Bare()
    assert b.supports_pause is False
    with pytest.raises(NotImplementedError):
        b.pause()
    with pytest.raises(NotImplementedError):
        b.resume()


from unittest.mock import MagicMock

from dynamiq.connections import E2B
from dynamiq.sandboxes.e2b import E2BSandbox


def _e2b(**kw):
    return E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-1", **kw)


def test_e2b_supports_pause():
    assert _e2b().supports_pause is True


def test_e2b_pause_calls_sdk_and_keeps_id_but_drops_handle():
    sb = _e2b()
    raw = MagicMock()
    sb._sandbox = raw  # _ensure_sandbox() returns this without connecting

    returned = sb.pause()

    raw.pause.assert_called_once()
    assert returned == "sbx-1"
    assert sb.sandbox_id == "sbx-1"   # id retained for resume
    assert sb._sandbox is None        # live handle dropped


def test_e2b_resume_reconnects_by_id(monkeypatch):
    sb = _e2b()
    reconnected = MagicMock()
    monkeypatch.setattr(E2BSandbox, "_reconnect_with_retry", lambda self: reconnected)

    returned = sb.resume()

    assert returned == "sbx-1"
    assert sb._sandbox is reconnected
