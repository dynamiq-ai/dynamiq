from unittest.mock import MagicMock

import pytest

from dynamiq.connections import E2B
from dynamiq.sandboxes.base import Sandbox, SandboxLifecyclePolicy
from dynamiq.sandboxes.e2b import E2BSandbox


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


def test_e2b_auto_pause_passes_lifecycle_on_create(monkeypatch):
    sb = E2BSandbox(connection=E2B(api_key="t"), auto_pause=True)  # no sandbox_id → create path
    captured = {}

    class _FakeSDK:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            m = MagicMock()
            m.sandbox_id = "sbx-new"
            return m

    monkeypatch.setattr(E2BSandbox, "_sdk_class", property(lambda self: _FakeSDK))
    sb._create_with_retry()
    assert captured.get("lifecycle") == {"on_timeout": "pause"}


def test_e2b_no_auto_pause_omits_lifecycle(monkeypatch):
    sb = E2BSandbox(connection=E2B(api_key="t"))  # auto_pause defaults False
    captured = {}

    class _FakeSDK:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            m = MagicMock()
            m.sandbox_id = "sbx-new"
            return m

    monkeypatch.setattr(E2BSandbox, "_sdk_class", property(lambda self: _FakeSDK))
    sb._create_with_retry()
    assert "lifecycle" not in captured
