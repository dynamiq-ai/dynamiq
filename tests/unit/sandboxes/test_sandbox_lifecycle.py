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
