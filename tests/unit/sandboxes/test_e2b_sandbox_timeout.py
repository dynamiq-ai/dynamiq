from unittest.mock import Mock, patch

import pytest

from dynamiq.connections import E2B
from dynamiq.sandboxes.e2b import E2BSandbox

TIMEOUT = 1800


class FakeE2B:
    """One E2B sandbox on a fake clock. Connecting only lengthens its timeout, and set_timeout can shorten it."""

    def __init__(self):
        self.now = 0.0
        self.expires_at = 0.0
        self.connects = 0
        self.failing_connects = 0
        self.on_long_command = None
        self.commands = Mock(run=self.run)

    def connect(self, timeout: int, **kwargs) -> "FakeE2B":
        self.connects += 1
        if self.failing_connects:
            self.failing_connects -= 1
            raise RuntimeError("E2B API unavailable")
        self.expires_at = max(self.expires_at, self.now + timeout)
        return self

    def set_timeout(self, timeout: int, **kwargs) -> None:
        self.expires_at = self.now + timeout

    def run(self, command: str, background: bool = False, **kwargs) -> Mock:
        assert self.now <= self.expires_at, f"the sandbox timed out before {command!r}"
        if command.startswith("sleep ") and not background:
            self.now += 1
            if self.on_long_command:
                self.on_long_command()
            self.now += int(command.split()[1]) - 1
            assert self.now <= self.expires_at, f"the sandbox timed out during {command!r}"
        return Mock(stdout="", stderr="", exit_code=0)


@pytest.fixture
def e2b():
    fake = FakeE2B()
    with (
        patch("dynamiq.sandboxes.e2b.time", Mock(monotonic=lambda: fake.now)),
        patch.object(E2BSandbox, "_reconnect_with_retry", lambda self: fake.connect(timeout=self.timeout)),
    ):
        yield fake


def e2b_sandbox() -> E2BSandbox:
    return E2BSandbox(connection=E2B(api_key="test-key"), sandbox_id="sbx-1", timeout=TIMEOUT)


def test_a_sandbox_in_use_stays_up_and_times_out_once_left_alone(e2b):
    sandbox = e2b_sandbox()

    for _ in range(120):
        sandbox.run_command_shell("ls")
        e2b.now += 59
    sandbox.run_command_shell("npm run dev", timeout=7200, run_in_background_enabled=True)

    assert e2b.connects <= e2b.now / 60 + 1
    assert TIMEOUT - 60 <= e2b.expires_at - e2b.now <= TIMEOUT


def test_a_long_command_outlasts_the_timeout_while_another_instance_uses_the_sandbox(e2b):
    owner, other = e2b_sandbox(), e2b_sandbox()
    e2b.on_long_command = lambda: other.run_command_shell("ls")

    owner.run_command_shell("sleep 3590", timeout=3600)

    assert e2b.expires_at - e2b.now >= TIMEOUT - 60


def test_a_command_keeps_its_sandbox_up_for_at_most_an_hour(e2b):
    sandbox = e2b_sandbox()

    sandbox.run_command_shell("pip install x", timeout=86400)

    assert e2b.expires_at - e2b.now <= E2BSandbox.MAX_COMMAND_TIMEOUT + E2BSandbox.TIMEOUT_REFRESH_INTERVAL


def test_a_failed_renewal_is_retried(e2b):
    sandbox = e2b_sandbox()
    sandbox.run_command_shell("ls")
    e2b.failing_connects = 1

    for _ in range(40):
        e2b.now += 59
        sandbox.run_command_shell("ls")

    assert e2b.expires_at - e2b.now >= TIMEOUT - 60
