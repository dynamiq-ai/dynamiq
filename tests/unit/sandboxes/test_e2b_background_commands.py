"""Background commands in an E2B sandbox."""

import threading
import time
from unittest.mock import MagicMock, patch

from dynamiq.connections import E2B
from dynamiq.sandboxes.e2b import E2BSandbox


def test_an_e2b_background_command_is_not_cut_off_after_a_minute():
    # The SDK's 60 s default bounds the connection a background process lives on (a dev server, a watcher).
    sandbox = E2BSandbox(connection=E2B(api_key="test-key"))
    remote = MagicMock()

    with patch.object(sandbox, "_ensure_sandbox", return_value=remote):
        result = sandbox.run_command_shell("npm run dev", run_in_background_enabled=True)

    assert result.background
    assert remote.commands.run.call_args.kwargs["timeout"] == 0


def test_a_background_command_the_sandbox_never_starts_does_not_block_the_caller():
    # With timeout=0 the SDK puts no limit on opening the command's stream: a hung envd never answers.
    sandbox = E2BSandbox(connection=E2B(api_key="test-key"))
    envd_answers = threading.Event()
    remote = MagicMock()
    remote.commands.run.side_effect = lambda *args, **kwargs: envd_answers.wait()

    try:
        with (
            patch.object(sandbox, "_ensure_sandbox", return_value=remote),
            patch.object(E2BSandbox, "BACKGROUND_START_TIMEOUT", 0.2),
        ):
            started_at = time.monotonic()
            result = sandbox.run_command_shell("npm run dev", run_in_background_enabled=True)
            waited = time.monotonic() - started_at
    finally:
        envd_answers.set()

    assert not result.background
    assert "did not report the command started within 0.2 seconds" in result.error
    assert waited < 5


def test_a_background_command_that_fails_to_start_is_reported():
    sandbox = E2BSandbox(connection=E2B(api_key="test-key"))
    remote = MagicMock()
    remote.commands.run.side_effect = RuntimeError("envd unreachable")

    with patch.object(sandbox, "_ensure_sandbox", return_value=remote):
        result = sandbox.run_command_shell("npm run dev", run_in_background_enabled=True)

    assert not result.background
    assert result.error == "envd unreachable"
