from unittest.mock import MagicMock, patch

from dynamiq.connections import E2B
from dynamiq.sandboxes.e2b import E2BSandbox


def test_a_sandbox_in_use_is_renewed_before_its_timeout_runs_out():
    # Runs can outlast the timeout (1.7 h runs are seen in production); E2B then kills the sandbox mid-run.
    sandbox = E2BSandbox(connection=E2B(api_key="test"), timeout=3600)
    remote = MagicMock()
    with (
        patch.object(E2BSandbox, "_create_with_retry", return_value=remote),
        patch("dynamiq.sandboxes.e2b.time.monotonic", side_effect=[0.0, 1000.0, 2000.0, 2100.0]),
    ):
        sandbox._ensure_sandbox()  # created at t=0
        sandbox._ensure_sandbox()  # t=1000: well within the timeout
        sandbox._ensure_sandbox()  # t=2000: past half of it
        sandbox._ensure_sandbox()  # t=2100: just renewed

    remote.set_timeout.assert_called_once_with(3600)
