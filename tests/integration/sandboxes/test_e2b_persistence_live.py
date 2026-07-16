"""Live E2B persistence round-trip (requires E2B_API_KEY).

Proves the P2 assumption the unit suite mocks: pausing a sandbox and reconnecting
to its sandbox_id resumes the SAME sandbox with its filesystem intact. Skipped
without E2B_API_KEY.
"""

import os
import uuid

import pytest

from dynamiq.connections import E2B as E2BConnection
from dynamiq.sandboxes.e2b import E2BSandbox


@pytest.mark.integration
def test_e2b_pause_then_resume_preserves_filesystem():
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY is not set; skipping live E2B persistence test.")

    marker = uuid.uuid4().hex[:12]
    sandbox = E2BSandbox(connection=E2BConnection())
    sandbox_id = None
    try:
        sandbox.ensure_started()
        sandbox_id = sandbox.current_sandbox_id
        assert sandbox_id is not None

        # write a file, then pause
        sandbox.upload_file("persist.txt", marker.encode())
        returned_id = sandbox.pause()
        assert returned_id == sandbox_id

        # a fresh backend bound to the same id resumes the paused sandbox
        resumed = E2BSandbox(connection=E2BConnection(), sandbox_id=sandbox_id)
        assert resumed.resume() == sandbox_id
        assert resumed.exists("persist.txt") is True
        assert resumed.retrieve("persist.txt") == marker.encode()
    finally:
        if sandbox_id is not None:
            E2BSandbox(connection=E2BConnection(), sandbox_id=sandbox_id).close(kill=True)
