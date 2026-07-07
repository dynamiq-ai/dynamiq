"""Live E2B smoke test for the shared-sandbox session (requires E2B_API_KEY).

The unit suite mocks the sandbox, so the SDK behaviors that sharing depends on
are never actually exercised. This test drives the real ``SharedSession`` and
its per-agent views against one live E2B sandbox and asserts each assumption:

  1. multiple concurrent connections reconnect to ONE ``sandbox_id``;
  2. ``run_command_shell`` honors each view's per-agent working dir (cwd);
  3. the per-agent workdir is created (``mkdir -p``) on reconnect;
  4. the filesystem is shared across views, but ``list_files`` stays
     workdir-isolated;
  5. tearing down one view (``close(kill=False)``) does NOT kill the shared
     sandbox that other agents are still using.

Skipped automatically when ``E2B_API_KEY`` is absent (e.g. CI without creds).
"""

import os
import uuid

import pytest

from dynamiq.connections import E2B as E2BConnection
from dynamiq.nodes.agents.shared_session import SharedSession
from dynamiq.sandboxes.e2b import E2BSandbox


@pytest.fixture(scope="module")
def owner_sandbox():
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY is not set; skipping live E2B test.")
    sandbox = E2BSandbox(connection=E2BConnection())
    sandbox.ensure_started()
    try:
        yield sandbox
    finally:
        sandbox.close(kill=True)


@pytest.mark.integration
def test_shared_session_two_views_live(owner_sandbox):
    session = SharedSession(sandbox=owner_sandbox, share_sandbox=True, owner_run_id="live")
    # E2B supports views, so sharing must be enabled (not degraded to no-sharing).
    assert session.share_sandbox is True

    marker = uuid.uuid4().hex[:8]
    view1 = session.sandbox_view_for(f"researcher-1-{marker}")
    view2 = session.sandbox_view_for(f"researcher-2-{marker}")
    assert view1 is not None and view2 is not None

    owner_id = owner_sandbox.current_sandbox_id
    assert owner_id is not None

    # (1) both views reconnect to the SAME underlying sandbox.
    assert view1.current_sandbox_id == owner_id
    assert view2.current_sandbox_id == owner_id

    # (2) isolated per-agent working directories under <base>/work/.
    work_root = f"{owner_sandbox.base_path.rstrip('/')}/work/"
    assert view1.base_path != view2.base_path
    assert view1.base_path.startswith(work_root)
    assert view2.base_path.startswith(work_root)

    # (3) run_command_shell runs in each view's own workdir; the dir was created
    #     (mkdir -p) on reconnect, or `pwd` here would fail.
    pwd1 = view1.run_command_shell("pwd")
    pwd2 = view2.run_command_shell("pwd")
    assert pwd1.exit_code == 0 and pwd1.stdout.strip() == view1.base_path
    assert pwd2.exit_code == 0 and pwd2.stdout.strip() == view2.base_path

    # each view writes a file into its own workdir (one via the files API, one
    # via a relative-path shell redirect that resolves against its cwd).
    view1.upload_file("a.txt", b"from-researcher-1")
    wrote2 = view2.run_command_shell("echo from-researcher-2 > b.txt")
    assert wrote2.exit_code == 0

    file1 = f"{view1.base_path}/a.txt"
    file2 = f"{view2.base_path}/b.txt"

    # (4a) workdir isolation: list_files() (defaults to the view's base_path)
    #      surfaces only that view's own file, not the sibling view's.
    files1 = view1.list_files()
    files2 = view2.list_files()
    assert file1 in files1 and file2 not in files1
    assert file2 in files2 and file1 not in files2

    # (4b) shared filesystem: each view can reach the other's file by abs path.
    assert view1.exists(file2) is True
    assert view2.exists(file1) is True
    cross = view1.run_command_shell(f"cat {file2}")
    assert cross.exit_code == 0 and "from-researcher-2" in cross.stdout
    assert view1.retrieve("a.txt") == b"from-researcher-1"

    # (5) closing one view keeps the shared sandbox alive for everyone else.
    view1.close(kill=False)
    alive = owner_sandbox.run_command_shell("echo alive")
    assert alive.exit_code == 0 and "alive" in alive.stdout
    assert owner_sandbox.current_sandbox_id == owner_id
