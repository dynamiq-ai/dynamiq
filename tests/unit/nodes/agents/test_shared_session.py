from dynamiq.nodes.agents.shared_session import SharedSession, _shared_session, slugify


class FakeSandbox:
    def __init__(self, sandbox_id=None, base_path="/home/user"):
        self.sandbox_id = sandbox_id
        self.base_path = base_path
        self.ensure_started_calls = 0
        self.created_views = []

    @property
    def current_sandbox_id(self):
        return self.sandbox_id

    def ensure_started(self):
        self.ensure_started_calls += 1
        self.sandbox_id = "sbx-new"
        return self.sandbox_id

    def create_view(self, base_path, sandbox_id=None):
        view = FakeSandbox(sandbox_id=sandbox_id or self.current_sandbox_id, base_path=base_path)
        self.created_views.append(view)
        return view


def test_slugify_replaces_unsafe_chars():
    assert slugify("Researcher #2/x") == "Researcher__2_x"
    assert slugify("   ") == "agent"


def test_share_sandbox_false_without_sandbox():
    ss = SharedSession(sandbox=None, share_sandbox=True, owner_run_id="o")
    assert ss.share_sandbox is False
    assert ss.get_sandbox() is None
    assert ss.sandbox_view_for("k") is None


def test_view_uses_existing_sandbox_id_without_materializing():
    sb = FakeSandbox(sandbox_id="sbx-1", base_path="/home/user")
    ss = SharedSession(sandbox=sb, share_sandbox=True, owner_run_id="o")
    view = ss.sandbox_view_for("Researcher-ab12")
    assert sb.ensure_started_calls == 0
    assert view.sandbox_id == "sbx-1"
    assert view.base_path == "/home/user/work/Researcher-ab12"


def test_view_materializes_when_no_sandbox_id():
    sb = FakeSandbox(sandbox_id=None, base_path="/home/user")
    ss = SharedSession(sandbox=sb, share_sandbox=True, owner_run_id="o")
    view = ss.sandbox_view_for("writer-cd34")
    assert sb.ensure_started_calls == 1
    assert view.sandbox_id == "sbx-new"
    assert view.base_path == "/home/user/work/writer-cd34"


def test_contextvar_defaults_to_none():
    assert _shared_session.get() is None
