import pytest

from dynamiq.connections import E2B
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.shared_session import _shared_session
from dynamiq.nodes.llms import OpenAI
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox


@pytest.fixture
def test_llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


def _sandboxed_agent(llm, **kwargs):
    return Agent(
        name=kwargs.pop("name", "Owner"),
        llm=llm,
        role="r",
        tools=[],
        sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared")),
        **kwargs,
    )


def test_owner_enters_session_when_flag_and_sandbox(test_llm):
    agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=True)
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    try:
        ss = _shared_session.get()
        assert ss is not None
        assert ss.share_sandbox is True
        assert ss.get_sandbox() is agent.sandbox_backend
    finally:
        agent._exit_shared_session(token)
    assert _shared_session.get() is None


def test_no_session_when_flag_off(test_llm):
    agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=False)
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    assert token is None
    assert _shared_session.get() is None


def test_no_session_when_no_sandbox(test_llm):
    agent = Agent(name="Owner", llm=test_llm, role="r", tools=[], share_sandbox_with_subagents=True)
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    assert token is None
    assert _shared_session.get() is None


def test_inherits_existing_session(test_llm):
    from dynamiq.nodes.agents.shared_session import SharedSession

    outer = SharedSession(sandbox=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="outer"), share_sandbox=True)
    outer_token = _shared_session.set(outer)
    try:
        agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=True)
        token = agent._maybe_enter_shared_session({"run_id": "run-2"})
        assert token is None  # inherited, did not create a new session
        assert _shared_session.get() is outer
    finally:
        _shared_session.reset(outer_token)


def test_cleanup_skips_shared_sandbox():
    from unittest.mock import MagicMock

    from dynamiq.nodes.tools.agent_tool import SubAgentTool

    agent = MagicMock()
    agent._sandbox_is_shared = True
    SubAgentTool.cleanup_factory_agent(agent)
    agent.sandbox_backend.close.assert_not_called()


def test_cleanup_kills_owned_sandbox():
    from unittest.mock import MagicMock

    from dynamiq.nodes.tools.agent_tool import SubAgentTool

    agent = MagicMock()
    agent._sandbox_is_shared = False
    SubAgentTool.cleanup_factory_agent(agent)
    agent.sandbox_backend.close.assert_called_once_with(kill=True)


def test_subagent_under_session_uses_view_backed_tools(test_llm):
    from dynamiq.nodes.agents.shared_session import SharedSession
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    try:
        sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])  # no own sandbox
        shell = next(t for t in sub.tools if isinstance(t, SandboxShellTool))
        assert sub._sandbox_is_shared is True
        assert shell.sandbox.sandbox_id == "sbx-shared"
        assert shell.sandbox.base_path.startswith("/home/user/work/researcher-")
    finally:
        _shared_session.reset(token)


def test_borrowed_subagent_sandbox_backend_is_the_shared_view(test_llm):
    """A subagent that borrows the shared sandbox must expose it as sandbox_backend,
    so execute()'s file upload / output collection / skills paths use the shared
    sandbox instead of falling back to an in-memory file store."""
    from dynamiq.nodes.agents.shared_session import SharedSession
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    try:
        sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])  # brings no sandbox
        assert sub._sandbox_is_shared is True
        assert sub.sandbox_backend is not None
        assert sub.sandbox_backend.sandbox_id == "sbx-shared"
        assert sub.sandbox_backend.base_path.startswith("/home/user/work/researcher-")
        # exactly the view feeding the shell tool — no split brain
        shell = next(t for t in sub.tools if isinstance(t, SandboxShellTool))
        assert sub.sandbox_backend is shell.sandbox
    finally:
        _shared_session.reset(token)


def test_subagent_with_own_sandbox_is_not_overridden(test_llm):
    """A subagent configured with its own sandbox keeps it — the shared session
    must not silently swap its tools onto the parent's view while other paths
    (cleanup, skills) still point at its own backend."""
    from dynamiq.nodes.agents.shared_session import SharedSession
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    try:
        sub = Agent(
            name="Sub",
            llm=test_llm,
            role="r",
            tools=[],
            sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-own")),
        )
        assert sub._sandbox_is_shared is False
        assert sub.sandbox_backend.sandbox_id == "sbx-own"
        shell = next(t for t in sub.tools if isinstance(t, SandboxShellTool))
        assert shell.sandbox.sandbox_id == "sbx-own"
    finally:
        _shared_session.reset(token)


def test_owner_without_active_session_uses_own_backend(test_llm):
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=True)  # no session set yet
    shell = next(t for t in agent.tools if isinstance(t, SandboxShellTool))
    assert agent._sandbox_is_shared is False
    assert shell.sandbox is agent.sandbox_backend


def test_two_subagents_share_one_sandbox_distinct_workdirs(test_llm):
    from dynamiq.nodes.agents.shared_session import _shared_session
    from dynamiq.nodes.tools.agent_tool import SubAgentTool
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    owner = _sandboxed_agent(test_llm, name="Owner", share_sandbox_with_subagents=True)
    token = owner._maybe_enter_shared_session({"run_id": "run-1"})
    try:
        sub1 = Agent(name="Researcher", llm=test_llm, role="r", tools=[])
        sub2 = Agent(name="Researcher", llm=test_llm, role="r", tools=[])

        shell1 = next(t for t in sub1.tools if isinstance(t, SandboxShellTool))
        shell2 = next(t for t in sub2.tools if isinstance(t, SandboxShellTool))

        # same underlying sandbox
        assert shell1.sandbox.sandbox_id == "sbx-shared"
        assert shell2.sandbox.sandbox_id == "sbx-shared"
        # isolated working directories
        assert shell1.sandbox.base_path != shell2.sandbox.base_path
        assert shell1.sandbox.base_path.startswith("/home/user/work/researcher-")
        assert shell2.sandbox.base_path.startswith("/home/user/work/researcher-")
        assert sub1._sandbox_is_shared and sub2._sandbox_is_shared

        # cleanup must not kill the shared sandbox
        SubAgentTool.cleanup_factory_agent(sub1)
        assert owner.sandbox_backend.current_sandbox_id == "sbx-shared"
    finally:
        owner._exit_shared_session(token)
    assert _shared_session.get() is None


def test_owner_session_defaults_to_all_scope(test_llm):
    from dynamiq.nodes.agents.shared_session import SandboxSharingScope

    agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=True)
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    try:
        assert _shared_session.get().sharing_scope == SandboxSharingScope.ALL
    finally:
        agent._exit_shared_session(token)


def test_owner_propagates_augment_scope_to_session(test_llm):
    from dynamiq.nodes.agents.shared_session import SandboxSharingScope

    agent = _sandboxed_agent(
        test_llm, share_sandbox_with_subagents=True, sandbox_sharing_scope=SandboxSharingScope.AUGMENT
    )
    token = agent._maybe_enter_shared_session({"run_id": "run-1"})
    try:
        assert _shared_session.get().sharing_scope == SandboxSharingScope.AUGMENT
    finally:
        agent._exit_shared_session(token)
