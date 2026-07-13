import pytest

from dynamiq.connections import E2B
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.shared_session import _shared_session
from dynamiq.nodes.llms import OpenAI
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.storages.file.base import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore


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


def _file_store_agent(llm):
    return Agent(
        name="Sub",
        llm=llm,
        role="r",
        tools=[],
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore()),
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


def test_owner_warns_when_flag_on_but_no_sandbox(test_llm):
    """share_sandbox_with_subagents on but the owner has no sandbox is a silent no-op; it must
    warn so the misconfiguration is visible (reviewer feedback)."""
    from unittest.mock import patch

    agent = Agent(name="Owner", llm=test_llm, role="r", tools=[], share_sandbox_with_subagents=True)
    with patch("dynamiq.nodes.agents.base.logger") as mock_logger:
        token = agent._maybe_enter_shared_session({"run_id": "r"})
    assert token is None
    assert mock_logger.warning.called
    assert "no sandbox" in mock_logger.warning.call_args[0][0]


def test_owner_does_not_warn_when_sandbox_present(test_llm):
    from unittest.mock import patch

    agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=True)
    with patch("dynamiq.nodes.agents.base.logger") as mock_logger:
        token = agent._maybe_enter_shared_session({"run_id": "r"})
    try:
        assert token is not None
        mock_logger.warning.assert_not_called()
    finally:
        agent._exit_shared_session(token)


def test_owner_warns_and_skips_session_when_backend_lacks_view_support(test_llm):
    """Flag on but the owner's sandbox backend can't do views (e.g. Daytona): sharing can't happen,
    so warn instead of publishing a useless session (Bugbot: viewless backend breaks sharing
    silently)."""
    from unittest.mock import MagicMock, patch

    agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=True)
    viewless = MagicMock()
    viewless.supports_views = False
    agent.sandbox.backend = viewless  # swap in a view-incapable backend

    with patch("dynamiq.nodes.agents.base.logger") as mock_logger:
        token = agent._maybe_enter_shared_session({"run_id": "r"})

    assert token is None  # no session published
    assert mock_logger.warning.called
    assert "does not support shared views" in mock_logger.warning.call_args[0][0]


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
    agent._configured_sandbox_backend.return_value = None  # pure borrower: no own sandbox
    SubAgentTool.cleanup_factory_agent(agent)
    agent.sandbox_backend.close.assert_not_called()


def test_cleanup_kills_owned_sandbox():
    from unittest.mock import MagicMock

    from dynamiq.nodes.tools.agent_tool import SubAgentTool

    agent = MagicMock()
    agent._sandbox_is_shared = False
    SubAgentTool.cleanup_factory_agent(agent)
    agent.sandbox_backend.close.assert_called_once_with(kill=True)


def test_construction_under_session_does_not_resolve_shared_sandbox(test_llm):
    """P1.5: construction no longer borrows — resolution moved to run time (execute)."""
    from dynamiq.nodes.agents.shared_session import SharedSession
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    try:
        sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])  # no own sandbox
        assert sub._sandbox_is_shared is False
        assert not any(isinstance(t, SandboxShellTool) for t in sub.tools)
    finally:
        _shared_session.reset(token)


def test_borrowed_subagent_sandbox_backend_is_the_shared_view(test_llm):
    """After run-time borrow, the subagent exposes the shared view as sandbox_backend so
    execute()'s file upload / output collection / skills paths target the shared sandbox."""
    from dynamiq.nodes.agents.shared_session import SharedSession
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    try:
        sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])
        overlay = sub._maybe_borrow_shared_sandbox()  # what execute() does at run time
        assert sub.sandbox_backend is sub._shared_sandbox_view
        assert sub.sandbox_backend.sandbox_id == "sbx-shared"
        shell = next(t for t in overlay if isinstance(t, SandboxShellTool))
        assert sub.sandbox_backend is shell.sandbox
    finally:
        _shared_session.reset(token)


def test_borrowing_subagent_ingests_skills_into_shared_view(test_llm):
    """A borrowing subagent must ingest its sandbox-skills into the borrowed view at run time:
    init-time ingestion is skipped because it has no own sandbox then, but the SkillsTool it
    exposes reads SKILL.md files from the shared sandbox (Bugbot: borrowing subagents skip
    skills ingestion)."""
    from unittest.mock import patch

    from dynamiq.nodes.agents.shared_session import SharedSession

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    seen_backend = []
    try:
        sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])
        with patch.object(
            Agent,
            "_ensure_skills_ingested_for_sandbox",
            autospec=True,
            side_effect=lambda self: seen_backend.append(self.sandbox_backend),
        ) as ingest:
            sub._maybe_borrow_shared_sandbox()
        ingest.assert_called_once()
        # Ingestion must target the borrowed shared view, not the absent own backend.
        assert sub._shared_sandbox_view is not None
        assert seen_backend == [sub._shared_sandbox_view]
    finally:
        _shared_session.reset(token)


def test_tool_less_borrower_react_prompt_advertises_sandbox_tools(test_llm):
    """A subagent tool-less at init that borrows a shared sandbox at run time must get a React
    prompt advertising the sandbox tools + environment, not the frozen 'no tools' prompt built at
    construction (Bugbot: borrowed sandbox omits XML tool instructions)."""
    from dynamiq.nodes.agents.base import _shared_sandbox_tools
    from dynamiq.nodes.agents.shared_session import SharedSession

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    try:
        sub = Agent(name="Coder", llm=test_llm, role="r", tools=[])
        # Built tool-less at construction: the React 'tools' block is empty.
        assert sub.system_prompt_manager._prompt_blocks.get("tools") == ""

        overlay = sub._maybe_borrow_shared_sandbox()  # what execute() does at run time
        overlay_token = _shared_sandbox_tools.set(overlay)
        try:
            sub._sync_react_prompt_for_shared_sandbox()  # what execute()'s prompt setup does
            assert sub.system_prompt_manager._prompt_blocks.get("tools") != ""
            assert sub.system_prompt_manager._prompt_blocks.get("environment")
        finally:
            _shared_sandbox_tools.reset(overlay_token)
    finally:
        _shared_session.reset(token)


def test_borrow_prompt_sync_restores_structure_when_not_borrowing(test_llm):
    """A reused subagent that borrowed then runs standalone must not keep advertising the shared
    sandbox: the sync restores the tool-less structure and clears the environment block."""
    from dynamiq.nodes.agents.base import _shared_sandbox_tools
    from dynamiq.nodes.agents.shared_session import SharedSession

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    sub = Agent(name="Coder", llm=test_llm, role="r", tools=[])
    try:
        overlay = sub._maybe_borrow_shared_sandbox()
        overlay_token = _shared_sandbox_tools.set(overlay)
        try:
            sub._sync_react_prompt_for_shared_sandbox()
            assert sub.system_prompt_manager._prompt_blocks.get("environment")
        finally:
            _shared_sandbox_tools.reset(overlay_token)
        sub._release_shared_sandbox_view()  # end of the borrow run
    finally:
        _shared_session.reset(token)

    # Standalone run: no session/overlay -> structure restored, environment cleared.
    sub._sync_react_prompt_for_shared_sandbox()
    assert sub.system_prompt_manager._prompt_blocks.get("tools") == ""
    assert not sub.system_prompt_manager._prompt_blocks.get("environment")


def test_borrowed_sandbox_tools_in_function_calling_schemas(test_llm):
    """In FUNCTION_CALLING mode, a borrowing subagent's sandbox tools must appear in the effective
    inference schemas — the FC-schema half of the same finding."""
    from dynamiq.nodes.agents.base import _shared_sandbox_tools
    from dynamiq.nodes.agents.shared_session import SharedSession
    from dynamiq.nodes.types import InferenceMode

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    try:
        sub = Agent(name="Coder", llm=test_llm, role="r", tools=[], inference_mode=InferenceMode.FUNCTION_CALLING)
        overlay = sub._maybe_borrow_shared_sandbox()
        overlay_token = _shared_sandbox_tools.set(overlay)
        try:
            effective = sub._effective_inference_schemas()
            assert effective[0]  # borrowed sandbox tools present in the FC schema
            assert effective[0] != sub._tools  # not the frozen tool-less init schema
        finally:
            _shared_sandbox_tools.reset(overlay_token)
    finally:
        _shared_session.reset(token)


def test_release_shared_sandbox_view_disconnects_without_kill(test_llm):
    from unittest.mock import MagicMock

    sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])
    view = MagicMock()
    sub._shared_sandbox_view = view
    sub._release_shared_sandbox_view()
    view.close.assert_called_once_with(kill=False)
    assert sub._shared_sandbox_view is None


def test_owner_without_active_session_uses_own_backend(test_llm):
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    agent = _sandboxed_agent(test_llm, share_sandbox_with_subagents=True)  # no session set yet
    shell = next(t for t in agent.tools if isinstance(t, SandboxShellTool))
    assert agent._sandbox_is_shared is False
    assert shell.sandbox is agent.sandbox_backend


def test_reused_subagent_keeps_stable_workdir_across_calls(test_llm):
    """A reused initialized subagent must land in the SAME workdir on every borrow so its
    relative-path files persist across parent calls (Bugbot: subagent workdir resets each call)."""
    from dynamiq.nodes.agents.shared_session import SharedSession
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    try:
        sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])

        ov1 = sub._maybe_borrow_shared_sandbox()
        wd1 = next(t for t in ov1 if isinstance(t, SandboxShellTool)).sandbox.base_path
        sub._release_shared_sandbox_view()  # end of call 1, as execute()'s finally does

        ov2 = sub._maybe_borrow_shared_sandbox()
        wd2 = next(t for t in ov2 if isinstance(t, SandboxShellTool)).sandbox.base_path

        assert wd1 == wd2  # same instance -> same workdir across calls
    finally:
        _shared_session.reset(token)


def test_two_subagents_share_one_sandbox_distinct_workdirs(test_llm):
    from dynamiq.nodes.tools.agent_tool import SubAgentTool
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    owner = _sandboxed_agent(test_llm, name="Owner", share_sandbox_with_subagents=True)
    token = owner._maybe_enter_shared_session({"run_id": "run-1"})
    try:
        sub1 = Agent(name="Researcher", llm=test_llm, role="r", tools=[])
        sub2 = Agent(name="Researcher", llm=test_llm, role="r", tools=[])
        ov1 = sub1._maybe_borrow_shared_sandbox()
        ov2 = sub2._maybe_borrow_shared_sandbox()

        shell1 = next(t for t in ov1 if isinstance(t, SandboxShellTool))
        shell2 = next(t for t in ov2 if isinstance(t, SandboxShellTool))

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


def test_runtime_tools_includes_overlay(test_llm):
    from dynamiq.nodes.agents.base import _shared_sandbox_tools
    from dynamiq.nodes.agents.shared_session import SharedSession
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    stoken = _shared_session.set(session)
    try:
        sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])
        overlay = sub._maybe_borrow_shared_sandbox()
        otoken = _shared_sandbox_tools.set(overlay)
        try:
            assert any(isinstance(t, SandboxShellTool) for t in sub._runtime_tools)
            assert not any(isinstance(t, SandboxShellTool) for t in sub.tools)
        finally:
            _shared_sandbox_tools.reset(otoken)
    finally:
        _shared_session.reset(stoken)


def test_runtime_tools_hides_own_sandbox_tools_when_overridden(test_llm):
    from dynamiq.nodes.agents.base import _shared_sandbox_tools
    from dynamiq.nodes.agents.shared_session import SandboxSharingScope, SharedSession
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(
        sandbox=shared, share_sandbox=True, owner_run_id="owner", sharing_scope=SandboxSharingScope.ALL
    )
    stoken = _shared_session.set(session)
    try:
        sub = Agent(
            name="Sub",
            llm=test_llm,
            role="r",
            tools=[],
            sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-own")),
        )
        overlay = sub._maybe_borrow_shared_sandbox()
        otoken = _shared_sandbox_tools.set(overlay)
        try:
            shells = [t for t in sub._runtime_tools if isinstance(t, SandboxShellTool)]
            assert len(shells) == 1
            assert shells[0].sandbox.sandbox_id == "sbx-shared"
        finally:
            _shared_sandbox_tools.reset(otoken)
    finally:
        _shared_session.reset(stoken)


def test_runtime_tools_composes_overlay_with_ltm(test_llm):
    from dynamiq.nodes.agents.base import _run_extra_tools, _shared_sandbox_tools
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])
    fake_ltm = SandboxShellTool(sandbox=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="ltm"))
    fake_sandbox = SandboxShellTool(sandbox=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="ov"))
    ltoken = _run_extra_tools.set([fake_ltm])
    otoken = _shared_sandbox_tools.set([fake_sandbox])
    try:
        rt = sub._runtime_tools
        assert fake_ltm in rt and fake_sandbox in rt
    finally:
        _shared_sandbox_tools.reset(otoken)
        _run_extra_tools.reset(ltoken)


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


def test_agent_records_its_own_sandbox_tool_ids(test_llm):
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    agent = _sandboxed_agent(test_llm)  # has its own sandbox
    own_shell = next(t for t in agent.tools if isinstance(t, SandboxShellTool))
    assert agent._own_sandbox_tool_ids  # non-empty
    assert own_shell.id in agent._own_sandbox_tool_ids


def test_agent_without_sandbox_records_no_own_sandbox_tool_ids(test_llm):
    agent = Agent(name="Plain", llm=test_llm, role="r", tools=[])
    assert agent._own_sandbox_tool_ids == set()


def test_shared_sandbox_tools_contextvar_defaults_none():
    from dynamiq.nodes.agents.base import _shared_sandbox_tools

    assert _shared_sandbox_tools.get() is None


def test_borrow_returns_none_without_session(test_llm):
    sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])
    assert sub._maybe_borrow_shared_sandbox() is None
    assert sub._sandbox_is_shared is False


def test_borrow_builds_view_tools_for_plain_subagent(test_llm):
    from dynamiq.nodes.agents.shared_session import SharedSession
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    try:
        sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])  # no own sandbox
        overlay = sub._maybe_borrow_shared_sandbox()
        assert overlay is not None
        shell = next(t for t in overlay if isinstance(t, SandboxShellTool))
        assert shell.sandbox.sandbox_id == "sbx-shared"
        assert shell.sandbox.base_path.startswith("/home/user/work/researcher-")
        assert shell.is_optimized_for_agents is True
        assert sub._sandbox_is_shared is True
        assert sub._shared_sandbox_view is shell.sandbox
    finally:
        _shared_session.reset(token)


def test_owner_does_not_borrow_its_own_shared_sandbox(test_llm):
    owner = _sandboxed_agent(test_llm, name="Owner", share_sandbox_with_subagents=True)
    token = owner._maybe_enter_shared_session({"run_id": "run-1"})
    try:
        assert owner._maybe_borrow_shared_sandbox() is None
        assert owner._sandbox_is_shared is False
    finally:
        owner._exit_shared_session(token)


def test_augment_scope_keeps_own_sandbox(test_llm):
    from dynamiq.nodes.agents.shared_session import SandboxSharingScope, SharedSession

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(
        sandbox=shared, share_sandbox=True, owner_run_id="owner", sharing_scope=SandboxSharingScope.AUGMENT
    )
    token = _shared_session.set(session)
    try:
        sub = Agent(
            name="Sub",
            llm=test_llm,
            role="r",
            tools=[],
            sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-own")),
        )
        assert sub._maybe_borrow_shared_sandbox() is None
        assert sub._sandbox_is_shared is False
    finally:
        _shared_session.reset(token)


def test_all_scope_overrides_own_sandbox(test_llm):
    from dynamiq.nodes.agents.shared_session import SandboxSharingScope, SharedSession
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(
        sandbox=shared, share_sandbox=True, owner_run_id="owner", sharing_scope=SandboxSharingScope.ALL
    )
    token = _shared_session.set(session)
    try:
        sub = Agent(
            name="Sub",
            llm=test_llm,
            role="r",
            tools=[],
            sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-own")),
        )
        overlay = sub._maybe_borrow_shared_sandbox()
        assert overlay is not None
        shell = next(t for t in overlay if isinstance(t, SandboxShellTool))
        assert shell.sandbox.sandbox_id == "sbx-shared"
        assert sub._sandbox_is_shared is True
    finally:
        _shared_session.reset(token)


def test_file_store_subagent_never_borrows(test_llm):
    from dynamiq.nodes.agents.shared_session import SharedSession

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner")
    token = _shared_session.set(session)
    try:
        sub = _file_store_agent(test_llm)
        assert sub.file_store_backend is not None
        assert sub._maybe_borrow_shared_sandbox() is None
        assert sub._sandbox_is_shared is False
    finally:
        _shared_session.reset(token)


def test_runtime_tools_empty_overlay_keeps_own_sandbox_tools(test_llm):
    """A non-None but EMPTY sandbox overlay must not strip the agent's own sandbox tools,
    otherwise the agent would be left with no sandbox tools at all (Bugbot: empty overlay
    hides sandbox tools)."""
    from dynamiq.nodes.agents.base import _shared_sandbox_tools

    agent = _sandboxed_agent(test_llm)  # brings its own sandbox tools
    assert agent._own_sandbox_tool_ids  # sanity: it owns sandbox tools
    otoken = _shared_sandbox_tools.set([])  # active-but-empty overlay
    try:
        assert [t.id for t in agent._runtime_tools] == [t.id for t in agent.tools]
    finally:
        _shared_sandbox_tools.reset(otoken)


def test_override_does_not_kill_own_sandbox_at_borrow_time(test_llm):
    """scope=ALL override routes the subagent onto the shared view but must NOT tear down its own
    dedicated sandbox at borrow time: this path is shared by reused initialized subagents, which
    must keep their own sandbox for later standalone use (Bugbot: own sandbox killed on reuse).
    Teardown of a *factory* agent's orphaned backend happens in cleanup_factory_agent instead."""
    from unittest.mock import MagicMock

    from dynamiq.nodes.agents.shared_session import SandboxSharingScope, SharedSession

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(
        sandbox=shared, share_sandbox=True, owner_run_id="owner", sharing_scope=SandboxSharingScope.ALL
    )
    token = _shared_session.set(session)
    try:
        sub = Agent(
            name="Sub",
            llm=test_llm,
            role="r",
            tools=[],
            sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-own")),
        )
        own_spy = MagicMock()
        sub.sandbox.backend = own_spy

        overlay = sub._maybe_borrow_shared_sandbox()

        assert overlay is not None
        own_spy.close.assert_not_called()  # own sandbox left intact for reuse
        assert sub._sandbox_is_shared is True
        assert sub._shared_sandbox_view.sandbox_id == "sbx-shared"
    finally:
        _shared_session.reset(token)


def test_cleanup_factory_agent_kills_overridden_own_sandbox(test_llm):
    """A factory agent whose own sandbox was overridden (scope=ALL) must have that orphaned
    dedicated backend torn down when the disposable factory agent is cleaned up."""
    from unittest.mock import MagicMock

    from dynamiq.nodes.tools.agent_tool import SubAgentTool

    sub = _sandboxed_agent(test_llm)  # has its own dedicated sandbox
    own_spy = MagicMock()
    sub.sandbox.backend = own_spy
    # Simulate the post-borrow state: routed onto a shared view, own distinct backend recorded as
    # the orphan to tear down (view already released in execute()).
    sub._sandbox_is_shared = True
    sub._shared_sandbox_view = None
    sub._overridden_own_backend = own_spy

    SubAgentTool.cleanup_factory_agent(sub)

    own_spy.close.assert_called_once_with(kill=True)


def test_cleanup_skips_teardown_when_no_overridden_backend(test_llm):
    """When a borrower recorded no orphaned own backend (its own backend WAS the shared object, or
    it had none), cleanup must tear down nothing (Bugbot: cleanup kills shared owner sandbox)."""
    from unittest.mock import MagicMock

    from dynamiq.nodes.tools.agent_tool import SubAgentTool

    sub = _sandboxed_agent(test_llm)
    shared_view = MagicMock()
    sub._sandbox_is_shared = True
    sub._shared_sandbox_view = shared_view
    sub._overridden_own_backend = None  # own was the shared object (or none) — no orphan to kill

    SubAgentTool.cleanup_factory_agent(sub)

    shared_view.close.assert_not_called()


def test_borrow_failure_rolls_back_and_does_not_latch_or_kill_own(test_llm):
    """If tool-building fails mid-borrow, the agent must be left untouched: no latched view,
    not marked shared, and its dedicated sandbox NOT torn down (Bugbot: shared sandbox setup
    lacks cleanup). Build-then-commit ordering guarantees this."""
    from unittest.mock import MagicMock

    from dynamiq.nodes.agents.shared_session import SandboxSharingScope, SharedSession

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(
        sandbox=shared, share_sandbox=True, owner_run_id="owner", sharing_scope=SandboxSharingScope.ALL
    )
    bad_view = MagicMock()
    bad_view.get_tools.side_effect = RuntimeError("boom")
    session.sandbox_view_for = MagicMock(return_value=bad_view)

    token = _shared_session.set(session)
    try:
        sub = Agent(
            name="Sub",
            llm=test_llm,
            role="r",
            tools=[],
            sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-own")),
        )
        own_spy = MagicMock()
        sub.sandbox.backend = own_spy

        with pytest.raises(RuntimeError, match="boom"):
            sub._maybe_borrow_shared_sandbox()

        assert sub._shared_sandbox_view is None
        assert sub._sandbox_is_shared is False
        own_spy.close.assert_not_called()
    finally:
        _shared_session.reset(token)


def test_augment_scope_does_not_tear_down_own_sandbox(test_llm):
    """Under AUGMENT, a subagent keeps its own sandbox — it must NOT be torn down."""
    from unittest.mock import MagicMock

    from dynamiq.nodes.agents.shared_session import SandboxSharingScope, SharedSession

    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session = SharedSession(
        sandbox=shared, share_sandbox=True, owner_run_id="owner", sharing_scope=SandboxSharingScope.AUGMENT
    )
    token = _shared_session.set(session)
    try:
        sub = Agent(
            name="Sub",
            llm=test_llm,
            role="r",
            tools=[],
            sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-own")),
        )
        own_spy = MagicMock()
        sub.sandbox.backend = own_spy

        assert sub._maybe_borrow_shared_sandbox() is None
        own_spy.close.assert_not_called()
        assert sub._sandbox_is_shared is False
    finally:
        _shared_session.reset(token)


def test_owner_identified_by_id_not_backend_object(test_llm):
    """A subagent configured with the SAME backend object as the owner is still a borrower, not
    the owner, so it must get its own isolated /work view (Bugbot: same backend skips workdir
    isolation)."""
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    owner = _sandboxed_agent(test_llm, name="Owner", share_sandbox_with_subagents=True)
    token = owner._maybe_enter_shared_session({"run_id": "run-1"})
    try:
        # Configure the subagent with the exact same Sandbox object the session holds.
        sub = Agent(
            name="Researcher",
            llm=test_llm,
            role="r",
            tools=[],
            sandbox=SandboxConfig(enabled=True, backend=owner.sandbox_backend),
        )
        assert sub._configured_sandbox_backend() is owner.sandbox_backend  # same object reference

        overlay = sub._maybe_borrow_shared_sandbox()

        assert overlay is not None  # borrowed, not mistaken for the owner
        shell = next(t for t in overlay if isinstance(t, SandboxShellTool))
        assert shell.sandbox.base_path.startswith("/home/user/work/researcher-")
        assert sub._sandbox_is_shared is True
        assert sub._overridden_own_backend is None  # own IS the shared sandbox — not an orphan
    finally:
        owner._exit_shared_session(token)


def test_release_restores_outer_view_for_nested_execute(test_llm):
    """A nested execute() on the same borrowing instance must restore the OUTER run's view in its
    finally, not clear it, so the outer run keeps its effective sandbox_backend (Bugbot: nested
    execute clears borrowed view)."""
    from unittest.mock import MagicMock

    sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])
    outer_view, inner_view = MagicMock(name="outer"), MagicMock(name="inner")

    sub._shared_sandbox_view = outer_view  # outer run is mid-flight with its borrowed view
    prev = sub._shared_sandbox_view  # what the nested run captures before borrowing
    sub._shared_sandbox_view = inner_view  # nested run borrows a fresh view

    sub._release_shared_sandbox_view(restore_to=prev)  # nested finally

    assert sub._shared_sandbox_view is outer_view  # outer view restored, not cleared
    assert sub.sandbox_backend is outer_view
    inner_view.close.assert_called_once_with(kill=False)  # only the nested view disconnected
    outer_view.close.assert_not_called()  # outer view left connected for the outer run


def test_clear_todos_clears_borrowed_overlay_todo_tool(test_llm):
    """Todo cleanup must clear TodoWriteTool from the shared-sandbox overlay, not only self.tools —
    otherwise a borrowed workdir keeps stale todos across runs (Bugbot: todo cleanup skips borrowed
    sandbox tools)."""
    from unittest.mock import MagicMock

    from dynamiq.nodes.agents.base import _shared_sandbox_tools
    from dynamiq.nodes.tools.todo_tools import TodoWriteTool

    sub = Agent(name="Researcher", llm=test_llm, role="r", tools=[])  # no own sandbox/todo tool
    overlay_todo = MagicMock(spec=TodoWriteTool)
    otoken = _shared_sandbox_tools.set([overlay_todo])  # borrowed sandbox tools, as execute() sets
    try:
        sub._clear_todos_file()
    finally:
        _shared_sandbox_tools.reset(otoken)

    overlay_todo.clear.assert_called_once()
