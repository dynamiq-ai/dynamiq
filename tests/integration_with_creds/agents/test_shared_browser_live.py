"""Live integration tests for shared browser sessions between an agent and its subagents.

Run against REAL Browserbase with REAL agents/subagents (requires OPENAI_API_KEY,
BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID). Expensive, slow and LLM-nondeterministic;
the whole module is skipped without all three credentials.

The design under test (Model D + Context — see docs/design/shared-browser-lease-fix.md §5.2)
has two deliberately separate axes:

  * INTRA-RUN — every agent in one run drives ONE live Browserbase session, so cookies,
    logins and the current page are visible to each other IMMEDIATELY, with nothing closing.
  * CROSS-RUN — that session loads a persistent Browserbase Context and writes it back when it
    ENDS (owner teardown), carrying state to a LATER run (e.g. the next turn of a conversation).

Each test states which property it proves. The deterministic signals are:
  * SESSION LOG — ``RecordingStagehand`` records the session id each tool actually drove (the
    tool clears ``_session_id`` on close). "Shared" == one session id across all agents.
  * PAGE CONTENT — asserted on each browser tool's own returned text (via ``CONTENT_LOG``), NOT on
    the owner agent's LLM-composed summary, whose wording is nondeterministic. Page-content tests
    use example.com ("Example Domain"), which is far more reliable than httpbin under load.
  * COOKIES — set on one agent, read back on another via httpbin; proves state crosses. httpbin is
    the only external dependency that can flake here, so cookie tests SKIP (not fail) on a 5xx.
  * SESSION STATUS — queried from Browserbase after the run; proves the session is ended exactly
    once, at the owner's teardown, and (crucially) that a subagent finishing did NOT end it.

Live page-position crossing is anchored by a deterministic, no-LLM test
(``test_page_position_crosses_the_detach_boundary_deterministic``); the agent-driven page tests
exercise the same property through real delegation but lean on that anchor for rigor.

The one guarantee NOT covered here — that using the browser and delegating browser work in the
same parallel batch fails with a clean error rather than hanging — is deterministic and lives in
the unit tests (test_shared_browser_session.py::test_acquire_times_out_instead_of_hanging); an LLM
cannot be reliably coerced into emitting that exact batch shape.
"""

import io
import os
import threading
import time
from collections import defaultdict
from urllib.parse import quote

import pytest

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Stagehand as StagehandConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import Stagehand
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.tools.stagehand import end_browserbase_session
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus

EXAMPLE_URL = "https://example.com/"
COOKIE_READ_URL = "https://httpbin.org/cookies"

# A deliberately rigid role for "read the current page" subagents. Left to their own devices, agents
# spend loops on observe/act (which trip on shadow DOM) and hit the loop limit; a single forced
# extract is what the mechanism actually needs and keeps the LLM out of its own way.
SINGLE_EXTRACT_ROLE = (
    "You have ONE job, done in a SINGLE tool call. Call your Stagehand browser tool exactly once "
    "with action_type 'extract' and instruction 'Return all visible text on the page.'. Do NOT use "
    "goto, observe, act or any other action_type, and do NOT call the tool more than once. Then give "
    "the text you extracted as your final answer."
)


def _cookie_set_url(name: str, value: str, *, persistent: bool) -> str:
    """A URL that sets a cookie. Persistent (Max-Age) cookies survive to a later run via the
    Context; session cookies (no expiry) only live within a run, so cross-run tests need Max-Age."""
    cookie = f"{name}={value}; Path=/"
    if persistent:
        cookie += "; Max-Age=86400"
    return f"https://httpbin.org/response-headers?Set-Cookie={quote(cookie)}"


# tool name -> [{"session_id": ..., "context_id": ...}] for every session that tool drove
SESSION_LOG: dict[str, list[dict]] = defaultdict(list)
# tool name -> [content, ...] for every action a tool returned. State-crossing is asserted on THIS,
# the raw page content the browser saw, not on the owner agent's LLM-composed summary (whose
# phrasing is nondeterministic and may drop the exact word we look for).
CONTENT_LOG: dict[str, list[str]] = defaultdict(list)


class RecordingStagehand(Stagehand):
    """Stagehand that records the session id it drove, the Context that session carries, and the
    content each of its actions returned."""

    async def _init_client(
        self,
        files: list[io.BytesIO],
        shared_context_id: str | None = None,
        create_shared_session: bool = False,
    ):
        await super()._init_client(files, shared_context_id, create_shared_session)
        record = {"session_id": self._session_id, "context_id": shared_context_id}
        if record not in SESSION_LOG[self.name]:
            SESSION_LOG[self.name].append(record)

    async def execute_async(self, input_data, config=None, **kwargs):
        result = await super().execute_async(input_data, config, **kwargs)
        CONTENT_LOG[self.name].append(str(result.get("content", "")))
        return result


@pytest.fixture(autouse=True)
def _session_log_and_leak_guard():
    """Clear the logs around each test and, as a safety net, end any session it left running so a
    failing test cannot leak a paid Browserbase session."""
    SESSION_LOG.clear()
    CONTENT_LOG.clear()
    yield
    api_key = os.getenv("BROWSERBASE_API_KEY")
    project_id = os.getenv("BROWSERBASE_PROJECT_ID")
    if api_key and project_id:
        for session_id in _all_logged_session_ids():
            try:
                end_browserbase_session(api_key, project_id, session_id)
            except Exception:
                pass  # already ended (the normal case) or unreachable — best effort
    SESSION_LOG.clear()
    CONTENT_LOG.clear()


def _require_creds():
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("BROWSERBASE_API_KEY") and os.getenv("BROWSERBASE_PROJECT_ID")):
        pytest.skip("OPENAI_API_KEY, BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID are required for this test.")


def _openai_llm() -> OpenAI:
    return OpenAI(
        connection=OpenAIConnection(api_key=os.getenv("OPENAI_API_KEY")),
        model="gpt-4o",
        temperature=0.0,
        max_tokens=800,
        is_postponed_component_init=True,
    )


def _stagehand_tool(name: str, **kwargs) -> RecordingStagehand:
    return RecordingStagehand(
        name=name,
        connection=StagehandConnection(model_api_key=os.getenv("OPENAI_API_KEY")),
        model_name="gpt-4o",
        is_return_live_view_url_enabled=True,
        is_postponed_component_init=True,
        **kwargs,
    )


def _browsing_agent(name: str, role: str, tool: Stagehand, **kwargs) -> Agent:
    return Agent(
        name=name,
        role=role,
        description=f"{name}: browses using its own Stagehand tool.",
        llm=_openai_llm(),
        tools=[tool],
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        max_loops=6,  # headroom so a stray observe/act does not exhaust the agent before it extracts
        is_postponed_component_init=True,
        **kwargs,
    )


def _owner(role: str, tools: list, **kwargs) -> Agent:
    return Agent(
        id="owner",
        name="owner",
        role=role,
        description="Owner agent that shares one browser session across its subagents.",
        llm=_openai_llm(),
        tools=tools,
        share_browser_session_with_subagents=True,
        inference_mode=InferenceMode.XML,
        max_loops=10,
        is_postponed_component_init=True,
        **kwargs,
    )


def _run_workflow(owner: Agent, prompt: str, *, timeout: float = 240.0):
    """Run the owner in a workflow on a worker thread, so a re-introduced deadlock fails fast
    (thread does not finish in time) instead of hanging the whole suite."""
    box: dict = {}

    def _target(cm):
        wf = Workflow(flow=Flow(connection_manager=cm, init_components=True, nodes=[owner]))
        box["result"] = wf.run(input_data={"input": prompt}, config=RunnableConfig())

    with get_connection_manager() as cm:
        worker = threading.Thread(target=_target, args=(cm,), daemon=True)
        worker.start()
        worker.join(timeout=timeout)
        assert (
            not worker.is_alive()
        ), f"workflow did not finish within {timeout:.0f}s — likely a shared-browser deadlock"
    return box["result"]


def _all_logged_session_ids() -> set[str]:
    return {r["session_id"] for records in SESSION_LOG.values() for r in records if r["session_id"]}


def _tool_saw(tool_name: str, needle: str) -> bool:
    """Did any action of this tool return content containing ``needle``? Asserted on the raw page
    content the browser saw, so it does not depend on how the owner agent phrased its summary."""
    return any(needle.lower() in content.lower() for content in CONTENT_LOG[tool_name])


def _skip_if_httpbin_unavailable(*tool_names: str) -> None:
    """Skip (not fail) when the cookie endpoint (httpbin.org) is 5xx-ing. Cookie tests depend on it;
    its downtime is an external flake, not a sharing regression, and must not read as a real failure."""
    blob = " ".join(c for name in tool_names for c in CONTENT_LOG[name]).lower()
    for marker in ("503", "service temporarily unavailable", "502 bad gateway", "504 gateway"):
        if marker in blob:
            pytest.skip("httpbin.org is unavailable (5xx) — external dependency, not a sharing failure")


def _assert_one_shared_session(*tool_names: str) -> str:
    """Every named tool drove the SAME single live session — the core intra-run contract."""
    for name in tool_names:
        assert SESSION_LOG[name], f"{name} never opened a browser session"
    session_ids = {r["session_id"] for name in tool_names for r in SESSION_LOG[name]}
    assert all(session_ids), "a session was opened without an id"
    assert len(session_ids) == 1, f"agents did not share ONE live session: {session_ids}"
    return session_ids.pop()


def _assert_shared_context(*tool_names: str) -> str:
    context_ids = {r["context_id"] for name in tool_names for r in SESSION_LOG[name] if r["context_id"]}
    assert len(context_ids) == 1, f"agents did not share ONE browser Context: {context_ids}"
    return context_ids.pop()


def _session_status(session_id: str) -> str:
    from browserbase import Browserbase

    bb = Browserbase(api_key=os.getenv("BROWSERBASE_API_KEY"))
    return bb.sessions.retrieve(session_id).status


def _assert_session_ended(session_id: str) -> None:
    """After the owner's teardown the session must no longer be RUNNING. REQUEST_RELEASE settles to
    COMPLETED asynchronously, so poll briefly rather than asserting once."""
    deadline = time.monotonic() + 30.0
    status = _session_status(session_id)
    while status == "RUNNING" and time.monotonic() < deadline:
        time.sleep(2.0)
        status = _session_status(session_id)
    assert status != "RUNNING", f"shared session {session_id} was left RUNNING (owner did not end it)"


def _owner_output(result, owner: Agent) -> dict:
    assert result.status == RunnableStatus.SUCCESS, f"run failed: {getattr(result, 'error', None)}"
    return result.output[owner.id]["output"]


# ---------------------------------------------------------------------------------------------
# 1. Two subagents, coordinator owner: one session, state crosses, session ended once at the end.
# ---------------------------------------------------------------------------------------------
@pytest.mark.integration
def test_two_subagents_share_session_and_state_crosses():
    """The canonical shape. Owner only delegates; a setter subagent stores a cookie, then a
    separate checker subagent reads it back — on the SAME live session, without either closing.

    Proves at once: (a) one shared session, (b) cookies cross agents live, (c) the checker's
    session was still alive after the setter's teardown (detach, not end — finding #6), and
    (d) the session is ended after the owner's turn.
    """
    _require_creds()
    setter_tool = _stagehand_tool("setter-browser")
    checker_tool = _stagehand_tool("checker-browser")

    setter = _browsing_agent(
        "setter",
        "Your ONLY job: use your Stagehand browser tool to navigate to the EXACT URL you are given, "
        "then confirm it loaded. Do nothing else.",
        setter_tool,
    )
    checker = _browsing_agent(
        "checker",
        "Your ONLY job: use your Stagehand browser tool to navigate to the EXACT URL you are given "
        "and report the page's text content verbatim. Report exactly what you see.",
        checker_tool,
    )
    owner = _owner(
        "You coordinate two sub-agents that share ONE browser profile, one at a time. You have NO "
        "browser tool of your own. Always invoke a tool as {'input': '<task>'}. First delegate to "
        "'setter'; ONLY AFTER it answers, delegate to 'checker'. Then report what the checker saw.",
        [
            SubAgentTool(name="setter", description="Open a URL. Pass {'input': '<task>'}.", agent=setter),
            SubAgentTool(name="checker", description="Read a page. Pass {'input': '<task>'}.", agent=checker),
        ],
        parallel_tool_calls_enabled=False,
    )

    set_url = _cookie_set_url("dynamiq_cross", "agent-A", persistent=False)
    result = _run_workflow(
        owner,
        f"First have the setter open {set_url}. After it confirms, have the checker open "
        f"{COOKIE_READ_URL} and report the cookies the page lists.",
    )

    output = _owner_output(result, owner)
    session_id = _assert_one_shared_session("setter-browser", "checker-browser")
    _assert_shared_context("setter-browser", "checker-browser")
    assert output.get("live_view_url"), "owner run result did not surface a live_view_url"
    _skip_if_httpbin_unavailable("setter-browser", "checker-browser")
    assert _tool_saw(
        "checker-browser", "dynamiq_cross"
    ), "the checker did not see the cookie the setter set — state did not cross the shared session"
    _assert_session_ended(session_id)


# ---------------------------------------------------------------------------------------------
# 2. Live page position crosses (the property that distinguishes this from close-based handoff).
# ---------------------------------------------------------------------------------------------
@pytest.mark.integration
def test_live_page_position_crosses_between_subagents():
    """A navigator subagent leaves the page on a distinctive URL; a reader subagent reports the
    CURRENT page WITHOUT navigating. Under a shared live session it sees the navigator's page —
    something close-based handoff (fresh blank session per agent) could never do.
    """
    _require_creds()
    navigator_tool = _stagehand_tool("navigator-browser")
    reader_tool = _stagehand_tool("reader-browser")

    navigator = _browsing_agent(
        "navigator",
        "Your ONLY job: use your Stagehand browser tool to navigate to the EXACT URL you are given, "
        "then confirm it loaded. Do nothing else.",
        navigator_tool,
    )
    reader = _browsing_agent("reader", SINGLE_EXTRACT_ROLE, reader_tool)
    owner = _owner(
        "You coordinate two sub-agents sharing ONE browser, one at a time. You have NO browser tool "
        "of your own. Always invoke a tool as {'input': '<task>'}. First delegate to 'navigator'; "
        "ONLY AFTER it answers, delegate to 'reader'. Then report what the reader saw.",
        [
            SubAgentTool(name="navigator", description="Open a URL. Pass {'input': '<task>'}.", agent=navigator),
            SubAgentTool(name="reader", description="Read the current page. Pass {'input': '<task>'}.", agent=reader),
        ],
        parallel_tool_calls_enabled=False,
    )

    result = _run_workflow(
        owner,
        f"First have the navigator open {EXAMPLE_URL}. After it confirms, have the reader report "
        "the current page's content WITHOUT navigating anywhere.",
    )

    _owner_output(result, owner)  # asserts the run succeeded
    _assert_one_shared_session("navigator-browser", "reader-browser")
    # example.com shows the distinctive text "Example Domain"; the reader, which never navigated,
    # can only see it if it inherited the navigator's live page.
    assert _tool_saw(
        "reader-browser", "example domain"
    ), "the reader did not see the page the navigator left open — live page position did not cross"


@pytest.mark.integration
def test_page_position_crosses_the_detach_boundary_deterministic():
    """The deterministic anchor for the live-page-position claim, with no LLM in the loop.

    Two real Stagehand tools driven directly through ``SharedSession``: tool A opens a distinctive
    page and then tears down exactly as a subagent would (``_shares_browser_session`` -> ``close``
    -> detach, without ending the session); tool B attaches to the same session and extracts the
    CURRENT page. B seeing A's page proves ``_detach_shared_browser`` leaves the remote page intact
    across the handoff — the mechanism the two agent-driven page tests exercise, minus their LLM
    nondeterminism.
    """
    _require_creds()
    from dynamiq.nodes.agents.shared_session import SharedSession, _current_agent_run, _shared_session

    ss = SharedSession(share_browser=True)
    session_token = _shared_session.set(ss)
    a = _stagehand_tool("detach-A")
    b = _stagehand_tool("detach-B")
    session_id = None
    try:
        run_a = _current_agent_run.set("run-A")
        a.run(
            input_data={"action_type": "goto", "url": EXAMPLE_URL, "brief": "A opens the page"},
            config=RunnableConfig(),
        )
        session_id = ss.browser_session_id()
        ss.release_page_control("run-A")
        _current_agent_run.reset(run_a)
        a.close()  # subagent-style teardown: detach (a._shares_browser_session is True), do NOT end

        run_b = _current_agent_run.set("run-B")
        result_b = b.run(
            input_data={
                "action_type": "extract",
                "instruction": "Return the full text of the page body verbatim.",
                "brief": "B reads the current page",
            },
            config=RunnableConfig(),
        )
        ss.release_page_control("run-B")
        _current_agent_run.reset(run_b)

        assert b._session_id == session_id, "B did not attach to A's session"
        assert (
            "example domain" in str(result_b.output.get("content", "")).lower()
        ), "B did not see the page A left open — page position did not survive A's detach"
    finally:
        _shared_session.reset(session_token)
        for tool in (b, a):
            try:
                tool._shares_browser_session = False
                tool.close()
            except Exception:
                pass
        if session_id and os.getenv("BROWSERBASE_API_KEY"):
            try:
                end_browserbase_session(
                    os.getenv("BROWSERBASE_API_KEY"), os.getenv("BROWSERBASE_PROJECT_ID"), session_id
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------------------------
# 3. Owner browses, THEN delegates (the old deadlock trap) — must not hang, must share.
# ---------------------------------------------------------------------------------------------
@pytest.mark.integration
def test_owner_browses_then_delegates_without_deadlock():
    """Owner uses its OWN browser first, then delegates to a browsing subagent. Releasing page
    control around the delegate call hands the live page over, so the subagent proceeds
    immediately instead of blocking until the page-control timeout. The worker-thread timeout in
    ``_run_workflow`` turns any regression into a fast failure rather than a hang.
    """
    _require_creds()
    owner_tool = _stagehand_tool("owner-browser")
    reader_tool = _stagehand_tool("reader-browser")

    reader = _browsing_agent("reader", SINGLE_EXTRACT_ROLE, reader_tool)
    owner = _owner(
        "You share ONE browser with your sub-agent, one at a time. You have your OWN Stagehand tool "
        "AND a 'reader' sub-agent. Always invoke a tool as {'input': '<task>'}. FIRST use your own "
        "tool to navigate to the URL you are given. THEN delegate to 'reader' to report the current "
        "page. Then give a short answer.",
        [
            owner_tool,
            SubAgentTool(name="reader", description="Read the current page. Pass {'input': '<task>'}.", agent=reader),
        ],
        parallel_tool_calls_enabled=False,
    )

    result = _run_workflow(
        owner,
        f"First navigate to {EXAMPLE_URL} using your own browser tool. After it loads, delegate to "
        "the reader to report the current page's content.",
    )

    _owner_output(result, owner)
    session_id = _assert_one_shared_session("owner-browser", "reader-browser")
    assert _tool_saw(
        "reader-browser", "example domain"
    ), "the reader did not see the page the owner opened — handoff after browsing failed"
    _assert_session_ended(session_id)


# ---------------------------------------------------------------------------------------------
# 4. Subagent browses, THEN owner browses — handoff in the other direction.
# ---------------------------------------------------------------------------------------------
@pytest.mark.integration
def test_subagent_hands_back_then_owner_browses():
    """A navigator subagent browses first; after it returns, the owner's own tool picks up the
    SAME live session and reads the page the subagent left open."""
    _require_creds()
    owner_tool = _stagehand_tool("owner-browser")
    sub_tool = _stagehand_tool("navigator-browser")

    navigator = _browsing_agent(
        "navigator",
        "Your ONLY job: use your Stagehand browser tool to navigate to the EXACT URL you are given, "
        "then confirm it loaded. Do nothing else.",
        sub_tool,
    )
    owner = _owner(
        "You share ONE browser with your sub-agent, one at a time. Do EXACTLY two steps, in order. "
        "STEP 1: delegate to the 'navigator' sub-agent to open the URL you are given, invoking it as "
        "{'input': '<task>'}; wait for its answer. STEP 2 (REQUIRED — you must not skip it or answer "
        "before doing it): call your OWN browser tool named 'owner-browser' exactly once with "
        "action_type 'extract' and instruction 'Return all visible text on the page.' — do not "
        "navigate. Then give the extracted text as your final answer. Never use your own tool before "
        "the navigator has answered.",
        [
            SubAgentTool(name="navigator", description="Open a URL. Pass {'input': '<task>'}.", agent=navigator),
            owner_tool,
        ],
        parallel_tool_calls_enabled=False,
    )

    result = _run_workflow(
        owner,
        f"STEP 1: delegate to the navigator to open {EXAMPLE_URL}. STEP 2: after it confirms, use "
        "your own 'owner-browser' tool to extract and report the current page's content.",
    )

    _owner_output(result, owner)
    session_id = _assert_one_shared_session("owner-browser", "navigator-browser")
    assert _tool_saw(
        "owner-browser", "example domain"
    ), "the owner did not see the page the navigator left — reverse handoff failed"
    _assert_session_ended(session_id)


# ---------------------------------------------------------------------------------------------
# 5. A non-browsing delegate must not disturb the owner's page (nothing closes unless needed).
# ---------------------------------------------------------------------------------------------
@pytest.mark.integration
def test_non_browsing_delegate_preserves_the_page():
    """Owner browses to a distinctive page, delegates to a pure-LLM subagent (no browser), then
    reads the page again. The page must be exactly where it was — releasing page control around a
    delegate closes nothing, so a non-browsing detour costs nothing.
    """
    _require_creds()
    owner_tool = _stagehand_tool("owner-browser")

    summarizer = Agent(
        name="summarizer",
        role="You are a text assistant with NO browser. Given a short instruction, reply with a "
        "one-sentence acknowledgement. Do not ask for tools.",
        description="A pure-LLM helper with no browser tool.",
        llm=_openai_llm(),
        tools=[],
        inference_mode=InferenceMode.XML,
        max_loops=3,
        is_postponed_component_init=True,
    )
    owner = _owner(
        "You have your OWN Stagehand browser tool named 'owner-browser' and a non-browser "
        "'summarizer' sub-agent. Do EXACTLY three steps in order. STEP 1: use 'owner-browser' with "
        "action_type 'goto' to open the URL you are given. STEP 2: delegate one short note to "
        "'summarizer' as {'input': '<task>'}. STEP 3 (REQUIRED): use 'owner-browser' again with "
        "action_type 'extract' and instruction 'Return all visible text on the page.' — do not "
        "navigate again. Then give the extracted text as your final answer.",
        [
            owner_tool,
            SubAgentTool(
                name="summarizer",
                description="Delegate a short note. Pass {'input': '<task>'}.",
                agent=summarizer,
            ),
        ],
        parallel_tool_calls_enabled=False,
    )

    result = _run_workflow(
        owner,
        f"First navigate to {EXAMPLE_URL}. Then ask the summarizer to note 'page opened'. Then "
        "report the current page's content.",
    )

    _owner_output(result, owner)
    # Only the owner browsed, so exactly one tool and one session; the page survived the detour.
    session_id = _assert_one_shared_session("owner-browser")
    assert _tool_saw("owner-browser", "example domain"), "the owner's page was lost across a non-browsing delegate"
    _assert_session_ended(session_id)


# ---------------------------------------------------------------------------------------------
# 6. Parallel browsing subagents serialize on the shared page instead of colliding.
# ---------------------------------------------------------------------------------------------
@pytest.mark.integration
def test_parallel_browsing_subagents_share_one_session():
    """Owner with parallel tool calls enabled dispatches two browsing subagents. Whether the LLM
    runs them together or in sequence, page control serializes them onto ONE session with no
    collision and no deadlock. (That they genuinely overlapped is not asserted — it is not
    reliably forceable — but one-session-and-success is the invariant that must always hold.)
    """
    _require_creds()
    a_tool = _stagehand_tool("browser-A")
    b_tool = _stagehand_tool("browser-B")

    agent_a = _browsing_agent(
        "worker_a",
        "Your ONLY job: navigate your Stagehand browser to the EXACT URL you are given and confirm.",
        a_tool,
    )
    agent_b = _browsing_agent(
        "worker_b",
        "Your ONLY job: navigate your Stagehand browser to the EXACT URL you are given and confirm.",
        b_tool,
    )
    owner = _owner(
        "You coordinate two browsing sub-agents that share ONE browser. Always invoke a tool as "
        "{'input': '<task>'}. Delegate a navigation task to BOTH 'worker_a' and 'worker_b'. Then "
        "give a short final answer.",
        [
            SubAgentTool(name="worker_a", description="Navigate. Pass {'input': '<task>'}.", agent=agent_a),
            SubAgentTool(name="worker_b", description="Navigate. Pass {'input': '<task>'}.", agent=agent_b),
        ],
        parallel_tool_calls_enabled=True,
    )

    result = _run_workflow(
        owner,
        f"Have worker_a open {EXAMPLE_URL} and worker_b open https://example.org/. Delegate to both.",
    )

    _owner_output(result, owner)
    session_id = _assert_one_shared_session("browser-A", "browser-B")
    _assert_session_ended(session_id)


# ---------------------------------------------------------------------------------------------
# 7. Factory-mode subagents (fresh instance per call, clone path) still share one session.
# ---------------------------------------------------------------------------------------------
@pytest.mark.integration
def test_factory_mode_subagents_share_one_session():
    """Subagents spawned from a factory (a fresh Agent per call, which exercises the clone path)
    must still attach to the run's one shared session — each clone gets its own run key, so page
    control serializes them correctly.
    """
    _require_creds()
    # Two named tools the factory hands out in turn, so the session log can distinguish the calls.
    tools = [_stagehand_tool("factory-browser-1"), _stagehand_tool("factory-browser-2")]
    handed = {"n": 0}

    def make_navigator() -> Agent:
        tool = tools[min(handed["n"], len(tools) - 1)]
        handed["n"] += 1
        return _browsing_agent(
            "factory-navigator",
            "Your ONLY job: navigate your Stagehand browser to the EXACT URL you are given and confirm.",
            tool,
        )

    owner = _owner(
        "You delegate browser navigations to the 'navigate' sub-agent, ONE at a time. Always invoke "
        "a tool as {'input': '<task>'}. Delegate opening the first URL, wait for the answer, then "
        "delegate opening the second URL. Then give a short final answer.",
        [
            SubAgentTool(
                name="navigate",
                description="Open a URL in the shared browser. Pass {'input': '<task>'}.",
                agent_factory=make_navigator,
            )
        ],
        parallel_tool_calls_enabled=False,
    )

    result = _run_workflow(
        owner,
        f"First delegate opening {EXAMPLE_URL}. After it answers, delegate opening https://example.org/.",
    )

    _owner_output(result, owner)
    # Both factory instances that actually browsed must have landed on the same session.
    browsed = [name for name in ("factory-browser-1", "factory-browser-2") if SESSION_LOG[name]]
    assert len(browsed) >= 1, "no factory subagent ever browsed"
    session_id = _assert_one_shared_session(*browsed)
    _assert_session_ended(session_id)


# ---------------------------------------------------------------------------------------------
# 8. CROSS-RUN: a supplied Context carries state from one whole run to a later, separate run.
# ---------------------------------------------------------------------------------------------
@pytest.mark.integration
def test_context_persists_state_across_separate_runs():
    """Two independent workflow runs share a stable ``browser_context_id`` (the "one per end user"
    pattern). Run 1 sets a PERSISTENT cookie and ends its session; run 2 — a brand-new session —
    loads the same Context and sees the cookie. Different sessions, one Context: this is the axis
    that mirrors carrying a user's login across conversation turns.
    """
    _require_creds()
    from browserbase import Browserbase

    api_key = os.getenv("BROWSERBASE_API_KEY")
    project_id = os.getenv("BROWSERBASE_PROJECT_ID")
    context_id = Browserbase(api_key=api_key).contexts.create(project_id=project_id).id

    def build_owner(tool_name: str) -> tuple[Agent, RecordingStagehand]:
        tool = _stagehand_tool(tool_name, browser_context_id=context_id)
        navigator = _browsing_agent(
            "navigator",
            "Your ONLY job: use your Stagehand browser tool to navigate to the EXACT URL you are "
            "given and report the page's text content verbatim.",
            tool,
        )
        owner = _owner(
            "You delegate one browser task to 'navigator'. Always invoke a tool as {'input': "
            "'<task>'}. Delegate opening the URL and report what the navigator saw.",
            [SubAgentTool(name="navigator", description="Open a URL. Pass {'input': '<task>'}.", agent=navigator)],
            parallel_tool_calls_enabled=False,
        )
        return owner, tool

    cookie_url = _cookie_set_url("dynamiq_ctx", "persisted", persistent=True)
    try:
        # --- RUN 1: set the persistent cookie, then let the owner teardown end the session, which
        # is what writes the cookie into the Context.
        owner1, _ = build_owner("run1-browser")
        result1 = _run_workflow(owner1, f"Have the navigator open {cookie_url} and confirm it loaded.")
        _owner_output(result1, owner1)
        run1_session = _assert_one_shared_session("run1-browser")
        _assert_session_ended(run1_session)
        # If httpbin 5xx'd while SETTING the cookie, there is nothing to carry — skip before run 2
        # even attempts it (the logs are keyed per tool name, so no clearing between runs is needed).
        _skip_if_httpbin_unavailable("run1-browser")

        # --- RUN 2: a fresh session loading the same Context sees the cookie.
        owner2, _ = build_owner("run2-browser")
        result2 = _run_workflow(owner2, f"Have the navigator open {COOKIE_READ_URL} and report the cookies listed.")
        _owner_output(result2, owner2)
        run2_session = _assert_one_shared_session("run2-browser")

        assert run1_session != run2_session, "the two runs unexpectedly reused one session"
        _skip_if_httpbin_unavailable("run2-browser")
        assert _tool_saw(
            "run2-browser", "dynamiq_ctx"
        ), "run 2 did not see run 1's cookie — the Context did not carry state across runs"
        _assert_session_ended(run2_session)
    finally:
        try:
            Browserbase(api_key=api_key).contexts.delete(context_id)
        except Exception:
            pass  # best effort — a leaked test Context is harmless, a failing cleanup should not be
