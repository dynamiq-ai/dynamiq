"""Diagnostic: can two Stagehand tools drive ONE live Browserbase session at the same time?

This is not an example of intended usage — it is a probe that answers the question the shared-browser
design (docs/design/shared-browser-lease-fix.md) currently ASSUMES the answer to. That design
(Model C) serializes agents and hands state over by closing sessions, because we assumed state only
crosses on close. That assumption was verified for Browserbase *Contexts* (they persist at session
end), but never for two clients attached to one live session — a different mechanism.

Playwright cookies live on the BrowserContext, not the page, and Stagehand attaches to the remote
browser's existing context (``stagehand/browser.py``), so live cookie sharing is plausible. If it
holds, agents could share auth instantly AND run in parallel, which is strictly better than Model C.

The probe runs three checks against one real session, with tool A never closing:

  1. ATTACH  — can B connect to A's live session at all?
  2. COOKIES — B reads a cookie that A set. Shared live, without A closing?
  3. PAGES   — B navigates elsewhere; is A still on its own page, or did it move too?

Requires OPENAI_API_KEY, BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID.
Run: python examples/components/tools/stagehand_tool/probe_live_session_sharing.py
"""

import os
import traceback

from dynamiq.connections import Stagehand as StagehandConnection
from dynamiq.nodes.tools import Stagehand
from dynamiq.runnables import RunnableConfig, RunnableStatus

COOKIE_NAME = "dynamiq_probe"
COOKIE_VALUE = "live-A"
COOKIE_SET_URL = f"https://httpbin.org/cookies/set/{COOKIE_NAME}/{COOKIE_VALUE}"
COOKIE_READ_URL = "https://httpbin.org/cookies"
OTHER_URL = "https://example.com/"
OTHER_MARKER = "example domain"

MODEL_NAME = "gpt-4o"


def _tool(name: str) -> Stagehand:
    return Stagehand(
        name=name,
        connection=StagehandConnection(model_api_key=os.getenv("OPENAI_API_KEY")),
        model_name=MODEL_NAME,
    )


def _act(tool: Stagehand, **input_data) -> str:
    """Run one Stagehand action and return its content as text, raising on failure."""
    result = tool.run(input_data=input_data, config=RunnableConfig())
    if result.status != RunnableStatus.SUCCESS:
        raise RuntimeError(f"{tool.name} {input_data.get('action_type')} failed: {result.error}")
    return str(result.output.get("content"))


def probe() -> dict:
    """Drive the three checks. Returns a verdict dict; never raises for an expected 'no'."""
    findings: dict = {"attach": None, "cookies": None, "pages": None, "detail": {}}

    tool_a = _tool("probe-A")
    tool_b = _tool("probe-B")
    try:
        # --- A: create a session and set a cookie in it. A stays OPEN for the whole probe.
        _act(tool_a, action_type="goto", url=COOKIE_SET_URL, brief="A sets a cookie")
        session_id = tool_a._session_id
        print(f"[A] session {session_id} created, cookie {COOKIE_NAME}={COOKIE_VALUE} set")

        # --- 1. ATTACH: point B at A's live session (the tool's existing resume path).
        tool_b._session_id = session_id
        try:
            _act(tool_b, action_type="goto", url=COOKIE_READ_URL, brief="B opens the cookie page")
        except Exception as exc:
            findings["attach"] = False
            findings["detail"]["attach_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[B] could NOT attach to A's live session: {exc}")
            return findings

        findings["attach"] = tool_b._session_id == session_id
        print(f"[B] attached, session {tool_b._session_id}")
        if not findings["attach"]:
            findings["detail"]["attach_error"] = "B opened its own session instead of joining A's"
            return findings

        # --- 2. COOKIES: does B see A's cookie while A is still open?
        b_cookies = _act(
            tool_b,
            action_type="extract",
            instruction="Return the full text of the page body verbatim.",
            brief="B reads the cookies the page reports",
        )
        findings["cookies"] = COOKIE_NAME in b_cookies
        findings["detail"]["b_sees"] = b_cookies[:300]
        print(f"[B] page body: {b_cookies[:200]}")

        # --- 3. PAGES: B navigates away; is A still where it was?
        _act(tool_b, action_type="goto", url=OTHER_URL, brief="B navigates elsewhere")
        a_now = _act(
            tool_a,
            action_type="extract",
            instruction="Return the full text of the page body verbatim.",
            brief="A reports the page it is on",
        )
        findings["detail"]["a_sees_after_b_navigated"] = a_now[:300]
        moved = OTHER_MARKER in a_now.lower()
        findings["pages"] = "shared" if moved else "independent"
        print(f"[A] page body after B navigated: {a_now[:200]}")

        return findings
    finally:
        for tool in (tool_b, tool_a):  # B first: both point at one session, A owns it
            try:
                tool.close()
            except Exception as exc:
                print(f"Warning: closing {tool.name} failed: {exc}")


def probe_own_page() -> dict:
    """Decisive follow-up: give B its OWN page instead of the one it inherits.

    A re-attaching client is handed ``existing_pages[0]`` by ``connect_browserbase_browser``, so
    phase 1 cannot tell context-level cookie sharing apart from both clients driving one page. Here
    B opens its own page in the same browser context, which separates the two: if A keeps its page
    AND B still sees A's cookie, agents can browse in parallel on shared auth.

    Reaches into ``client.context`` directly — probe-only. Real support would mean the tool managing
    a page per agent.
    """
    findings: dict = {"own_page": None, "pages": None, "cookies": None, "detail": {}}

    tool_a = _tool("probe-A2")
    tool_b = _tool("probe-B2")
    try:
        _act(tool_a, action_type="goto", url=COOKIE_SET_URL, brief="A sets a cookie")
        session_id = tool_a._session_id
        print(f"[A2] session {session_id}, cookie set")

        tool_b._session_id = session_id
        _act(tool_b, action_type="goto", url=COOKIE_READ_URL, brief="B attaches")

        # Give B its own page in the shared browser context.
        try:
            page = tool_b._run_in_loop(tool_b.client.context.new_page())
            tool_b.client.page = page
            findings["own_page"] = True
            print("[B2] opened its own page via StagehandContext.new_page()")
        except Exception as exc:
            findings["own_page"] = False
            findings["detail"]["new_page_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[B2] could NOT open its own page: {exc}")
            return findings

        # B drives its own page somewhere distinctive.
        _act(tool_b, action_type="goto", url=OTHER_URL, brief="B navigates its own page")

        # Is A still on ITS page?
        a_now = _act(
            tool_a,
            action_type="extract",
            instruction="Return the full text of the page body verbatim.",
            brief="A reports its page",
        )
        findings["detail"]["a_sees"] = a_now[:300]
        findings["pages"] = "shared" if OTHER_MARKER in a_now.lower() else "independent"
        print(f"[A2] page body: {a_now[:160]}")

        # And does B, on its own page, still see A's cookie?
        _act(tool_b, action_type="goto", url=COOKIE_READ_URL, brief="B re-reads cookies on its page")
        b_cookies = _act(
            tool_b,
            action_type="extract",
            instruction="Return the full text of the page body verbatim.",
            brief="B reads cookies from its own page",
        )
        findings["detail"]["b_sees"] = b_cookies[:300]
        findings["cookies"] = COOKIE_NAME in b_cookies
        print(f"[B2] page body: {b_cookies[:160]}")

        return findings
    finally:
        for tool in (tool_b, tool_a):
            try:
                tool.close()
            except Exception as exc:
                print(f"Warning: closing {tool.name} failed: {exc}")


def report_own_page(findings: dict) -> None:
    print("\n" + "=" * 72)
    print("VERDICT — phase 2 (B on its own page)")
    print("=" * 72)
    print(f"  B could open its own page          : {findings['own_page']}")
    print(f"  Pages after B navigated            : {findings['pages']}")
    print(f"  B still sees A's cookie            : {findings['cookies']}")
    print("-" * 72)
    if not findings["own_page"]:
        print("  => Agents cannot hold separate pages. Keep exclusion for page control.")
    elif findings["pages"] == "independent" and findings["cookies"]:
        print("  => PARALLEL MODEL AVAILABLE. Separate pages, shared auth, live. Serialization")
        print("     and close-based handoff are both unnecessary: give each agent its own page")
        print("     on one shared session. Ownership/park-and-steal can go.")
    elif findings["pages"] == "independent":
        print("  => Separate pages work, but auth did NOT carry across them. Investigate before")
        print("     redesigning: cookies should be per-context, so this suggests separate contexts.")
    else:
        print("  => Pages still collide even with new_page(). Keep exclusion for page control,")
        print("     but close-based handoff can still go (state is live).")
    print("=" * 72)


def report(findings: dict) -> None:
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  1. B can attach to A's live session : {findings['attach']}")
    print(f"  2. Cookies shared live (A not closed): {findings['cookies']}")
    print(f"  3. Pages                            : {findings['pages']}")
    print("-" * 72)

    if not findings["attach"]:
        print("  => Model C is REQUIRED. Agents cannot share a live session at all, so")
        print("     close-based handoff is the only mechanism. Keep the current design.")
    elif not findings["cookies"]:
        print("  => Model C is REQUIRED. Agents can attach, but state does NOT cross while")
        print("     a session is open, so handoff really does depend on close.")
    elif findings["pages"] == "independent":
        print("  => Model C is the WRONG SHAPE. Agents share auth live AND hold their own")
        print("     pages: they could run in PARALLEL with no serialization, no close-based")
        print("     handoff, and no lost page position. Redesign around a shared live session")
        print("     with a page per agent; ownership/park-and-steal would be unnecessary.")
    else:
        print("  => Model C is TOO STRONG. Agents share auth live, but collide on one page,")
        print("     so page control still needs serializing. A middle design wins: keep")
        print("     exclusion for page control, drop close-based handoff (state is already")
        print("     live) — no lost page position, no per-turn close, simpler than today.")
        print("     Check whether StagehandContext.new_page() gives each agent its own page;")
        print("     if so, this collapses into the 'independent' case above.")
    print("=" * 72)


def main():
    missing = [
        name for name in ("OPENAI_API_KEY", "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID") if not os.getenv(name)
    ]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}. Set them and re-run.")
        return

    print(f"Probing live Browserbase session sharing (cookie target: {COOKIE_READ_URL})\n")
    try:
        findings = probe()
    except Exception:
        print("\nProbe failed before reaching a verdict:\n")
        traceback.print_exc()
        print("\nIf this is a network/httpbin failure rather than a browser one, re-run;")
        print("the probe depends on httpbin.org being reachable.")
        return

    report(findings)

    # Phase 1 cannot separate "shared cookies" from "same page" when both clients inherit page[0].
    # Only worth resolving if they could attach and state did cross.
    if findings.get("attach") and findings.get("cookies"):
        print("\nPhase 1 could not separate context-level sharing from page sharing. Resolving...\n")
        try:
            report_own_page(probe_own_page())
        except Exception:
            print("\nPhase 2 failed:\n")
            traceback.print_exc()


if __name__ == "__main__":
    main()
