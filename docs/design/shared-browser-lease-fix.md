# Design fix: shared-browser mutual exclusion + state hand-off

**Status:** v5 — **Model D + Context is implemented and verified live** (§5.2). §4 (Model C) is
superseded; it is kept only as the record of a design built on a premise §5.1 disproved.
**Scope:** `feat/shared-execution-session-p3-browser-sharing`
**Addresses review findings:** #1, #2, #5, #6, #7 (concurrency + lifecycle blockers)
**Already fixed on branch:** #3 (stale `_session_id`), #4 (live-view back-fill), #8 (guard dedup)

---

## 0. Corrected requirement (from the maintainer)

> "Only one agent may execute Stagehand at a time, because session data (cookies, logins)
> becomes available only when the tool **closes**."

This is **turn-level** (whole-agent) exclusion, **not** command-level. It also imposes a hand-off
*timing* constraint: agent B must not start until agent A's browser state is durable. That rules
out the first draft's per-command lock (see §3).

---

## 1. Verified facts (checked against the code + the installed `stagehand`/`browserbase` libs)

1. **`close()` ends the Browserbase session server-side.** `StagehandClient.close()` calls
   `_execute("end", {"sessionId": ...})` — docstring: *"For BROWSERBASE: Ends the session on the
   server"* (`stagehand/main.py:476-505`). Session end is the persist/finalize point; this is the
   mechanic behind "state only available on close."
2. **Close is NOT automatic per agent turn.** No agent/flow/workflow lifecycle calls
   `Stagehand.close()`. `Flow` only shuts down its thread-pool executor (`flows/flow.py:565,783`),
   never nodes. The only close triggers are:
   - `Stagehand.__del__` → GC, non-deterministic; and
   - **this branch only:** the owner agent's `_teardown_shared_browser` → `SharedSession.close_browser()`
     → the **creator** tool's `close_callback` (`stagehand.py:255`).
   So a subagent finishing its turn closes nothing; only the *creator's* session closes, and only at
   the *owner's* turn end.
3. **Resume needs a RUNNING session** (`stagehand/browser.py:44-49`) and attaches via
   `connect_over_cdp`. The branch's model is concurrent live CDP connections to one `session_id`.
4. **Finding #6 CONFIRMED (upgraded from PLAUSIBLE).** Attached tools set `client.session_id` to the
   shared id, so any attached tool's `close()` — including via GC — calls `end` on the **shared**
   session and kills it for every other agent.

### Consequence

Facts 1 + 2 together mean: **if state crosses only on close, and close happens only at the owner's
turn end, then subagents cannot hand logged-in state to one another mid-run via close.** The only
thing sharing state mid-run today is the *live* CDP connection (fact 3). So the design must pick a
lane, below.

---

## 2. The fork: how does state actually hand off?

| | **Model C — persistent Context + serial sessions** (chosen) | **Model D — one live session + fixed lease** |
|---|---|---|
| Sharing unit | a persistent Browserbase **Context** (durable cookie/auth store) | one live `session_id`, many concurrent CDP connections |
| State hand-off | on **close** (session end persists the Context); next agent's fresh session loads it | **live** (same remote browser), no close needed |
| Matches "state only on close"? | **Yes** — by construction | No — assumes live connections share cookies |
| Exclusion needed | one agent's session at a time; next waits for prior **close** | one **driver** at a time over the shared session |
| Per-turn close required? | **Yes** — must add it (fact 2 says it's missing) | No — close once at the very end |
| Live-view continuity | breaks across hand-off (new session = new live-view URL) | preserved (one session) |
| Effort / risk | higher (new context wiring + per-turn close) | lower (keep architecture, fix the lease) |
| Open dependency | configure a persistable Context | **must** confirm concurrent CDP connections share cookies (live Browserbase test) |

**Chosen: Model C** — it is the only model consistent with the stated constraint. Model D is
retained in §5 as the lighter path **iff** a live test ever proves concurrent connections propagate
cookies (which would contradict the maintainer's observation, so it must be proven, not assumed).

---

## 3. Why the earlier per-command lock (old "Option A") is wrong here

A per-command lock releases between `goto` and `act`, so agent B can interleave mid-turn. If A's
state is only durable on close, B would drive with stale/unauthenticated state. Command-level
granularity is the *opposite* of what's required. **Rejected.** (Kept in appendix for the record.)

---

## 4. Implemented design — Model C

### 4.1 Sharing via a persistent Context, not a live session id

- The first agent to browse creates a Browserbase **Context** and offers its id to
  `SharedSession.adopt_browser_context_id` (first writer wins); every later agent gets that same id
  back. `Stagehand._apply_shared_browser_context` injects it into
  `browserbase_session_create_params` as `browser_settings.context = {id, persist: True}`, so each
  agent's own session **loads** the shared context on start and **saves** it on end.
- `record_browser` / `browser_session_id` / the live-session attach in
  `_attach_shared_browser_before_init` are gone, replaced by "adopt the shared **context id**."
- The shared **live-view URL** flips to *latest wins* (`set_browser_live_view_url`): each agent has
  its own session, so the newest URL is the one worth watching.

### 4.2 Agent-level exclusion (one session at a time)

- `SharedSession.acquire_browser_ownership(agent_run_key)` is taken on an agent's **first** browser
  use in a turn and released at that agent's turn end. While held, no other agent may open a session
  against the shared context, which preserves ordered hand-off and stops two sessions writing the
  context concurrently.
- **Timeout** (default 300s) on acquisition converts the old silent `wait()` hang (#5) into a clear
  error naming the cause.
- **Keying:** per agent-run, and **idempotent rather than counted**. Acquire happens per browser
  *call*, release exactly once per agent *turn*, so a counted (re-entrant-by-depth) acquire would
  leave depth > 0 forever the moment an agent browsed twice — the browser would never be handed on
  and every later agent would time out. Ownership is a `Condition`, not a `Lock`: it is not
  thread-owned (claimed in a tool's thread, released in the agent's), and concurrent calls from the
  *same* agent must pass straight through instead of one waiting on the other.

### 4.2.1 Parallel tool calls within one agent (#2)

Parallel non-subagent tool calls are cloned for isolation (`base.py`). For Stagehand under an active
shared session that is exactly wrong: both clones share the agent's run key, so both pass the
ownership check, and each clone then opens its **own** session against the shared Context — the two
load the same cookies, diverge, and clobber each other when they persist on close. So:

- `Node.is_clone_safe_for_parallel()` (default `True`) is overridden by `Stagehand` to return `False`
  while a shared browser session is active, keeping parallel calls on one instance and therefore one
  session per agent turn. Steel connections stay clone-safe (they don't join the Context).
- `Stagehand.execute` then serializes calls on that instance with a per-instance `_call_lock`, since
  they would otherwise race on one client/page. Clones get a **fresh** lock via
  `init_call_lock` in `_clone_init_methods_names` — `clone()` copies private attrs shallowly, so
  without it independent (unshared) parallel calls would serialize against each other.

Parallel *subagents* need none of this: each gets its own run key from its own `Agent.execute`, so
they serialize correctly through ownership — they queue rather than fail, at the cost of the
parallelism itself and with the acquisition timeout as a ceiling on how long the tail may wait.
- **No fallback key** (#7): a tool running outside `Agent.execute` (no `_current_agent_run`) logs a
  warning and runs an isolated, unshared session. Acquiring under a substituted key would leave the
  browser locked for the whole run, since the release path only matches the real key.

### 4.3 Deterministic close-on-turn-end

- `Agent._teardown_shared_browser` calls `release_browser_ownership` in `execute()`'s `finally`;
  that closes every session the agent registered (→ context persisted) *before* freeing the lock,
  so the next agent unblocks only once the previous agent's state is durable.
- Closes accumulate rather than replace (`register_active_session_close`), since one agent may drive
  several Stagehand tool instances in a turn. A close that raises is logged and the lock is still
  freed — a failing teardown must not deadlock every waiting agent.
- Because `close()` also tears down the tool's event loop, `Stagehand.execute` now rebuilds it when
  the tool is reused in a later turn.

### 4.4 Nested co-drive + delegate

Ownership is held for the whole turn so a session survives an agent's successive calls. But an agent
parked in a delegate call is *not* browsing and cannot resume until the subagent returns — so
holding there would block the subagent for nothing. **Park-and-steal** exploits exactly that slack:

- `park_browser_ownership` marks the holder stealable for the duration of a delegate call
  (`Agent._run_tool`). A waiter that finds a parked owner closes that owner's sessions on its behalf
  — persisting the cookies — and takes over. On return, `unpark_browser_ownership` blocks while a
  steal is mid-flight, then no-ops if ownership moved; the agent transparently re-acquires on a
  fresh session at its next browser call.
- **Nothing is closed unless someone actually needs it**, so delegating to a *non-browsing* subagent
  leaves the owner's session (and page) intact. That ruled out the simpler "always release on
  delegate", which would have regressed that case.
- `_browser_stealing` serializes the steal itself: other waiters queue, the owner's unpark waits,
  and `release_browser_ownership` defers to an in-flight steal so a session is never closed twice.

Note how this differs from the ancestor-borrow it replaces (Appendix B). Model A *inferred* "the
holder is blocked" from agent-tree topology, which parallel execution falsified. Parked is a fact
the holder asserts about its own execution point, so it stays true under parallelism.

**Still unsupported:** using the browser and delegating browser work in the same *parallel* batch. A
sibling browser call of ours may be mid-command, and closing its session under it would break it, so
parking is skipped when `is_parallel` — the subagent fails with the timeout error. (A possible
refinement: park when a parallel batch contains no browser call of our own. Needs group-level
information that `_run_tool` does not have.)

`use_shared_browser_session.py` uses the coordinator shape as the default recommendation, and the
live integration test covers "subagent browses, then the owner does".

### 4.5 Fixing #6 in this model

Each agent owns exactly the session it created and closes exactly that one — no tool ever attaches
to a foreign live `session_id`, so the "attached tool ends the shared session" hazard disappears
structurally. (The old attach path is deleted.)

### 4.6 What was deleted

`BrowserSession`, `record_browser`, `browser_session_id`, `acquire_browser`, `release_browser`,
`_browser_owner_depth`,
`close_browser`, `_lease_stack`, `_lease_cond`, the ancestor-borrow logic, the `_agent_run_chain`
ContextVar and its `base.py` plumbing, and the live-session attach. `_current_agent_run` is kept as
the ownership key. Replaced by: a context id on `SharedSession`, one ownership lock with timeout,
and close-on-turn-end.

### 4.7 How the tests prove it

The live test can no longer assert "same `session_id`" — that would now mean the *opposite* of
correct — and `_session_id` is cleared on close anyway. `RecordingStagehand` logs each session's id
and the Context id it loaded, and the assertion becomes: **different session ids, one shared context
id**. State continuity is exercised as cookies (set on one agent's session, read back on another),
not as page position: a fresh session starts blank.

---

## 5. Alternative not taken — Model D (lighter, conditional)

Keep one live session; replace the lease with correct **turn-level** exclusion:

- One ownership lock, acquired on first browser use, held to turn end, with a timeout (fixes #5 as an
  error not a hang; the whole-turn hold is now *intended*, per §0).
- Delete the ancestor-borrow: nested delegation is handled by requiring the owner to finish its
  browser calls before delegating (it isn't executing a browser command while parked in the
  subagent, so the descendant acquires cleanly) — fixes #1.
- Per-agent-run keying + turn-hold removes the reentrant-no-op double-session path (#2); no fallback
  key means no leak (#7).
- **#6 still needs an explicit fix here:** mark tools attached-vs-owner; an attached tool's `close()`
  must *detach* (drop local client) without calling `end`; only the creator's `close_callback` ends
  the session, once.
- **Gate:** requires a live Browserbase test proving two concurrent CDP connections see each other's
  cookies. If they don't, Model D cannot satisfy §0 and Model C is mandatory.

---

## 5.1 EMPIRICAL FINDINGS — the Model D gate is satisfied (Model C's premise is false)

Run against live Browserbase via `examples/components/tools/stagehand_tool/probe_live_session_sharing.py`
(two Dynamiq `Stagehand` tools, one session, tool A never closed):

| Check | Result |
|---|---|
| B attaches to A's **live** session | **Yes** — same session id |
| B reads a cookie A set, **A still open** | **Yes** — `dynamiq_probe=live-A` |
| B gets an independent page | **No** — A and B always drive one page |
| B's own `StagehandContext.new_page()` helps | **No** — see mechanism below |

**Mechanism.** `use_api` defaults to `True` (`stagehand/config.py:102`), so every action is dispatched
server-side to `/sessions/{session_id}/{method}` (`stagehand/api.py:101`), keyed on the session
alone. The server acts on the session's active page, so a client-side `new_page()` is invisible to it.

**Consequences.**

1. **§0's premise is false.** State does NOT only cross on close — it is shared live. The whole
   close-based handoff, and the serialization built to order it, address a problem that isn't there.
2. **Close-on-turn-end (§4.3) is unnecessary**, and with it the lost page position, the
   session-recreation cost, and the loop rebuild it forced.
3. **The shared persistent Context (§4.1) is unnecessary** for intra-run sharing. It remains a
   legitimate but separate feature for *cross-run* persistence.
4. **Exclusion is still required** — but for **page control**, not for state ordering: agents share
   one page, so concurrent commands would stomp each other.
5. **Park-and-steal (§4.4) collapses to something trivial.** With nothing to close, a parked owner
   simply releases exclusion and re-acquires afterwards — no steal, no close, no in-flight race, and
   the page is left where the subagent put it (continuity, rather than loss).
6. **Finding #6 returns as a live hazard.** Back on a shared live session, any attached tool's
   `close()` ends it for everyone. Model D's creator-vs-attached distinction (§5) becomes required.

**Therefore: Model D (§5) is indicated, not Model C.** This should have been probed before building
Model C; the assumption was inherited from an observation about Context persistence and generalised
to the whole problem without test.

## 5.2 IMPLEMENTED — Model D + Context (two axes, deliberately separate)

| Axis | Mechanism | Verified |
|---|---|---|
| **Intra-run** (agent ↔ subagent) | ONE live session, all agents attached. State is live; nothing closes | Both subagents drove session `a8d63996…`; the checker read back the setter's cookie |
| **Cross-run** (run → later run) | That session loads a persistent Context and writes it back **when it ends** | Two runs, different sessions, same Context: `dynamiq_persist=ok` survived |

Sharing this shape is what the maintainer's superagent experience was actually about: waiting for
turn end was the *Context* axis, and it is preserved — it just is not how agents share within a run.

**Live agent/subagent coverage** — `tests/integration_with_creds/agents/test_shared_browser_live.py`
(skipped without creds) drives real agents against real Browserbase, asserting on each browser
tool's own returned content (not the owner's LLM-composed summary, whose wording is nondeterministic)
and on the Browserbase session status: two subagents share one session + cookie crosses; live page
position crosses; owner-browses-then-delegates does not deadlock; subagent-then-owner handoff; a
non-browsing delegate preserves the page; parallel browsing subagents share one session; factory-mode
subagents share one session; and a supplied Context carries a cookie across two separate runs.

**What it does:**

- `SharedSession.adopt_browser_session_id` — first agent to browse creates the session; every later
  agent attaches to it (`Stagehand._join_shared_browser` sets `_session_id`, so `_init_client`
  resumes rather than creating).
- `browser_context_id` on the tool — pass a stable id (e.g. one per end user) to carry state across
  runs; omit it for a throwaway per-run Context.
- `end_browserbase_session` (`sessions.update(status="REQUEST_RELEASE")`) is registered with the
  SharedSession and called **once**, at the owner's teardown. Deliberately independent of any tool
  instance: a subagent's tool may be collected long before the run ends.
- **Detach, not close (#6 fixed).** Stagehand has no detach — `close()` always ends the session
  (`main.py:496`), which would kill the browser for every other agent, including via `__del__` on a
  collected subagent tool. `_detach_shared_browser` cleans up the local browser/playwright
  resources and leaves the session running.
- `keep_alive=True` and an explicit `timeout` on the shared session: it now spans the whole run, so
  it must survive its creator disconnecting and must not inherit a short project default.
- **Page control** replaces ownership: agents share one page, so only one drives at a time. Held for
  a turn (so multi-step sequences are not interrupted), released at turn end and around delegate
  calls. Releasing costs nothing now — nothing closes, no state is lost, the page stays put.

**Two traps found only by running it live**, both worth remembering:

1. `api_timeout` (the SDK's field name) is camel-cased naively by Stagehand into `apiTimeout`, which
   the API rejects with a 400. The wire field is plain `timeout`.
2. `connection.config` is a **property that builds a fresh object per access** — mutating
   `tool.connection.config` mutates a throwaway. `_init_client` takes the config once and mutates
   that instance, which is correct; ad-hoc callers must do the same.

## 6. Open questions / follow-ups

Resolved while building:

- **Context persistence config** — `browserbase_session_create_params` with
  `browser_settings.context = {"id": ..., "persist": True}`, set on the `StagehandConfig` the tool
  builds per call (not via connection `extra_config`, so a user's own params are preserved).
  Verified against both session-creation paths in the installed lib (`stagehand/browser.py:51-64`
  local, `stagehand/api.py:23-45` server), which is why `project_id` is also filled in.
- **Ownership-lock timeout** — an independent 300s default rather than the tool's 3600s `timeout`:
  a wait this long almost always means an unsupported topology (owner delegating while holding the
  browser), and the error should arrive well before the tool's own budget expires.

Still open:

1. **Context lifecycle.** Nothing deletes auto-created Contexts, so a run that does not pass
   `browser_context_id` leaves one behind each time. Callers using a stable per-user id are
   unaffected. Auto-created ones should probably be deleted at owner teardown.
2. **Session cookies do not survive.** Only cookies with an expiry are persisted to a Context —
   browsers discard session cookies when the browser closes. Sites whose login is a pure session
   cookie will not carry across runs (they carry fine *within* a run, on the live session).
2. **Cross-process.** A `threading` lock only serializes within one process; agents spread across
   processes could open concurrent sessions on one Context. Out of scope for P3.
3. **Live verification.** The integration tests need real credentials and have not been run here;
   the cookie-continuity leg also depends on a public endpoint (httpbin) and is best-effort.
4. **(Model D gate, only if that path is ever revisited) Do concurrent CDP connections to one
   Browserbase session share live cookies?**

---

## Appendix A — rejected: per-command lock

Protect only the physical browser command with a short mutex. Correct for *physical* serialization
and never hangs, but allows mid-turn interleaving — invalid under close-only state hand-off (§3).
Recorded so the fork isn't lost: viable only if a future requirement drops the close-hand-off
constraint (e.g. true live co-driving on one session).

## Appendix B — rejected: topology-inferred reentrant borrow (the current branch)

Infer "holder is blocked" from agent-tree ancestry and hold the lease for the whole turn. The
ancestry signal is only valid under sequential execution; parallel tool execution copies ContextVars
into concurrent worker threads, so the "blocked ancestor" assumption is false and two runs drive at
once (#1), sibling calls share one key and no-op their lease (#2), and the hold has no timeout (#5).
Root cause: it conflates *session identity sharing* with *mutual exclusion* and uses topology —
the wrong signal — for the latter.
