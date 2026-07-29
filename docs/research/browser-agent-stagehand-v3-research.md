# Browser agent (Stagehand tool) — configuration research, July 2026

Scope: why our Stagehand/Browserbase-based browser agent underperforms compared to
current stock Stagehand, cross-checked against the latest Stagehand documentation and
SDKs, plus a review of our own wiring (`dynamiq/nodes/tools/stagehand.py`,
`dynamiq/connections/connections.py`, shared browser session in
`dynamiq/nodes/agents/shared_session.py`).

## TL;DR

1. **We are two generations behind Stagehand.** `pyproject.toml` pins
   `stagehand>=0.4.0,<0.5.0` (lock file: **0.4.1**, released 2026-01-13). The PyPI
   `stagehand` package is now at **3.22.0** (released 2026-07-22) — a complete v3
   rewrite. All of Browserbase's 2026 quality work shipped to the v3 execution
   engine, not the one we call.
2. **Every act/extract/observe we run executes on the legacy v2 server.** The 0.4.x
   SDK defaults to `use_api=True` with `env=BROWSERBASE`, which sends operations to
   `api.stagehand.browserbase.com/v1`. The v3 server (used by SDK 3.x) is where the
   accuracy improvements live: accessibility-tree-based `extract` by default,
   collapsed/deduplicated text nodes, CDP screenshots, `ignore_selectors` for
   observe/extract, verified sessions, and the Model Gateway. This is the most
   plausible root cause of "our browser agent doesn't perform well" — it is not a
   single misconfigured flag on our side.
3. **Our example configs pin a May-2025 model.** `anthropic/claude-sonnet-4-20250514`
   (and `gpt-4o` in several examples). Stagehand docs now recommend
   `anthropic/claude-sonnet-4-6`, `openai/gpt-5`, or `google/gemini-3-flash-preview`
   (fastest) — and offer the **Browserbase Model Gateway**: pass the Browserbase API
   key as the model key and Browserbase routes to the provider with retries, rate
   limit handling, and action caching included. No separate provider key needed.
4. **The v3 Python SDK is a pure API client** — it does not manage a browser or
   embed Playwright. Sessions are explicit: `sessions.start(model_name=...) →
   act/extract/observe/navigate/execute → end`, with `browserbase_session_id` resume
   and `browserbase_session_create_params` (context + persist, keep-alive, timeout)
   still supported. That maps cleanly onto our `SharedSession` design, but adopting
   it is a rewrite of `dynamiq/nodes/tools/stagehand.py`, not a version bump.
5. **New capability worth adopting:** v3's `session.execute` runs a full multi-step
   agent loop server-side from one natural-language task, with SSE streaming
   (`stream_response=True`). This could replace many of our per-step act loops and
   fits a chat-UI streaming experience directly.

## What we run today

- `dynamiq.nodes.tools.Stagehand` wraps the v2-era Python client
  (`stagehand.Stagehand`, 0.4.1) on a dedicated background event loop, with
  Browserbase or Steel connections.
- Browserbase path: `StagehandConfig(env="BROWSERBASE", ...)` → SDK default
  `use_api=True` → session created on the legacy Stagehand API v1 endpoint;
  `modelName` is bound into the session at creation time.
- Steel path: `env="LOCAL"` with a CDP URL → `use_api=False` → act/extract/observe
  run **client-side** through the old SDK's litellm-based inference and 2025-era
  prompts. Quality profile is different from (and now well behind) the hosted path.
- Shared browser session (PR #845): first tool call creates a keep-alive Browserbase
  session bound to a persistent Context (`persist: True`), later agents attach via
  `browserbase_session_id`, teardown detaches instead of closing. This design is
  sound and survives the v3 migration — v3 keeps both `browserbase_session_id`
  resume and the same session-create params.

### 0.4.1 defaults we inherit (mostly unexposed)

| Setting | Default | Note |
|---|---|---|
| `use_api` | `True` | Silently flips to `False` if `experimental=True` **or** `browserbase_session_create_params.region != "us-west-2"`. Setting a non-default region switches the entire execution engine to the legacy client-side path — worth knowing before blaming the model. |
| `self_heal` | `True` | Retry/repair of failed actions. |
| `dom_settle_timeout_ms` | `3000` | Can be raised for heavy SPAs via connection `extra_config`. |
| `verbose` | `1` | |
| `system_prompt` | `None` | Injectable via `extra_config` — we never use it. |
| `model_name` | `gpt-4o` | We always override; enum in 0.4.1 only knows 2025 models, but arbitrary strings pass through. |

Anything above can be tuned today without code changes through
`Browserbase.extra_config` / `SteelBrowser.extra_config` (forwarded into
`StagehandConfig`), which is our only real short-term tuning lever.

## Details verified in our wiring (correct, keep as-is)

- `client.model_name` is set **before** `client.init()`, and the v2 client sends
  `modelName` from that attribute at `_create_session` time — so the configured
  model does reach the server. Caveat: when a tool *attaches* to an existing shared
  session, session creation is skipped, so a differing `model_name` on the attaching
  tool is silently ignored — the creator's model wins for the whole shared session.
  (Same per-session model semantics in v3.)
- `keep_alive`, explicit `timeout`, and `context: {id, persist: true}` are applied
  when creating the shared session; the `"timeout"`-not-`"api_timeout"` camelCase
  pitfall is handled.
- Signal-handler suppression, detach-instead-of-close under sharing, and the
  clone/serialization guard for parallel calls are all still required with 0.4.x.

## UI-facing finding: live view URL is off by default

`is_return_live_view_url_enabled` defaults to `False`. The live view URL
(Browserbase `debugger_fullscreen_url` / Steel `session_viewer_url`) is only fetched
when that flag is on, and under browser sharing
`shared_browser.set_browser_live_view_url(...)` is then called with `None`, so
`Agent._maybe_surface_live_view` has nothing to surface into the execution result.
If the platform chat UI is expected to render an embedded live browser panel and
doesn't, check that the workflow YAML the UI generates sets
`is_return_live_view_url_enabled: true` on the Stagehand tool. This is the one
plain "misconfigured in the UI" candidate found in this repo; the rest of the UI
flow (chats endpoint streaming) lives in the platform repo and should be checked
there against the `live_view_url` key emitted by the agent.

## Stagehand v3 Python SDK (3.22.0) — what changes

- Package is a Stainless-generated API client: `Stagehand(server="remote",
  browserbase_api_key=..., model_api_key=...)` then
  `client.sessions.start(model_name="anthropic/claude-sonnet-4-6", ...)` returning a
  session handle with `.act / .extract / .observe / .navigate / .execute / .end`.
  There is no `StagehandConfig`, no `.page`, and no embedded Playwright.
- `server="local"` runs a bundled Stagehand server binary against a local Chrome —
  this replaces the old `env="LOCAL"` path and is how the Steel/CDP story would be
  rebuilt (the session start params accept browser launch options incl. CDP).
- Session start params still accept: `browserbase_session_create_params` (browser
  settings, context, fingerprint, viewport, proxies), `browserbase_session_id`
  (resume), `dom_settle_timeout_ms`, `self_heal`, `system_prompt`, `verbose`,
  `x_stream_response` (SSE).
- `browserbase_project_id` is deprecated (v3.20.0); the API key alone scopes the
  project.
- Our file-chooser `upload` action relies on `client.page.expect_file_chooser()` —
  gone in v3. Replacement: connect our own Playwright to the Browserbase session
  over CDP for chooser interception, or keep using the Browserbase SDK upload API
  (`sessions.uploads.create`), which we already use and which stays valid. Downloads
  and screenshots likewise remain available via the Browserbase SDK / CDP.
- Recent v3 additions we would gain: `ignore_selectors` on observe/extract
  (3.21.0), screenshot option on extract, verified sessions, Bedrock/Vertex/Azure
  Entra model auth passthrough (3.20.0–3.22.0).

## Recommendations (ordered)

1. **Short term (no code change):** refresh the model in workflow configs to a
   current one (e.g. `anthropic/claude-sonnet-4-6` or `google/gemini-3-flash-preview`
   for speed) and consider the Browserbase Model Gateway so the platform doesn't have
   to manage provider keys. Verify no config sets a non-`us-west-2` region without
   realizing it disables the hosted engine. Turn on
   `is_return_live_view_url_enabled` wherever the UI should show the browser.
2. **Main fix:** migrate `dynamiq/nodes/tools/stagehand.py` to the v3 SDK
   (`stagehand==3.x`). Because the package name is unchanged, the old and new SDKs
   cannot coexist in one environment — this is a hard switch release, not a
   feature-flag rollout. The tool's public schema (action_type/instruction/brief/
   url/files) can stay identical; the shared-session design carries over via
   `browserbase_session_id` + session-create params. The upload action needs the CDP
   Playwright rework described above; Steel support moves to `server="local"` with
   CDP launch options.
3. **Evaluate `session.execute`** (server-side multi-step agent with streaming) as a
   higher-level alternative to our per-step act loop for the chat/"super agent" UI —
   fewer round trips through our agent loop, and step events can stream to the UI.
4. Bump `browserbase` (currently 1.7.0) alongside, for verified sessions and current
   uploads/downloads endpoints.

## Sources

- PyPI `stagehand` (3.22.0, 2026-07-22): https://pypi.org/project/stagehand/
- Python SDK repo + v2→v3 migration pointer: https://github.com/browserbase/stagehand-python
- Python SDK changelog (3.20.0–3.22.0 entries): https://github.com/browserbase/stagehand-python/blob/main/CHANGELOG.md
- Python migration guide: https://docs.stagehand.dev/v3/migrations/python
- Stagehand v3 announcement: https://www.browserbase.com/changelog/stagehand-v3
- Model Gateway: https://www.browserbase.com/changelog/model-gateway and https://docs.browserbase.com/platform/model-gateway/overview
- Locally inspected wheels: `stagehand==0.4.1` vs `stagehand==3.22.0` (config
  defaults, session start params, custom session wrapper).
