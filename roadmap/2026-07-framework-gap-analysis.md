# Dynamiq vs. the Field — Framework Gap Analysis & Product Roadmap

**July 2026 · internal product document**

This is a code-level competitive audit of Dynamiq against the four agentic-framework stacks the market evaluates us against, and the product roadmap that falls out of it. It is based on reading source code, not marketing pages.

## How this was produced

All six codebases were cloned at their latest commits (July 21–22, 2026) and audited across a shared 14-dimension taxonomy by parallel code auditors. A cross-framework comparison then produced candidate gaps per dimension, and **every non-trivial claimed gap was adversarially verified by an independent pass whose job was to refute the claim by searching the Dynamiq codebase** — so features that exist but are under-documented were killed before reaching this document. Result: **64 confirmed gaps, 10 partial gaps, 0 refuted claims** (plus 19 unverified low-severity items). The full verified inventory, with per-gap evidence and Dynamiq strengths per dimension, is in the [appendix](./2026-07-gap-inventory-appendix.md).

| Framework | Version audited | Snapshot |
|---|---|---|
| **Dynamiq** (us) | 0.59.0 | 2026-07-22 |
| Google ADK (`adk-python`) | 2.5.0 (released 2026-07-16) | 2026-07-22 |
| CrewAI | 1.15.5 (crewai / -core / -cli / -tools) | 2026-07-21 |
| LangChain | langchain 1.3.14, langchain-core 1.5.0 + 15 partner pkgs | 2026-07-22 |
| LangGraph | 1.2.9 (+ checkpoint 4.1.1, sdk 0.4.2, cli) | 2026-07-21 |
| LangChain DeepAgents | 0.6.12 (+ deepagents-code, -cli, -acp, sandbox pkgs) | 2026-07-22 |

## Executive summary

Dynamiq enters H2 2026 with genuinely differentiated infrastructure: the deepest checkpoint/durability granularity in open source (mid-ReAct-loop resume with pending-tool-call replay), the only complete OSS knowledge-graph RAG loop, team-shared sandbox and browser sessions with browser-takeover HITL, per-call USD cost tracking no competitor ships, Pipedream's long-tail SaaS reach, and a full YAML round-trip that no LangChain-family framework matches.

But we are losing on the surfaces buyers evaluate **first**. We have zero OpenTelemetry support while ADK ships full OTel and CrewAI has ~18 vendor instrumentors. We have no local dev UI, no `serve` command, and no project scaffolding while every competitor ships an interactive first-five-minutes. Our callback system is observation-only, which single-handedly blocks the middleware-shaped ecosystem (guardrails, per-step models, context policies, tool interception) that all five competitors have standardized on. And we cannot evaluate the one thing we sell — agent trajectories.

The 4 critical and 20 high-severity verified gaps cluster around **two root causes**: one architectural (no interception layer on the agent loop) and one go-to-market (everything interactive or operational — trace UI, deploy, triggers — is gated on the paid Dynamiq platform, which reads as "the good parts are paid" during OSS evaluation). This roadmap fixes the architecture once, un-gates the minimum OSS surface needed to win evaluations, and rides the protocol wave (MCP server, A2A) before those become RFP hard-blockers.

## Competitive scorecard

Maturity per dimension, judged from code by independent auditors (`none < basic < solid < advanced < best-in-class`):

| Dimension | Dynamiq | ADK | CrewAI | LangChain | LangGraph | DeepAgents |
|---|---|---|---|---|---|---|
| Agent abstractions & reasoning | **advanced** | advanced | advanced | *best-in-class* | solid | advanced |
| Multi-agent orchestration | **advanced** | *best-in-class* | advanced | solid | advanced | solid |
| Tools & MCP | **advanced** | *best-in-class* | *best-in-class* | advanced | solid | solid |
| Memory & state | **advanced** | advanced | advanced | solid | *best-in-class* | solid |
| Context engineering | **advanced** | advanced | solid | *best-in-class* | basic | *best-in-class* |
| Streaming & realtime | **solid** | *best-in-class* | solid | advanced | advanced | basic |
| HITL & durability | **advanced** | advanced | advanced | advanced | *best-in-class* | advanced |
| RAG & knowledge | **advanced** — best of the six | basic | solid | advanced | basic | basic |
| Model support & multimodal | **advanced** | advanced | advanced | *best-in-class* | basic | advanced |
| Observability | **solid** | advanced | solid | advanced | advanced | solid |
| Evaluation & testing | **solid** | *best-in-class* | solid | basic | none | advanced |
| Deployment & dev experience | **solid** | *best-in-class* | advanced | solid | *best-in-class* | advanced |
| Safety & guardrails | **solid** | solid | solid | advanced | basic | solid |
| Interop & protocols | **basic** | *best-in-class* | advanced | basic | solid | solid |

Reading: we are competitive-to-leading in the *engine room* (orchestration, memory, durability, RAG, context) and weakest in the *showroom* (observability, evaluation, dev UX, protocols) — precisely the dimensions decided in the first hour of a framework evaluation.

## Where Dynamiq wins today

Verified differentiators no audited competitor matches (full detail per dimension in the appendix):

- **Durability**: mid-agent-loop checkpointing with pending-tool-call replay; declarative checkpoint triggers (`on_failure` by default, `on_cancel`, `on_input_timeout`); cancel yields a *resumable* run. Finer-grained than LangGraph's node-level semantics.
- **HITL depth**: approval gates on *any* node type (not just tools) with field-level `mutable_data_params` edit allow-lists; plan-level approval (`PlanApprovalConfig`); unanswered approvals auto-convert to resumable checkpoints; **browser-takeover HITL** mid-run — unique in the market.
- **Team-shared execution environments**: shared sandbox sessions with per-subagent isolated views, and shared live browser sessions with page-control locking. No competitor has either.
- **RAG stack**: 8 in-repo vector stores with writers+retrievers, 11 splitter strategies (incl. contextual retrieval), in-core rerankers, the only complete OSS knowledge-graph RAG loop (extraction → 3 graph stores → Cypher tool), hybrid dense+sparse retrieval, dry-run ingestion.
- **Cost observability**: per-call USD cost tracking incl. cache-token accounting via a shipped pricing registry — the only OSS framework of the six with dollars, not just tokens.
- **Streaming production**: bidirectional input over the live event stream (approvals/feedback in-stream, tied into checkpoint-on-timeout), tool-input delta streaming, uniform per-node streaming across all 28 providers.
- **Declarative story**: full YAML load *and* dump round-trip for entire multi-agent systems, plus workflow-generation utilities — the LangChain family has no declarative path at all.
- **Multimodal generation in core**: image generation/edit/variation across 6 providers, TTS/STT nodes — competitors have at most tool-level wrappers.
- **Safety detectors**: the only OSS prompt-injection/jailbreak detector of the six, plus the broadest moderation coverage (LlamaGuard, Lakera) and a three-tier code-execution containment story.
- **Trust posture**: no phone-home telemetry (CrewAI ships opt-out telemetry), everything in this repo works without the paid platform, Apache-2.0.

Also at parity and worth saying out loud (auditors flagged readers may assume otherwise): structured output (Pydantic/JSON-schema `response_format` across four inference modes), a skills system (matching ADK/CrewAI/DeepAgents — LangChain/LangGraph have none), MCP *client* support, and 28-provider LLM coverage.

## What competitors are building toward (momentum)

- **Google ADK** — weekly releases; 2.0 shipped a graph-style Workflow engine (aimed at LangGraph), then Workflow-as-Tool, managed server-side agents, agent/skill registries, `to_mcp_server()`, mature A2A, realtime voice/video avatars, GEPA prompt optimization, and blanket mTLS. Direction: regulated-enterprise GCP + protocols + realtime.
- **CrewAI** — pivoting from code-first crews to declarative versioned Flows (JSON/YAML) that run in Studio/TUI/control-plane; checkpoint fork/resume/diff CLI; skills registry with authenticated downloads; A2A + A2UI. Direction: low-code platform monetization + governance.
- **LangChain** — the agent loop is thin; value concentrates in composable **middleware** (summarization, HITL, PII, context editing, tool selection). Aggressive cross-provider standardization and cache-cost accounting. Direction: model-abstraction + middleware substrate.
- **LangGraph** — durable-execution runtime: delta checkpointing, crash-resumable error handling, WebSocket/reconnect/projection streaming for agent frontends. Direction: datacenter-grade reliability funneling to the paid platform.
- **DeepAgents** — batteries-included harness with per-model harness profiles and in-repo benchmarks; vertical products on top (dcode terminal agent, talon ambient runtime, ACP editor embedding); sandbox partner ecosystem. Direction: harness science + persistent personal agents.

Common thread: **middleware/interception, durable execution, MCP-server + A2A, and evals are where everyone is converging.** Two of those four are Dynamiq weaknesses today; one (durability) is our strength to defend.

---

# Product Roadmap — H2 2026

Eight themes, ordered by strategic priority. Horizons: **Now** = 0–3 months, **Next** = 3–6 months, **Later** = 6–12 months. Effort: S/M/L/XL.

## Theme 1: The Interception Layer (the keystone)

One investment unblocks the largest cluster of confirmed gaps: a middleware/hook pipeline on the agent loop with `before_model` / `after_model` / `wrap_model_call` / `wrap_tool_call` semantics that can **mutate, veto, retry, and short-circuit** — not just observe. Guardrails, PII redaction, per-step model routing, cache-aware compaction, tool policies, and reflection all become middleware instead of five separate subsystems. Ship the runtime first, then first-party middleware on top.

| Item | Horizon | Effort | Rationale |
|---|---|---|---|
| Agent middleware runtime: typed hook chain around `_run_llm` and `_run_tool` with mutate/veto/short-circuit contracts; keep existing callbacks as the observational tier | **Now** | XL | LangChain's `AgentMiddleware` is the 2026 standard extension mechanism; all 5 competitors have an equivalent — our single largest architectural deficit |
| Per-step / dynamic model control: accept `Callable[state] -> BaseLLM` for the model slot; separate planner/summarizer LLM fields on agent + summarization config | **Now** | M | LangGraph's callable-model and CrewAI's planning-LLM split are the standard cost-optimization ask; trivial once `wrap_model_call` exists |
| Tool-call policy middleware: programmatic allow/deny/mutate rules (glob-scoped like DeepAgents), arg sanitization, result override, per-tool call limits, SQL executor read-only mode | **Next** | M | LangGraph `on_tool_call` / ADK write-mode policies set the bar; our only gate today is human approval, which is not a policy system |
| Agent-level output guardrails with auto-retry: `guardrails=[...]` on Agent (predicate or LLM-judge), retry budget, human-substitutes-output decision | **Next** | M | CrewAI's `guardrail`/`guardrail_max_retries` + HallucinationGuardrail is a one-line config we currently answer with graph surgery |
| PII **redaction** middleware with local regex/Luhn detectors (block/redact/mask/hash), incl. streamed-delta rewriting; wire existing detectors in as middleware | **Next** | M | LangChain `PIIMiddleware`; our detection-only, remote-API-only PIIDetector is half a feature for the privacy-sensitive enterprise buyer |
| Context-editing middleware: zero-LLM-cost clear-old-tool-uses / truncate-historic-args passes; cache-aware compaction (stable prefix preservation); reactive context-overflow recovery | **Next** | M | LangChain `ClearToolUsesEdit` + DeepAgents' cache-safe middleware ordering; our compaction is LLM-cost-only and silently invalidates provider caches |
| Reflection/critic middleware (rubric grader loop) and a pluggable plan-and-execute planner abstraction | **Later** | L | DeepAgents `RubricMiddleware` and CrewAI's plan→execute→replan executor; build on middleware rather than as bespoke agent variants |

## Theme 2: Observability & Production Trust

Our clearest disqualification risk. A Dynamiq user today cannot get a trace into *any* standard backend without writing a custom handler, and no vendor ships a Dynamiq instrumentor. We already build a complete vendor-neutral Run tree in-process — the export layer is the missing 20%.

| Item | Horizon | Effort | Rationale |
|---|---|---|---|
| OpenTelemetry exporter: map the existing `TracingCallbackHandler` Run tree to OTel spans with GenAI semantic conventions, OTLP env-var configuration | **Now** | M | ADK's full OTel stack means any OTLP backend (Langfuse, Phoenix, Datadog) works with env vars; this is the 2026 enterprise pass/fail question |
| Promote Langfuse/AgentOps handlers from example scripts into shipped `dynamiq.callbacks.integrations` with pip extras; publish an instrumentor-authoring guide to seed vendor coverage | **Now** | S | CrewAI has ~18 vendor instrumentors; ours are copy-paste examples in a dev-only dependency group |
| Ship `py.typed` + adopt mypy in CI | **Now** | S | All five competitors ship it; cheap fix with outsized perception impact on the senior engineers who run evaluations |
| Console/rich verbose callback handler (pretty-printed steps, tool calls, costs) + run-level token/cost aggregation helper | **Now** | S | CrewAI's Rich rendering, LangChain's usage callback; we compute richer per-call USD data than anyone and then hide it in the trace tree |
| Structured logging overhaul: namespaced `dynamiq.*` logger hierarchy, JSON formatter option, remove import-time `basicConfig` (which hijacks host-app logging) | **Next** | S | ADK's namespaced loggers; the import-time side effect is actively hostile to embedding applications |

## Theme 3: Developer Experience — the First Five Minutes

Every competitor ships an interactive local surface; we ship a CLI whose every command talks to the paid cloud. This is the most visible differentiator in framework evaluations and the strongest "the good parts are paid" signal we currently send. Un-gate the minimum.

| Item | Horizon | Effort | Rationale |
|---|---|---|---|
| `dynamiq run` terminal chat + `dynamiq init` scaffolding (agent, RAG, multi-agent, YAML-workflow templates) | **Now** | M | ADK `adk create`/`adk run`, CrewAI templates, `langgraph new` — table-stakes onboarding we fail entirely today |
| `dynamiq serve`: generated FastAPI app with run/SSE/WebSocket endpoints over any workflow, session handling, using our existing typed streaming events | **Now** | L | ADK `api_server` and `langgraph dev` set the bar; today the answer is "copy an example or buy the platform" — friction exactly when a prototype becomes an endpoint |
| Docs: narrative concept guides beyond the 3 tutorials, `llms.txt`/`llms-full.txt`, AGENTS.md; fix the mkdocs build | **Now** | M | ADK ships llms.txt for AI-assisted coding; our 273 examples carry teaching load that docs should |
| Publish a 1.0 plan: semver stability guarantees, deprecation policy | **Now** | S | LangChain/LangGraph market "1.x stable APIs"; at v0.59 with no policy, we read as pre-production to conservative buyers |
| Local dev UI on top of `serve`: chat pane, live trace/step viewer (we already stream tool-input deltas and reasoning), run history | **Next** | XL | `adk web` and LangGraph Studio are the single most visible DX gap; our leading-edge streaming events deserve a first-party renderer |
| Graph/workflow visualization: packaged `to_mermaid()`/HTML renderer for YAML workflows and orchestrators + CLI `plot` command | **Next** | S | CrewAI `flow plot`; we uniquely have a declarative YAML DAG that nobody can currently *see* — cheap win on our own strength |
| Published streaming event schema (versioned) + a minimal TypeScript client for the serve endpoints | **Later** | M | LangGraph's SDK/protocol work; our typed Pydantic events are 80% of a contract nobody can consume from JS |

## Theme 4: Protocols & Interop — Don't Miss the MCP/A2A Window

ADK, CrewAI, and LangGraph are converging on MCP-server + A2A as the enterprise federation layer, and it's appearing in RFPs. We are strictly client-side today with no HTTP server at all; Theme 3's `serve` foundation makes these incremental.

| Item | Horizon | Effort | Rationale |
|---|---|---|---|
| `to_mcp_server()`: expose any Dynamiq agent/workflow as an MCP server (stdio + streamable HTTP) | **Now** | M | ADK's `to_mcp_server` and LangGraph's `/mcp` routes; being a tool inside Claude/IDEs/other agents is the fastest-growing agent distribution channel |
| MCP client OAuth: wire the MCP SDK's `auth` parameter (OAuthClientProvider, token storage/refresh) and callable header providers into MCPSse/MCPStreamableHTTP | **Now** | S | ADK/DeepAgents/CrewAI all have it; Notion/Linear/GitHub remote MCP servers are OAuth-protected and we require out-of-band tokens |
| OpenAPI spec → toolset generator: parse a spec into typed HttpApiCall-derived tools with per-operation auth | **Next** | M | ADK-only today but a frequent enterprise evaluation checkbox; a 30-operation internal API currently means 30 hand-written tool configs |
| A2A support: consume (RemoteAgent node) first, then serve via the `serve` stack with agent cards | **Next** | L | ADK is the reference implementation and CrewAI ships deep bidirectional A2A; consuming remote agents also fixes our "agents are strictly in-process" gap |
| Provider-native server-side tools in the agent loop (Anthropic web_search/code_execution, OpenAI built-ins) | **Next** | M | LangChain and ADK map these; provider-executed tools are cheaper and better than our third-party client-side equivalents |
| LangChain tool adapter (`from_langchain_tool`) | **Later** | S | ADK and CrewAI both wrap the ~700-integration catalog; lowers switching cost into Dynamiq for near-zero effort |
| Deferred tool loading / LLM tool pre-selection for large catalogs | **Later** | M | LangChain's tool-search middleware; amplified by our own Pipedream/MCP breadth — our strength currently worsens our token bill |

## Theme 5: Model Layer Modernization

We stay on LiteLLM (see "What not to build"), but three high-severity gaps are fixable within it.

| Item | Horizon | Effort | Rationale |
|---|---|---|---|
| Preserve and surface reasoning/thinking output: extract `reasoning_content`, typed reasoning blocks, thinking-chunk stream events, signature preservation across tool-use turns | **Now** | M | CrewAI/LangChain/ADK all handle it; we let users pay for thinking tokens they cannot see, and break Anthropic's thinking+tools contract on replay |
| OpenAI Responses API path (litellm.responses or targeted native call): stateful reasoning items, server-side built-ins | **Next** | L | CrewAI, LangChain, DeepAgents, ADK all support it and OpenAI's reasoning roadmap is Responses-only; chat-completions-only is a dead end |
| Prompt caching beyond Anthropic: Gemini/Bedrock/Azure cache controls, generalized `CacheControl` config; make compaction cache-prefix-aware (with Theme 1) | **Next** | M | ADK's Gemini cache manager and DeepAgents' Bedrock middleware; direct token-cost hit for long-running agents on non-Anthropic models |
| `input_audio` content block in VisionMessage for speech understanding (Gemini/GPT-4o-audio) | **Later** | S | CrewAI/LangChain/ADK route audio natively; cheap completeness fix that also softens our no-voice story |
| Ordered fallback chain (list, not single fallback LLM) + optional response cache | **Later** | S | LangChain `with_fallbacks`; our trigger conditions are better than theirs — chain depth of 1 is the only flaw |

## Theme 6: Orchestration & State — Make GraphOrchestrator Competitive

LangGraph normalized typed state, reducers, and parallel branches; ADK 2.0's new Workflow engine confirms this is where orchestration is converging. Sequenced deliberately: reducers before parallelism, because reducers are what make parallel writes safe.

| Item | Horizon | Effort | Rationale |
|---|---|---|---|
| Typed state schemas + declarative reducers: Pydantic context models on GraphOrchestrator, per-key merge policies replacing raise-on-conflict `merge_contexts` | **Next** | M | All four competitors have typed state; plain-dict + runtime-crash merge is below the DX bar users expect in 2026 |
| Concurrent branches, joins, and dynamic fan-out (Send-style dispatch) inside the LLM-routed graph | **Next** | L | LangGraph BSP/Send and ADK ParallelAgent+JoinNode; today fan-out means dropping to the DAG layer and losing dynamic routing — our own gpt_researcher port runs "parallel" research sequentially |
| Peer handoff primitive: agent-initiated `transfer_to(agent)` with state, expressible in graphs and from tools | **Next** | M | ADK `transfer_to_agent` and LangGraph Command; swarm/handoff architectures currently require regex-parsing conditional edges |
| SQLite + Redis checkpoint backends; publish the internal BackendTestMixin as a conformance kit | **Next** | S | LangGraph's conformance suite spawned a community backend ecosystem; we already have the test mixin — package it |
| Validated checkpoint edit/rewind API (`update_state`-style) before resume | **Later** | M | LangGraph `update_state`; "fix the wrong tool output and continue" is a production HITL pattern our excellent checkpoint substrate almost supports |
| Delta/incremental checkpoint storage for APPEND mode | **Later** | L | LangGraph's DeltaChannel (their explicit 1.2 focus); full-snapshot-per-save is quadratic for exactly the long-running agents our durability story attracts |

## Theme 7: Evaluation & HITL Ergonomics

Evaluation is a critical gap for a framework whose product is agents; HITL is a strength with one ergonomic flaw worth fixing before competitors weaponize it. (Note: our RAG metric suite is genuinely good — the gap is everything *around* it: trajectories, datasets, CI.)

| Item | Horizon | Effort | Rationale |
|---|---|---|---|
| Trajectory evaluation: a trace/trajectory schema fed from our existing tracing Run tree, plus tool-selection, parameter-quality, and task-success judges | **Now** | L | ADK's 13 metrics and CrewAI's trajectory judges; output-only RAG metrics miss what users most need to evaluate — whether the agent took the right actions |
| Retry policy upgrade: `retry_on` exception filters + jitter on ErrorHandling | **Now** | S | LangGraph/ADK/LangChain all have it; retrying auth errors with deterministic backoff burns spend and duplicates side effects — small fix, real production pain |
| Dataset/experiment primitives: `EvalSet`/`EvalCase` models, an ExperimentRunner that executes cases against a workflow and persists results | **Next** | M | ADK EvalSet and CrewAI ExperimentRunner; hand-assembled parallel Python lists are not an eval workflow |
| pytest integration + baseline comparison for CI regression (`assert_experiment` helpers, score thresholds) | **Next** | M | ADK's AgentEvaluator pytest entry point; "how do we put agent quality under CI" is a common deal-breaker question |
| Suspend-and-return HITL: a `PENDING` run status returning a typed pending-approval payload immediately, decision passed as an argument to resume — replacing blocked-thread waits | **Next** | L | LangGraph `interrupt()`/`Command(resume)` is the natural web-server shape; our timeout→checkpoint→re-prompt round-trip works but reads as legacy in comparisons |
| `dynamiq eval` CLI command; results table in the dev UI (Theme 3) | **Later** | S | ADK `adk eval` / CrewAI `crewai test`; rides on the runner and UI once they exist |
| Conditional approval policies: `when` predicates on ApprovalConfig gating by tool-call arguments | **Later** | S | LangChain's per-tool predicates; "approve only payments above $X" currently means approve everything |

## Theme 8: Hardening & Trust Fundamentals

| Item | Horizon | Effort | Rationale |
|---|---|---|---|
| SSRF protection on HttpApiCall and web tools: resolved-IP validation, private/reserved-range blocking, redirect credential stripping, DNS pinning | **Now** | S | LangChain/CrewAI/DeepAgents all ship it; a prompt-injected agent steering into cloud metadata endpoints is a concrete VPC risk with our `follow_redirects=True, trust_env=True` defaults |
| Local Docker container sandbox backend (self-hosted alternative to E2B/Daytona) | **Next** | M | ADK ships container/GKE/local code executors and DeepAgents has a 5-vendor sandbox ecosystem; our sandbox tier currently *requires* a paid cloud vendor — a hard blocker for air-gapped enterprise |
| Local FilesystemFileStore (+ S3 later) as an artifact/offload backend so oversized-output offload and compaction-evicted history work without a remote sandbox | **Next** | M | DeepAgents' backend protocol and ADK's artifact services; our differentiated offload story currently degrades to destructive truncation in default local deployments |
| Encrypted checkpoint serializer (AES-GCM, pluggable cipher) | **Later** | S | LangGraph EncryptedSerializer; a compliance checkbox in regulated deals where agent state contains conversation content and approval responses |
| Long-term-memory intelligence layer: importance scoring, recency-decay composite recall, background consolidation, TTL — on our existing 5-backend substrate | **Later** | L | CrewAI's scored/consolidated memory headline; we have the best OSS substrate and the least intelligence on top of it |

---

## What NOT to build (or consciously defer)

- **Realtime bidirectional voice/video runtime.** Only ADK has it, it's welded to Gemini Live, and it's an XL+ platform investment. Revisit in 2027 only if a provider-agnostic realtime layer can be scoped as a wedge; do the cheap adjacent fix (audio input blocks, Theme 5) instead.
- **Native provider SDK rewrite.** CrewAI's six-SDK re-architecture is not worth copying. Stay on LiteLLM; do *targeted* native escapes only where LiteLLM blocks a roadmap item (Responses API, thinking-block fidelity).
- **Event-driven flow engine, cron scheduling, and trigger endpoints in OSS.** This is the deliberate OSS/platform boundary — the paid platform's core job. Ship `serve` so users have a webhook-able surface, and stop there; don't chase CrewAI's Flow event engine.
- **A2UI / generative-UI component protocol.** CrewAI and LangGraph are still competing to define this space; premature to pick. Publishing our typed streaming event schema + a thin TS client (Theme 3) captures most of the value.
- **A full JS/TS SDK port.** LangChain.js/LangGraph.js exist and ADK has Java/Go, but a second runtime is a company-scale commitment. The TS *client* for serve endpoints (Theme 3) is the wedge; revisit based on pull.
- **Multi-turn user simulation and GEPA-style prompt optimization.** Real trends (ADK ships both), but they sit atop an eval stack we don't have yet. Sequence strictly after Theme 7's runner and trajectory metrics.
- **Chat-channel connectors (Slack/WhatsApp/Telegram) and in-process local embedding models.** Community-PR shapes. For local embeddings, document the OpenAI-compatible `url` override against Ollama/vLLM (near-zero effort) rather than shipping torch.
- **Third-party framework *agent* adapters** (running LangGraph/OpenAI-Agents agents inside Dynamiq). Tool-level adapters (Theme 4) capture most migration value; embedding foreign runtimes is high-maintenance surface for low strategic return.

## Defending our strengths

- **Durability is our best story — extend it before LangGraph closes in.** Mid-loop resume, pending-tool-call replay, cancel-to-resumable-run, and checkpoint-on-failure-by-default beat LangGraph's node-level semantics today, but their 1.2 line (delta channels, crash-resumable error handlers) is aimed straight at us. Priority: delta storage and the edit API (Theme 6), then *market it* — publish a durability comparison benchmark, because right now nobody knows we win here.
- **YAML round-trip + workflow generation is a moat against the LangChain family**, which has no declarative path at all. Make YAML the artifact format for the dev UI, eval sets, and A2A agent cards so every new feature compounds it; the visualization item makes it visible.
- **Shared sandbox/browser sessions and browser-takeover HITL are unique in the market.** Feature them in flagship examples and the dev UI live-view; they demo spectacularly.
- **Cost observability**: we're the only OSS framework with per-call USD tracking. Theme 2's aggregation helper plus OTel export turns a hidden internals feature into a headline — "know what every agent run costs, in any backend, for free" — landing exactly on 2026's cost-control anxiety.
- **Deepest OSS RAG stack of the six.** Defend by fixing the two credibility gaps that undercut it in evaluations: structured citations (a `Citation` model with source offsets validated from retriever provenance — LangChain sets the bar) and incremental indexing on our existing deterministic chunk IDs. Both Next-horizon, M-effort.
- **Streaming leadership** at the event level (tool-input deltas, bidirectional input) is a generation ahead in *production* and a generation behind in *consumption*. The serve command, published schema, and TS client convert an invisible strength into a visible one competitors can't match: approval flows and browser takeover over the same live stream.
- **Trust posture**: no telemetry phone-home, fully-functional OSS, Apache-2.0. Say it on the README.

## Beyond code: gaps the audit surfaced that engineering can't fix alone

- **Community concentration**: 8 all-time commit authors vs. hundreds for LangChain/CrewAI. The conformance kits, instrumentor guide, and middleware API in this roadmap are specifically the community-leverage shapes — each creates a surface outsiders can contribute to.
- **Benchmark absence**: DeepAgents ships in-repo benchmark suites (tau2, harbor) and publishes model-harness results; we publish none. The durability benchmark and trajectory evals give us two credible entries.
- **Version signal**: pre-1.0 at v0.59 with no stability policy (Theme 3 item).
- **Docs debt**: in-repo docs are thin relative to 273 examples, and the mkdocs build is currently broken (Theme 3 item).

## Suggested sequencing (the "Now" column in one view)

Quarter-scale view of the Now items: the middleware runtime (Theme 1) is the long pole and starts immediately; OTel + integrations + `py.typed` + console handler (Theme 2) and SSRF + retry fixes (Themes 7/8) are small, independent, and shippable in weeks; `init`/`run`/`serve` + docs + 1.0 plan (Theme 3) build the evaluation-winning surface; `to_mcp_server` + MCP OAuth (Theme 4) ride on `serve`; thinking-output preservation (Theme 5) and trajectory evals (Theme 7) round out the quarter. Everything in **Next** assumes the middleware runtime and `serve` exist — those two are the load-bearing investments.
