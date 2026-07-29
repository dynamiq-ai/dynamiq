# Claude Code–style AskUserQuestion for the Super Agent — research & surgical changes

Cross-repo research covering `dynamiq` (framework), `catalyst`, `runtime`, `mono` (nexus/synapse), `ui`, and `charts`.
Goal: assess what it takes to support **multi-option questions, several questions per round-trip, sequential
rounds, and long pauses (1–24 h+)** in the `/chat` super agent, similar to Claude Code's `AskUserQuestion`.

## TL;DR

The plumbing is ~80 % there. Transport, per-node correlation, sequential Q&A, mid-run page refresh, and — on
the **apps path** — the full *timeout → checkpoint → `run.paused` → resume-with-replay* loop already exist and
work. Four things are missing, and all four are surgical rather than architectural:

1. **A structured question schema.** Today an ask is one plain string out (`prompt`) and one plain string back
   (`content`). Options, multi-select, and batching of several questions have no representation anywhere.
2. **Pause/resume on the chat path.** The chat runtime handler never wires checkpoints, so an unanswered
   question hard-fails the turn after **600 s** (`run.failed`, turn lost). The identical handler for apps
   already solves this; the block needs porting, plus a resume dispatcher on the backend.
3. **UI question form + reload durability.** The current "ask-user" renderer is a 12-line static box; answers
   are typed into the composer. A pending question does not survive a page reload even though the ask event
   *is* persisted server-side — the history parser simply drops it.
4. **A per-ask `request_id`.** Replies are matched only by node id. Combined with JetStream
   `DeliverPolicy.ALL` replay, a stale answer from round 1 can satisfy round 2 after a resume, and mono's
   deterministic request key collides for sequential questions from the same node.

No new services, queues, or storage systems are needed. NATS subjects, event persistence, checkpoint storage,
and the resume protocol all exist.

---

## 1. How it works today (verified end-to-end)

### Chat (super agent) path

```
UI (SSE)                nexus (Go)                 catalyst (Py)              runtime (Py)
POST /v3/conversations/{id}/messages
  └─> SendConversationMessageNew ──HTTP──> POST /v2/conversations/{id}/messages
                                             build super-agent Workflow (YAML)
                                             publish ChatYamlRunRequest ──NATS──> dynamiq.chat.conversations.runtimes.v{ver}.queue
                                                                                  ack immediately, workflow.run_async in-process
UI <──SSE── nexus <──JetStream── dynamiq.chat.conversations.{conv}.messages.{msg}.events (durable consumer also persists → conversation_event)
UI ──POST /v1/conversation-messages/{msg}/input──> nexus ──publish──> ...messages.{msg}.inputs ──> runtime per-node queue → tool unblocks
```

- Super agent: `catalyst/app/services/conversations/agent_conversations.py:789-1142` — Agent id `super-agent`,
  `max_loops=250`, `parallel_tool_calls_enabled=True`, E2B sandbox, flat `Input → Agent → Output` workflow.
- HITL tools (`:857-885`): `ask-user` and `browser-takeover`, both `HumanFeedbackTool` with
  `input_method=STREAM`, `ErrorHandling(timeout_seconds=600, behavior=RAISE)`. The `ask-user` description
  explicitly says *"Prefer a single focused question over multiple questions at once"* and *"The user can only
  provide text responses"* (`:157-182`) — the prompt-side mirror of the missing schema.
- Ask wire format (`dynamiq/nodes/tools/human_feedback.py:24-39`): out `{prompt: str, action, is_browser_takeover}`,
  back `{content: str}`. The tool blocks in `get_input_streaming_event` polling a per-node `queue.Queue`
  (`dynamiq/nodes/node.py:1530-1602`), default timeout **600 s** (`dynamiq/types/streaming.py:210`).
- Runtime chat handler (`runtime/app/services/nats_chat.py`):
  - `build_hitl_nodes_override` (`:1037-1084`) gives **each HITL node its own queue** (no cross-node races) but
    passes **no `timeout`** → 600 s always, regardless of YAML.
  - `_hitl_input_listener` (`:1145-1312`) subscribes `...messages.{msg}.inputs` with `DeliverPolicy.ALL`
    (`runtime/app/core/clients/nats.py:180`), routes by `data.entity_id` into the node's queue, and publishes a
    `run.human_feedback.received` event. Unknown `entity_id` creates a fresh queue on the fly — i.e. a
    mis-addressed answer is silently black-holed (`:1174-1182`).
  - **Zero checkpoint references in this file.** On timeout: `InputStreamingTimeoutError` → `run.failed`. The
    turn is lost; nexus marks the message `failed`.
- nexus input endpoint is a blind passthrough: authz check, then publish the opaque body to `.inputs`
  (`mono/services/nexus/internal/features/conversations/service/message.go:204-216`). No registry, no
  validation, no status change.
- Every event — including the HITL ask — **is persisted** to `conversation_event`
  (`service/message_event.go:52` runs unconditionally), and stream replay
  (`GET /v1/conversation-messages/{id}/stream?after_sequence=N`) rebuilds the full run. That's why a page
  refresh **during** the 600 s window correctly re-renders the pending question (the UI re-parses the replayed
  events through `processLLMStream`), but a return after timeout finds a `failed` message and nothing pending.

### Apps path (deployed agents) — the pattern to copy

`runtime/app/services/nats.py:492-538` wires `CheckpointConfig(enabled, backend=nexus /v1/checkpoints,
behavior=REPLACE, checkpoint_on_input_timeout_enabled=True, resume_from=…)`. On timeout the framework saves a
`PENDING_INPUT` checkpoint capturing the agent's **entire conversation, loop counter, and the in-flight tool
call** (`dynamiq/nodes/agents/checkpoint.py:26-137`); the runtime converts the failure into
`run.paused(checkpoint_id)` and releases the KV dedup lock (`nats.py:596-621`). Synapse marks the run `paused`,
keeps a pending `app_run_input_request` row (Postgres, no TTL), and when the answer arrives **days later**,
`SendRunInput` re-dispatches a `RunStartMessage{resume_from}` and then publishes the answer to `.inputs`
(`mono/services/synapse/.../run/service.go:439-484`). The resumed agent **replays the exact ask without
re-calling the LLM**, the listener replays the already-published answer off the stream (`DeliverPolicy.ALL`),
and the run continues. There is no server-side answer deadline. This is precisely the 24-hour story — already
built, tested (`runtime/clients/checkpoints/nats_demo.py`), and running for apps.

### The target (Claude Code's AskUserQuestion, for reference)

One tool call carries 1–4 questions; each has `question`, a short `header`, `multiSelect`, and 2–4 options
(`label` + `description`); the UI renders clickable chips with a free-text "Other" escape hatch; all answers
come back in **one** submission; the agent can follow up with another round. That's the UX bar.

---

## 2. Gap analysis

| Capability | Today | Gap |
|---|---|---|
| Single free-text question | ✅ works end to end | — |
| Question with predefined options | ❌ no schema anywhere | Framework event + UI + answer payload |
| Several questions, one round-trip | ❌ N sequential asks (HF tool is not parallel-eligible — `dynamiq/nodes/node.py:280` — so batched LLM calls serialize; safe but slow) | Put `questions[]` in **one** ask call/event |
| Sequential rounds | ⚠️ works live; collision risks (below) | `request_id` |
| Pause > 600 s (chat) | ❌ `run.failed`, turn lost | Port apps checkpoint block + resume dispatch |
| Pause > 600 s (apps) | ✅ pause/resume via checkpoint | — |
| Refresh during wait | ✅ full replay re-renders question | — |
| Reload after long pause | ❌ UI drops HITL events from history (`ui/src/pages/chat/components/Messages/parseMessageEvents.ts` has no HF branch) even though rows exist | UI history branch + pending-state derivation |
| Answer while disconnected | ✅ run continues with nobody connected (durable consumers) | — |

**Correctness issues found while cross-checking** (worth fixing in the same effort):

1. **Stale-answer replay.** The inputs listener uses `DeliverPolicy.ALL` over a 7-day stream
   (`DYNAMIQ_CHAT`/`DYNAMIQ_APPS`, `MaxAge: 7d`). On resume (or listener restart) *every* prior answer for that
   message replays into the per-node queue; `get_input_streaming_event` filters only by event name
   (`node.py:1589-1598`), so a round-1 answer can satisfy a round-2 question. A `request_id` echoed
   through the reply and checked at the consumer fixes it (and makes the resume race safe *by design* instead of
   by replay accident).
2. **mono request-key collision (apps path).** `requestStore.GenerateKey = uuid.Hash(WfRunID + ":" + EntityID)`
   (`synapse/.../run/request_store.go:130-134`) is deterministic and duplicate-swallowing — two sequential
   questions from the same node collide into one row. Include the `request_id` in the key.
3. **UI treats `info` as a question.** `processLLMStream.ts:126-152` keys on `source.type == HumanFeedbackTool`
   without checking `data.action`, so a fire-and-forget `info` message transiently renders "Waiting for your
   answer". One-line fix (`action !== 'info'`).
4. **Hard-kill orphan (both paths).** Queue messages are acked at dispatch; if a pod dies mid-wait *before* the
   600 s timeout, there is no `run.paused`, the `processing.{id}` KV lock is never released (7-day bucket TTL),
   and the run stays `started` forever — mono has no reaper (`ReasonTimeout` is declared but unused). Worth a
   sweeper, but orthogonal to this feature.
5. **Ops blind spots.** Runtime HPA scales on CPU only (parked runs are invisible); checkpoint load on resume is
   a synchronous `requests.get` **on the event loop with no timeout** (`nats.py:499`,
   `runtime/app/core/clients/nexus.py:312-370`); self-hosted chart ships no `terminationGracePeriodSeconds`
   (30 s default) vs 1800 s in the SaaS kustomize overlays.

---

## 3. Surgical changes, by repo

Ordered so each stage is independently shippable. (1)+(2)+(5) alone deliver multi-question/options with the
existing 600 s window; (3)+(4) add the long-pause story.

### 3.1 `dynamiq` framework — structured questions + `request_id` (small)

`dynamiq/nodes/tools/human_feedback.py`:

```python
class QuestionOption(BaseModel):
    label: str
    description: str | None = None

class Question(BaseModel):
    id: str | None = None            # defaults to index
    header: str | None = None        # short chip label, e.g. "Auth method"
    question: str
    options: list[QuestionOption] = []
    multi_select: bool = False
    allow_custom_answer: bool = True

class HumanFeedbackInputSchema(BaseModel):
    action: HumanFeedbackAction = ASK
    input: str = ""                              # kept: plain ask / fallback text
    questions: list[Question] | None = None      # NEW: structured batch (1–4)
```

- `HFStreamingOutputEventMessageData` += `questions: list[Question] | None`, `request_id: str`
  (uuid minted per ask). Keep `prompt` populated (rendered fallback) so Slack/Telegram/old UIs degrade to text.
- `HFStreamingInputEventMessageData` += `answers: list[Answer] | None`, `request_id: str | None`
  (`Answer{question_id, selected: list[str], custom_text: str | None}`); `content` stays for plain replies.
- In `input_method_streaming`, loop until the reply's `request_id` matches (or is absent, for
  backward compat) — discard non-matching messages instead of consuming them. This kills the stale-replay bug at
  the layer that owns the queue.
- Format structured answers into a readable string for the LLM observation
  (`"Q: …\nA: label1, label2 (other: …)"`) and also return the raw `answers` in the tool output dict.
- `to_dict`/YAML round-trip: the new fields are plain pydantic — `WorkflowYAMLDumper` picks them up for free.

Nothing else in the framework needs to change: checkpoint-on-input-timeout, pending-tool-call replay, and
per-node queues already behave correctly for the batched form (one ask call = one event = one reply).

### 3.2 `catalyst` — teach the agent (tiny)

- Bump the `dynamiq` pin (currently `0.59.0`, `pyproject.toml:9`).
- Rewrite `ASK_USER_TOOL_DESCRIPTION` (`agent_conversations.py:157-182`): allow up to N questions per call,
  prefer options when the choice space is finite, keep free-text for open questions. Drop *"Prefer a single
  focused question"* and *"text responses only"*; add guidance mirroring AskUserQuestion (headers ≤ 12 chars,
  2–4 options, mark recommended option first).
- No transport changes — the YAML serialization and NATS dispatch are schema-agnostic.

### 3.3 `runtime` — port pause/resume to chat + pass structured payloads (medium)

`app/services/nats_chat.py` (mirroring `nats.py:492-538` and `:596-621`):

- Add `checkpoints: CheckpointsConfig` and `resume_from: ResumeFrom | None` to `ChatYamlRunRequest`
  (`app/schemas/nats.py:166-177`) — the apps `RunRequest` already has both.
- Wire `CheckpointConfig(enabled, backend=AppCtx.checkpoint_backend, behavior=REPLACE,
  checkpoint_on_input_timeout_enabled=True, resume_from=…)` into the run config; on FAILURE with
  `CheckpointStatus.PENDING_INPUT` → publish `run.paused(checkpoint_id)` + `kv_delete("processing.{message_id}")`
  instead of `run.failed`; emit `run.resumed` on resume. (Same for `nats_agents.py` later if async agent runs
  should ask questions — today they run with `human_feedback_enabled=False`.)
- `_hitl_input_listener` (`:1187-1215`): pass `answers`/`request_id` from the input payload through to
  `HFStreamingInputEventMessageData`, and include them in the `run.human_feedback.received` event so the
  answer is auditable/persisted.
- Fix while there: make the checkpoint backend async or `asyncio.to_thread` it, with a timeout (`nexus.py:312`).

The 600 s in-memory wait becomes a **fast path**, not a cliff: answered quickly → zero extra latency; otherwise
the run parks durably. Keeping 600 s is reasonable; lowering it (e.g. 180–300 s) frees the semaphore slot,
thread, and E2B sandbox sooner at the cost of a slower resume for medium waits.

### 3.4 `mono` (nexus) — paused status + resume dispatch (medium; the biggest piece)

- Consumer (`service/message_event.go`): handle `run.paused` → new `conversation_message` status `paused`
  (enum + migration) and `run.resumed` → back to `streaming`. (Unknown types are already persisted, so the
  event row itself needs no schema change; the `checkpoint_id` can live inside the event `data` and be read
  back synapse-style — no new column strictly required.)
- Input endpoint (`service/message.go:204-216`): if the message is `paused`, trigger a resume **before**
  publishing the answer — mirror synapse's `SendRunInput` (`run/service.go:439-484`): find the newest
  `run.paused` event's `checkpoint_id`, re-dispatch, then publish. `DeliverPolicy.ALL` makes the
  answer-before-listener ordering safe.
- **Who re-dispatches:** the chat queue payload embeds `workflow_yaml_config`, which mono never sees. Two options:
  - **(a) Recommended:** catalyst persists the serialized YAML at first dispatch (S3 via its existing storage
    client, or a column on its `conversation_message`) and exposes
    `POST /v2/conversations/{id}/messages/{message_id}/resume {checkpoint_id, sequence}` which republishes the
    *original* `ChatYamlRunRequest` + `resume_from`. Nexus calls it from the input endpoint. Exact-workflow
    fidelity; node ids (`input`/`super-agent`/`output`) are stable constants so checkpoint restore maps cleanly.
  - (b) Catalyst rebuilds the workflow fresh on resume — less storage, but skills/KB drift between dispatch and
    resume can change the toolset under a replayed tool call.
- Optional (recommended for product robustness, not required for v1): a `conversation_input_request` registry
  mirroring `app_run_input_request`, giving pending questions a queryable identity (badge counts, Slack
  deep-links, multi-device). For v1, pending state is fully derivable from the event log:
  *last HF ask event with no later `run.human_feedback.received` for the same `request_id` and message not
  terminal*.
- UI-facing: expose message status `paused` in the message list/stream APIs (it already flows as an SSE event).

### 3.5 `ui` — the question form + durability (medium)

- **Form:** grow `AskUserFeedback.tsx` (currently 12 lines) into an options renderer: per-question chip rows
  (single/multi-select), free-text "Other" per question, one submit for the whole batch. `ApprovalComponent`
  (Formik form inside a message) and `SamplePrompts` chips are existing in-repo precedents. Encode the payload
  as JSON children of the tag (the `<approval>{json}</approval>` technique) instead of more lowercase HTML
  attributes — `buildHumanFeedbackTag.ts` currently interpolates unescaped strings.
- **Submit:** POST the existing `/v1/conversation-messages/{id}/input` with
  `{type: "human_feedback", data: {request_id, entity_id, wf_run_id, event, content: <rendered text>, answers: [...]}}` —
  `content` keeps every downstream consumer working.
- **Durability:** add a `HumanFeedbackTool` branch to `parseMessageEvents.ts` that re-emits the tag from
  history, and derive "still pending" from the event log + message status (`streaming` or `paused`). Extend
  `useStreamingReconnect` to also reconnect after answering a `paused` message (it currently only fires on
  `streaming`).
- Keep the single `humanFeedbackTool` slot — a batch arrives as **one** event, and sequential rounds replace
  the slot naturally. (True parallel asks from different nodes are a multi-agent-flow concern; out of scope.)
- Drive-by: skip `action === 'info'` events when setting the pending state.

### 3.6 Ops / `charts` (small, optional but wise)

- Self-hosted chart: add `terminationGracePeriodSeconds` (SaaS overlays use 1800 s) and document ingress
  idle-timeout annotations for long SSE waits (ALB/nginx default 60 s).
- Consider scaling runtime on queue depth / active-run count instead of CPU once parked runs become normal.
- 7-day JetStream `MaxAge` is fine for 24 h pauses; pending state must live in Postgres (it does, via
  `conversation_event` — and the registry if added), never in the stream.

### Known limitation to decide on: sandbox state across a pause

A checkpoint restores the **conversation** (messages, loop, pending tool call) but not the **E2B sandbox
filesystem**. Both sandbox providers support reconnect-by-id (`dynamiq/sandboxes/e2b.py` / `daytona.py`:
`sandbox_id` field, `close(delete=False)`); the surgical path is to persist `current_sandbox_id` into the
agent's checkpoint state and reconnect on resume, accepting the provider's sandbox-lifetime limits (E2B
pause/persistence for long gaps). Fallback: accept a fresh sandbox on resume and prompt-instruct the agent that
files may need regeneration. Recommend deciding this explicitly rather than discovering it in QA.

---

## 4. Suggested sequencing

1. **PR 1 (framework):** schema + `request_id` echo/filter + answer formatting. Backward compatible; unlocks
   everything else. Fixes the stale-replay class of bugs for apps too.
2. **PR 2 (catalyst + ui):** tool description + question form + history durability. Ships batched multi-option
   questions working within the 600 s window — already a big UX jump.
3. **PR 3 (runtime + mono + catalyst resume endpoint):** chat pause/resume. Ships the 1–24 h story.
4. **PR 4 (mono, optional):** `conversation_input_request` registry + reaper for orphaned runs + request-key fix
   in synapse.

## 5. Open decision points

1. Extend `ask-user` (one tool, `questions[]` optional) vs a separate `ask-user-options` tool. Recommendation:
   extend — one tool keeps the agent's decision surface small, and plain `input` remains valid.
2. YAML persistence for resume: storage-object (a) vs rebuild (b) above.
3. Timeout tuning: keep 600 s fast-path vs shorten to free sandbox/slots sooner.
4. Registry now (PR 4) or derive-from-events indefinitely.
5. Sandbox continuity: reconnect-by-id vs fresh-sandbox-with-instruction.
