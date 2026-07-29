# PRD: Seamless chat stream resume (Claude/ChatGPT-style continuity for Super Agent)

| | |
|---|---|
| **Status** | Proposal — research complete, validated against code in `ui`, `mono`, `catalyst`, `runtime`, `dynamiq` |
| **Date** | 2026-07-29 |
| **Scope** | `/chat` (Super Agent / LLM chat) in `dynamiq-ai/ui`, conversations API in `dynamiq-ai/mono` (nexus) |
| **TL;DR** | The backend already persists every chat event with a monotonic `sequence` and the stream endpoint already accepts `after_sequence` — the UI just never uses either. Fix is ~90% frontend + one ~5-line additive backend field. |

---

## 1. Problem

Super Agent conversations run server-side, so a user can start a run on their phone and pick it up on desktop — the execution infrastructure for that already works. The client experience around it doesn't:

- **Switching browser tabs restarts the stream.** Even a momentary tab switch on desktop aborts a perfectly healthy SSE connection. On return, the UI clears the in-progress assistant message, refetches the entire conversation (all messages *with* full event history, page size 500), and re-opens the stream **from event 1**, visibly replaying everything that was already on screen.
- **Reload / second device mid-run replays from zero.** Opening a conversation that is mid-stream renders nothing for the in-flight message, then replays the whole event stream from the beginning instead of showing what already happened and tailing live.
- **Every reconnect re-downloads and re-renders all events**, which for long agent runs (many tool calls, token deltas) is slow, janky, and wasteful for both client and server.

Target behavior — what Claude and ChatGPT do: returning to a conversation (tab switch, refresh, another device) instantly shows everything generated so far and the stream continues *from that point*, with no replay flicker and no duplicated content.

## 2. How it works today (verified end-to-end)

```mermaid
sequenceDiagram
    participant UI as ui (chatStore)
    participant NX as mono/nexus (Go)
    participant CAT as catalyst (FastAPI)
    participant RT as runtime (dynamiq executor)
    participant JS as NATS JetStream
    participant PG as Postgres

    UI->>NX: POST /v3/conversations/{id}/messages
    NX->>CAT: CreateConversationMessageV2
    CAT->>JS: publish workflow YAML → conversations.runtimes.v{ver}.queue
    RT->>JS: pull queue, run dynamiq Workflow
    RT->>JS: publish events → conversations.{cid}.messages.{mid}.events<br/>(envelope: id, type, sequence 1..N, timestamp, data, source)
    JS-->>NX: ephemeral consumer per HTTP stream
    NX-->>UI: SSE (POST response, or GET .../stream)
    JS-->>NX: durable consumer "chat-message-events-processor"
    NX->>PG: insert conversation_events (per-event row, sequence column)<br/>update conversation_messages.status
```

Key facts, with sources:

1. **Every event has a monotonic per-message `sequence` (1, 2, 3…).** Assigned by the runtime when wrapping dynamiq streaming output into the event envelope, including HITL events which share the same counter. → `runtime/app/services/nats_chat.py` (`_execute_workflow`, `_publish_streaming_events`, `_hitl_input_listener`)
2. **Events are durably stored twice:** JetStream retains the full stream for **7 days** (`mono/services/nexus/internal/features/conversations/conversations.go`), and nexus persists every event into the `conversation_events` table keyed by event ID with a `sequence` column (`.../conversations/service/message_event.go`).
3. **The resume API already exists.** `GET /v1/conversation-messages/{message_id}/stream?after_sequence=N` replays only events with `sequence > N`, in strict order, and closes after the terminal event. Same for agent runs. → `.../conversations/handler/handler.go` (routes + `streamConversationMessageEventsQuery`), `.../service/message.go` (`consumeConversationEvents`)
4. **The SSE payload already carries `sequence`** — each `data:` line is the full envelope `{id, type, sequence, timestamp, data, source}` → `mono/internal/types/api/stream.go`, `.../service/types.go`. The server also sends a heartbeat comment every **15s**.
5. **Message status is tracked server-side** (`streaming` → `completed`/`failed`/`canceled` on terminal events), and `GET /v1/conversations/{id}/messages` embeds each message's persisted events (ordered by sequence) — this is how completed messages already re-render tool calls after a refresh.
6. **`dynamiq` and `runtime` need no changes.** Streaming originates from `dynamiq/callbacks/streaming.py` (`AsyncStreamingIteratorCallbackHandler`); sequencing/persistence live above the framework. The runtime even has an established sequence-continuity precedent (checkpoint `run.paused`/`run.resumed` on the apps path).

## 3. Root causes (all in `dynamiq-ai/ui`)

| # | Cause | Where |
|---|---|---|
| 1 | On **every** `visibilitychange → visible`, the UI aborts the active stream and wipes in-progress state (`messagePart`, `thinkingText`, `tools`) — even when the connection is healthy. The hook was written for iOS Safari killing background connections, but `visibilitychange` also fires on every desktop tab switch. | `src/pages/chat/components/Messages/hooks/useVisibilityReconnect.ts` |
| 2 | Reconnection always calls `GET /v1/conversation-messages/{id}/stream` **without `after_sequence`**, so the server replays from event 1. | `src/stores/chatStore/actions/streamConversationMessage.ts`, `reconnectStream.ts` |
| 3 | The client never reads the `sequence` field it already receives in every SSE event, so it has no cursor to resume from and no duplicate protection. | `src/stores/chatStore/actions/processLLMStream.ts` |
| 4 | On load of a mid-stream conversation, the streaming message's already-persisted events (present in the messages response) are **discarded** (`status === 'streaming'` messages are filtered out), and the full replay is used to rebuild the UI instead. | `src/pages/chat/components/Messages/hooks/useSyncMessages.ts`, `useStreamingReconnect.ts` |
| 5 | `reconnectStream` explicitly clears partial content before re-streaming ("clear partial content before re-streaming") — necessary today *because* of the full replay, but it's what causes the visible reset. | `src/stores/chatStore/actions/reconnectStream.ts` |

One backend gap: the messages list embeds events as **raw inner payloads only** — the envelope's `sequence` is stripped (`mono/services/nexus/internal/api/types.go`, `ConversationMessageFrom`). So after a fresh load the UI can render persisted events but cannot know which sequence to resume after. (The standalone `GET /v1/conversation-messages/{id}/events` endpoint *does* return sequences, but using it would add an extra paginated request per reconnect.)

## 4. Proposed changes

### P0 — resume instead of replay (the actual fix)

**mono (nexus), ~5 lines, backward compatible:**

- **M1.** Add `last_event_sequence` to the message DTO in `ConversationMessageFrom` — the max `Sequence` of the loaded events (they're already loaded, ordered). This gives the client its resume cursor for the fresh-load case.

**ui:**

- **U1. Track the cursor.** In `chatStore`, add `lastEventSequence` and `lastStreamActivityAt`. In `processLLMStream`: update `lastStreamActivityAt` on every reader chunk (heartbeats count); read `parsed.sequence` from each envelope, **skip any event with `sequence <= lastEventSequence`** (dedupe safety for at-least-once delivery), then advance the cursor.
- **U2. Stop killing healthy streams.** In `useVisibilityReconnect`: on becoming visible, if `isAnswering` and the stream showed activity within the staleness threshold (suggest **45s** = 3 missed heartbeats), do nothing. Only when stale (backgrounded long enough for the browser/OS to have killed the connection — the original iOS case): abort and fall through to reconnect. This alone fixes the common desktop tab-switch complaint and also removes the heavyweight messages refetch on every tab switch.
- **U3. Resume with the cursor.** `streamConversationMessage` and `reconnectStream` append `?after_sequence=<lastEventSequence>` when a cursor exists, and **stop wiping partial content** when resuming with a cursor — new events append to the existing `messagePart`/`thinkingText`/`tools` state.
- **U4. Seed from persisted events on fresh load.** In `useSyncMessages`/`useStreamingReconnect`: for a message with `status === 'streaming'`, run its embedded `events` through the existing `parseMessageEvents` (same code path completed messages use) to seed the store's partial state, set `lastEventSequence = message.last_event_sequence`, then open the stream with `after_sequence`. Result: instant render of everything generated so far + live tail, on refresh and on a second device.
- **U5. Robustness edge.** Reconnect for any *non-terminal* status, not just `streaming` — a message still queued (`created`) before `run.started` currently falls through the reconnect check entirely.

No changes to `catalyst`, `runtime`, or `dynamiq` for P0.

### P1 — polish and hardening (small, independent)

- **SSE `id:` field.** Emit the envelope sequence as the SSE `id:` in `api.StreamEvents`, and optionally accept a `Last-Event-ID` header as an alias for `after_sequence`. Aligns with the SSE standard and unlocks off-the-shelf clients (e.g. `@microsoft/fetch-event-source`) later.
- **Server replay efficiency.** `consumeConversationEvents` uses `DeliverAllPolicy` and skips client-side, so each reconnect still scans the subject from the start server-side. Bounded per message, so acceptable — but `DeliverByStartTime` (timestamp of the last-seen event) is a cheap optimization for very long runs.
- **Slimmer conversation fetch.** `GET /v1/conversations/{id}/messages` (page size 500) embeds the full event history of every message. Consider lazy-loading events per message, or embedding them only for the streaming/last message and letting completed messages fetch on demand.

### P2 — full cross-device "live" parity (separate project, optional)

- **Conversation-level event stream** (`GET /v1/conversations/{id}/stream`): bridge `message.created` + per-message events so a conversation open in another tab/device picks up *new* messages in real time, not just on refocus/refetch. The NATS subject layout (`conversations.{cid}.messages.*.events`) already supports this with a wildcard consumer.
- **Cross-tab stream dedupe** (SharedWorker/Web Locks) so N tabs of the same conversation share one connection.

## 5. Acceptance criteria

1. Desktop: start a run, switch tabs briefly (<45s), return → no visual reset, no messages refetch, zero additional `/stream` requests.
2. Tab backgrounded long enough for the connection to die → on return the UI resumes with `after_sequence=<last seen>`; already-rendered content does not flicker or duplicate; only new events arrive (verify in devtools).
3. Hard refresh mid-run, or opening the conversation on a second device → partial content renders immediately from persisted events; stream tails live from `last_event_sequence`; no replay from event 1.
4. iOS Safari background/foreground mid-run → same as (2) (this was the original reason for the visibility hook — must not regress).
5. Run completes/fails/cancels while the tab is hidden → on return the final message renders once, correctly (existing terminal-status handling preserved; a terminal event *before* the resume cursor closes the stream immediately server-side).
6. HITL: pending human-feedback / approval prompt survives a refresh mid-wait (prompt is re-rendered from persisted events) and answering it still works.
7. Duplicate delivery from NATS (at-least-once) never renders twice — cursor dedupe in `processLLMStream`.

## 6. Effort estimate

| Work | Estimate |
|---|---|
| mono: `last_event_sequence` field (+ optional SSE `id:`) | 0.5–1 day |
| ui: U1–U5 + unit tests (hook tests already exist, e.g. `useSyncMessages.test.ts`) | 3–4 days |
| QA across Chrome/Safari/iOS + multi-device | 1–2 days |

## 7. Open questions

1. Staleness threshold for "the stream is probably dead" — proposal: 45s (3× the 15s server heartbeat). Alternatively reconnect-on-any-hidden->visible after >60s hidden.
2. Keep the unconditional messages refetch on refocus for *data freshness* (titles, new messages from other devices) even when the stream is healthy? Proposal: no for the chat body (P2 solves it properly); the conversation list sidebar can keep its own cheaper invalidation.
3. JetStream retention is 7 days — fine for chat runs, but confirm no product plans for pausable runs longer than that (persisted events in Postgres cover rendering either way; only live-resume via JetStream is bounded).
4. Is multi-tab double-streaming (one SSE per tab of the same conversation) acceptable server load for now? (Each is an ephemeral consumer; P2 dedupe removes it.)

---

*Appendix — primary code references:*
`ui`: `src/pages/chat/components/Messages/hooks/{useVisibilityReconnect,useStreamingReconnect,useSyncMessages}.ts`, `src/stores/chatStore/actions/{processLLMStream,streamConversationMessage,reconnectStream,sendLlmMessage}.ts`, `src/pages/chat/components/Messages/parseMessageEvents.ts` ·
`mono`: `services/nexus/internal/features/conversations/{handler/handler.go,service/message.go,service/message_event.go,consumer/message_event.go,conversations.go}`, `internal/types/api/stream.go`, `services/nexus/internal/api/types.go` ·
`catalyst`: `app/services/conversations/agent_conversations.py` ·
`runtime`: `app/services/nats_chat.py` ·
`dynamiq`: `dynamiq/callbacks/streaming.py`
