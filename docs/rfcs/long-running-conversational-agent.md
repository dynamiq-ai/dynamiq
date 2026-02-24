# RFC: Long-Running Conversational Agent Support

## 1. Problem Statement

The dynamiq Agent is designed around a single-input-per-call model (`input: str`). Building a conversational agent (like Manus AI) that maintains rich, multi-turn context with tool calls requires workarounds. Specifically:

- **No direct message array input**: Users cannot pass `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "...", "tool_calls": [...]}]`. The only path is `input: str` + memory via `user_id`/`session_id`.
- **Message class too narrow**: `Message` (in `dynamiq/prompts/prompts.py`) only has `content: str`, `role`, `metadata`, `static`. No `tool_calls`, `tool_call_id`, or `name` fields -- cannot represent the OpenAI/Anthropic function-calling message format.
- **Summarization is from-scratch each time**: The recent commit `1df12b1a` significantly improved summarization (chunked summaries, `preserve_last_messages`, merge passes), but it still re-summarizes everything from scratch each time. When the context overflows again, the previous summary gets re-summarized (compounding information loss).
- **No reversible compression layer**: There is no "compaction" step (replacing stale tool outputs with short references) before resorting to lossy summarization.
- **Memory only stores flat (role, content) pairs**: The full ReAct trajectory (Thought + Action + Observation) is lost when saving to memory -- only the final user input and assistant answer are persisted.

---

## 2. Current Architecture

### 2.1 Agent Input/Output

```
AgentInputSchema (dynamiq/nodes/agents/base.py:129)
```

- `input: str` -- the user query (required, non-empty by default)
- `user_id: str | None` -- for memory lookup
- `session_id: str | None` -- for memory lookup
- `images`, `files`, `metadata`, `tool_params` -- auxiliary fields

There is no `messages` field. Multi-turn requires the Memory system.

### 2.2 Message Format

```
Message (dynamiq/prompts/prompts.py:55)
```

- `content: str` (required)
- `role: MessageRole` (user/system/assistant)
- `metadata: dict | None`
- `static: bool`

Missing: `tool_calls`, `tool_call_id`, `name`. Tool calls are handled separately via `InferenceMode.FUNCTION_CALLING` and never stored in the message itself.

### 2.3 ReAct Loop Message Accumulation

During execution (`_run_agent` in `dynamiq/nodes/agents/agent.py`), messages build up as:

| #   | Role      | Content Pattern                                    | Created By                         |
| --- | --------- | -------------------------------------------------- | ---------------------------------- |
| 0   | SYSTEM    | Full ReAct system prompt                           | `_setup_prompt_and_stop_sequences` |
| 1   | USER      | Original user query                                | `_setup_prompt_and_stop_sequences` |
| 2   | ASSISTANT | `"Thought: ...\nAction: ...\nAction Input: {...}"` | `_append_assistant_message`        |
| 3   | USER      | `"\nObservation: [tool output]\n"`                 | `_add_observation`                 |
| 4   | ASSISTANT | `"Thought: ...\nAnswer: ..."`                      | `_append_assistant_message`        |

This is the content that gets summarized when context overflows.

### 2.4 Current Summarization Flow (post-commit `1df12b1a`)

```
_try_summarize_history (agent.py:1068)
  -> is_token_limit_exceeded() checks ratio or absolute token limit
  -> Invokes ContextManagerTool with messages[_history_offset:-preserve_n]
    -> _summarize_replace_history (context_manager.py:212)
      -> If fits in budget: single-pass LLM summary
      -> If too large: chunk -> summarize each -> merge summaries
  -> _compact_history(summary=tool_result)
    -> Replaces messages with: [prefix] + [summary as Observation] + [last N messages]
```

This is solid for a single summarization pass. The gap is that on the second overflow, the summary from pass 1 is just another message in the middle section that gets re-summarized, with no awareness that it was already a summary.

### 2.5 Memory System

`Memory` (in `dynamiq/memory/memory.py`) is a conversation store with backends (InMemory, Pinecone, Qdrant, PostgreSQL, DynamoDB, Weaviate, SQLite). It stores `(role, content, metadata)` tuples. There is no "genetic memory" or "agentic memory" in the codebase -- this is the only memory system. It is **not suited** for storing rich agent trajectories (tool calls, intermediate reasoning). It is designed for simple user/assistant message pairs.

---

## 3. Competitor Analysis

### 3.1 OpenHands (Condenser System)

- **Rolling window**: keeps first N events (system prompt) + last N events, summarizes middle
- **Incremental**: `Condensation` events are first-class in an append-only event log; summaries accumulate
- **Dual trigger**: automatic (event count > 120) + reactive (on context window API errors)
- **Target size**: after condensation, ~50% of max size (leaves room for growth)
- **Result**: equivalent accuracy, up to 2x cost reduction

### 3.2 Claude Code (Three-Layer Architecture)

- **Layer 1 - Microcompaction**: Large tool outputs saved to disk, only references kept in context. Maintains a "hot tail" (recent results in full) + "cold storage" (older results as file refs)
- **Layer 2 - Auto-compaction**: Triggers based on headroom accounting (reserves space for output + compaction itself)
- **Layer 3 - Manual `/compact`**: User-triggered with optional focus hints
- **Structured summarization prompt**: Checklist-based (user intent, key decisions, files touched, errors/fixes, pending tasks, next step)
- **Post-compaction rehydration**: Re-reads 5 most recently accessed files after summarizing

### 3.3 Manus AI (Three-Pillar Context Engineering)

- **Pillar 1 - Reduce**: Compaction (reversible, full->compact tool outputs) before Summarization (lossy, schema-based)
- **Pillar 2 - Isolate**: Sub-agents get their own context windows for discrete tasks
- **Pillar 3 - Offload**: Filesystem as infinite context window; agent uses grep/glob to retrieve
- **KV-cache optimization**: Append-only context, stable prompt prefixes, explicit cache breakpoints
- **Tool masking**: Tools stay in context but unavailable ones are suppressed via logit masking

### 3.4 OpenCode (Three-Tier Compaction)

- **Pruning**: Removes old tool call results first (cheap, no LLM call)
- **LLM summarization**: Specialized summarization agent compresses remaining history
- **Compaction markers**: `type: "compaction"` parts mark where trimming occurred
- **Buffer**: 20K token buffer + provider max-output reserved
- **DCP plugin**: Zero-LLM-cost plugin using deduplication, superseded-write detection, error purging

### 3.5 Aider (Two-Tier Architecture)

- **Repo map**: Tree-sitter extracts + graph ranking for codebase orientation (token-efficient)
- **Recursive chat summarization**: `ChatSummary` class splits history, recursively summarizes older portions via LLM, preserves recent messages
- **Aggressive defaults**: Only 1-2K tokens for history (rest of budget for repo map + current work)
- **"Done" vs "Current"**: Only completed conversation portions get summarized; current work stays in full

### 3.6 SWE-agent (Observation Summarization + CAT)

- **SimpleSummarizer**: Saves oversized command output to file, shows warning
- **LMSummarizer**: LLM-driven summary of oversized observations, problem-aware
- **CAT framework** (research): Structured workspace with stable task semantics + condensed long-term memory + full-resolution short-term interactions

### 3.7 Letta / MemGPT (Virtual Context Management)

- **Three-tier memory**: Core Memory (in-context, character-limited blocks pinned to system prompt), Recall Memory (full conversation history in DB, searchable), Archival Memory (vector DB for long-term knowledge)
- **Agent-driven management**: The LLM itself decides what to page in/out via tool calls (`core_memory_append`, `core_memory_replace`, `conversation_search`, `archival_memory_search`)
- **Recursive summarization**: When message buffer overflows, oldest ~30% is summarized. If previous summary exists, it's folded in. Sliding window with configurable percentage.
- **Cheap model for compaction**: Summarization can use a separate cheaper model (e.g., gpt-4o-mini)
- **Nothing ever lost**: Complete event history persisted to DB; context window is just a "view"
- **Shared memory blocks**: Memory blocks can be attached to multiple agents for multi-agent coordination

### Key Patterns Across All Competitors

1. **Two-phase compression**: Cheap/reversible first (prune, compact), expensive/lossy second (LLM summarization)
2. **Incremental summaries**: Build on previous summaries rather than re-summarizing from scratch
3. **Structured summarization prompts**: Checklist-based to ensure nothing critical is lost
4. **Preserve recent context**: Always keep the most recent N messages/events in full
5. **Append-only design**: Never mutate earlier messages (KV-cache friendly)

---

## 4. Proposed Changes

### Change 1: Extend `Message` to Support Tool Calls

**Files**: `dynamiq/prompts/prompts.py`

Add optional fields to `Message`:

```python
class Message(BaseModel):
    content: str | None = None  # Allow None for tool_calls-only messages
    role: MessageRole = MessageRole.USER
    metadata: dict | None = None
    static: bool = Field(default=False, exclude=True)
    tool_calls: list[dict] | None = Field(default=None)
    tool_call_id: str | None = Field(default=None)
    name: str | None = Field(default=None)
```

Update `Prompt.format_messages()` to include these fields when non-None, and exclude when None. This mirrors the OpenAI/Anthropic chat completion spec. The `model_dump()` in `count_tokens()` and `format_messages()` must conditionally include these fields.

**Backward compatibility**: All existing code passes `content: str` and gets the same behavior. New fields default to `None` and are excluded from serialization when absent.

**Validation**: Add a model validator -- at least one of `content` or `tool_calls` must be non-None.

### Change 2: Accept `messages` Array in `AgentInputSchema`

**Files**: `dynamiq/nodes/agents/base.py`

Make `input` optional and add a `messages` field:

```python
class AgentInputSchema(BaseModel):
    input: str = Field(default="", description="Text input for the agent.")
    messages: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional pre-built conversation history as list of message dicts "
            "(with keys: role, content, and optionally tool_calls, tool_call_id, name). "
            "When provided, the last user message is used as the current query "
            "and preceding messages are used as history (bypassing memory retrieval)."
        ),
    )
    # ... all existing fields unchanged
```

**Wiring in `BaseAgent.execute()`** (`base.py:529`):

```python
if input_data.messages:
    # Convert dicts to Message objects
    history_messages = [Message(**m) for m in input_data.messages[:-1]]
    last_msg = input_data.messages[-1]
    input_message = Message(**last_msg)
    # Skip memory retrieval -- messages are the source of truth
elif use_memory:
    history_messages = self._retrieve_memory(input_data)
    # ... existing flow
else:
    history_messages = None
```

**Validation**: Add a model validator that ensures either `input` is non-empty OR `messages` is provided (at least one message). When `messages` is provided, the last message must have `role: "user"`.

**Why this matters**: This is the single most important change for building conversational agents. It allows callers to maintain their own conversation state and pass it directly, which is essential for:

- Serverless/stateless deployments where the caller owns the state
- Complex orchestrations where multiple agents share context
- Replaying/debugging specific conversation states

### Change 3: Incremental (Rolling) Summarization

**Files**:

- `dynamiq/nodes/agents/components/history_manager.py`
- `dynamiq/nodes/agents/utils.py` (SummarizationConfig)
- `dynamiq/nodes/tools/context_manager.py`
- `dynamiq/nodes/agents/prompts/react/instructions.py`

**Current problem**: When summarization triggers a second time, the summary from pass 1 is just another message that gets re-summarized. Information compounds with each pass.

**Proposed approach** (combines OpenHands condenser + Claude Code structured summary):

1. Add `_running_summary: str | None` private attribute to `HistoryManagerMixin`:

```python
class HistoryManagerMixin:
    _running_summary: str | None = None
```

2. Add `incremental: bool = True` to `SummarizationConfig`:

```python
class SummarizationConfig(BaseModel):
    enabled: bool = False
    max_token_context_length: int | None = None
    context_usage_ratio: float = 0.8
    preserve_last_messages: int = 2
    incremental: bool = True
```

3. When incremental summarization triggers:
    - The `ContextManagerTool` receives: (a) the existing `_running_summary` as a preamble, (b) only the NEW messages since the last summarization (not the summary message itself)
    - The summarization prompt becomes: "Here is the existing summary of the conversation so far:\n\n{running_summary}\n\nSummarize the following NEW messages, incorporating them into the existing summary. Preserve all key decisions, tool outputs, errors, current state, and next steps."
    - The result becomes the new `_running_summary`
    - `_compact_history()` replaces old messages with a single summary message + preserved tail

4. Add an incremental summarization prompt to `instructions.py`:

```python
HISTORY_SUMMARIZATION_PROMPT_INCREMENTAL = """You are updating a running summary of a conversation.

EXISTING SUMMARY:
{running_summary}

NEW MESSAGES TO INCORPORATE:
(See messages above)

Produce an updated summary that incorporates the new messages into the existing summary.
Focus on: key decisions, tool outputs, errors encountered, current state, and next steps.
The summary should enable continuing the task without access to the raw messages."""
```

5. In `_try_summarize_history`, when `incremental=True`:
    - If `_running_summary` exists, exclude the summary message from the messages sent to `ContextManagerTool`
    - Pass `_running_summary` as context to the tool
    - Store result back in `_running_summary`

**Why this matters**: This is the key to "infinite context." Each summarization pass only processes NEW messages while building on the accumulated summary. Information loss is bounded (one LLM interpretation layer) instead of compounding (summary-of-summary-of-summary).

### Change 4: Tool Output Compaction (Pre-Summarization)

**Files**:

- `dynamiq/nodes/agents/components/history_manager.py`
- `dynamiq/nodes/agents/agent.py`
- `dynamiq/nodes/agents/utils.py`

**Inspiration**: Manus AI's "Raw > Compaction > Summarization" hierarchy and OpenCode's pruning phase.

Before triggering the expensive LLM-based summarization, apply a cheap compaction pass that replaces stale tool outputs with short references:

1. Add config to `SummarizationConfig`:

```python
    compaction_enabled: bool = True
    compaction_token_threshold: int = 500
```

2. Add `_compact_tool_outputs()` to `HistoryManagerMixin`:

```python
def _compact_tool_outputs(self) -> bool:
    """Replace stale tool observation content with compact references.

    Returns True if any compaction was performed.
    """
    preserve_n = self.summarization_config.preserve_last_messages
    compactable = self._prompt.messages[self._history_offset:-preserve_n] if preserve_n > 0 \
                  else self._prompt.messages[self._history_offset:]

    compacted_any = False
    for msg in compactable:
        if (msg.role == MessageRole.USER
            and msg.content
            and msg.content.startswith("\nObservation:")
            and self._count_tokens_single(msg) > self.summarization_config.compaction_token_threshold):

            ref_id = generate_uuid()
            if self.file_store_backend:
                self.file_store_backend.store(f"compacted/{ref_id}.txt", msg.content.encode())

            msg.content = (
                f"\nObservation: [Output compacted - ref:{ref_id}. "
                f"Use file_read tool with path 'compacted/{ref_id}.txt' to retrieve full content if needed.]\n"
            )
            msg.static = True
            compacted_any = True

    return compacted_any
```

3. In the agent's ReAct loop (before each LLM call), the check sequence becomes:

```python
if self.summarization_config.enabled and self.is_token_limit_exceeded():
    if self.summarization_config.compaction_enabled:
        self._compact_tool_outputs()  # Step 1: cheap, reversible
    if self.is_token_limit_exceeded():
        self._try_summarize_history(...)  # Step 2: expensive, lossy (only if still over)
```

**Why this matters**: Tool outputs (web scrapes, file contents, command results) are often the largest messages. Compacting them is free (no LLM call) and reversible (agent can retrieve the full content). This can often avoid summarization entirely, preserving the full reasoning chain.

### Change 5: Structured Summarization Prompt

**Files**: `dynamiq/nodes/agents/prompts/react/instructions.py`

Replace the current generic summarization prompt with a structured checklist (inspired by Claude Code):

```python
HISTORY_SUMMARIZATION_PROMPT_REPLACE = """Produce a structured summary of the conversation above.
Your summary MUST include all of the following sections:

## User Intent
What the user asked for and any refinements to the original request.

## Progress
What has been accomplished so far, including successful tool calls and their key results.

## Key Decisions & Findings
Important technical decisions, discovered facts, and constraints.

## Errors & Resolutions
Any errors encountered and how they were resolved (or if they remain unresolved).

## Current State
The exact state of the work right now -- what files exist, what has been modified, etc.

## Next Steps
What remains to be done to complete the task.

Be concise but preserve all information needed to continue the task without access to the raw history."""
```

**Why this matters**: The current prompt ("Focus on key decisions, important information, and tool outputs") is too vague. A structured checklist ensures the LLM covers all critical areas, which is especially important for the ReAct intermediate steps (Thought/Action/Observation chains) that contain valuable reasoning context.

---

## 5. What NOT to Change

- **Memory system**: The existing `Memory` class is fine for its purpose (cross-session persistence of user/assistant pairs). It is NOT the right place for storing full agent trajectories. Changes 1-2 (message array input) give callers the ability to own their state externally.
- **ContextManagerTool architecture**: The recent refactor (`1df12b1a`) is solid. Changes 3-5 build on top of it rather than replacing it.
- **ReAct loop structure**: The Thought/Action/Observation cycle works well. We are improving how it gets *preserved* during summarization, not changing how it *executes*.
- **Existing `input: str` flow**: This must remain the default for backward compatibility. The `messages` field is purely additive.

---

## 6. Execution Priority

| Priority | Change                                     | Effort            | Impact                                                |
| -------- | ------------------------------------------ | ----------------- | ----------------------------------------------------- |
| **P0**   | Change 2: `messages` array input           | Small (1-2 days)  | Unblocks conversational agents                        |
| **P0**   | Change 1: Extend `Message` with tool_calls | Small (1 day)     | Required for Change 2 to be useful                    |
| **P1**   | Change 5: Structured summarization prompt  | Tiny (hours)      | Immediate quality improvement, no code changes needed |
| **P1**   | Change 3: Incremental summarization        | Medium (2-3 days) | Key to infinite context                               |
| **P2**   | Change 4: Tool output compaction           | Medium (2-3 days) | Cost reduction, preserves reasoning chains            |

Changes 1 + 2 should be done together as a single PR. Change 5 can be done independently as a quick win. Changes 3 + 4 can follow as a second PR.

---

## 7. Open Questions

1. **Message schema strictness**: Should we validate `tool_calls` structure (OpenAI format: `[{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]`) or keep it as `list[dict]` for flexibility across providers?
2. **Compaction storage**: Should compacted tool outputs use the agent's existing `file_store_backend`, or a dedicated compaction store? The file store approach is simpler but mixes compacted outputs with user files.
3. **Running summary visibility**: Should `_running_summary` be exposed in the agent's output (e.g., `execution_result["summary"]`) so callers can persist it and pass it back on subsequent calls?
4. **FUNCTION_CALLING mode interaction**: When the agent uses `InferenceMode.FUNCTION_CALLING`, tool calls come via the LLM's native function calling. Should these be stored as `tool_calls` in the `Message`, or continue to be represented as text (`"Function call: name(args)"`)?

---

## 8. Implementation Status

All 5 changes have been implemented and validated.

### Files Modified (10 total)

| File                                                 | Change                                                                                               |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `dynamiq/prompts/prompts.py`                         | Extended Message with tool_calls/tool_call_id/name, TOOL role, to_api_dict()                         |
| `dynamiq/nodes/agents/base.py`                       | Added messages array to AgentInputSchema, _build_history_from_messages(), incremental summary wiring |
| `dynamiq/nodes/agents/agent.py`                      | Two-phase compression in _try_summarize_history()                                                    |
| `dynamiq/nodes/agents/components/history_manager.py` | _running_summary, _compact_tool_outputs(), updated _compact_history()                                |
| `dynamiq/nodes/agents/utils.py`                      | SummarizationConfig: incremental, compaction_enabled, compaction_token_threshold                     |
| `dynamiq/nodes/tools/context_manager.py`             | _summarize_incremental(), running_summary input, budget-aware token accounting                       |
| `dynamiq/nodes/agents/prompts/react/instructions.py` | Structured summarization prompt, incremental prompt                                                  |
| `dynamiq/memory/memory.py`                           | None-safe content handling                                                                           |
| `dynamiq/memory/backends/in_memory.py`               | None-safe BM25 search                                                                               |
| `dynamiq/memory/backends/dynamo_db.py`               | None-safe content.lower() in search                                                                  |

### Test Results

- **561 unit tests passed**, 0 failures
- 10 edge case tests covering: mixed message types, tool-call-only messages, single message history, copy preservation, token counting, format_messages
- Backward compatibility verified: existing `input: str` flow unchanged

### What Was NOT Changed

- Streaming: uses `streaming_callback.accumulated_content` and `llm_result.output.get("content", "")` -- safe
- File support: already uses `(input_message.content or '')` pattern -- safe
- Sub-agents: `Message(**m)` construction works with new fields -- safe
- Orchestrators: no direct `Message.content` access -- safe
- Sandbox: no Message interaction -- safe
- ReAct loop: message construction unchanged; only summarization/compaction added as pre-step
