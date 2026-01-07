# RFC-001-09: UI & Chat Workflow Integration

**Status:** Final Draft v7.0  
**Created:** January 6, 2026  
**Part:** 9 of 9 (Additional)

---

## 1. Overview

This document addresses how checkpoint/resume integrates with UI-facing workflows, specifically:

- **Agentic chat workflows** - Single agent with streaming responses
- **Multi-turn conversations** - Threads with multiple runs
- **Real-time UI updates** - How checkpoints affect streaming events
- **Browser/device persistence** - Resuming after disconnect

---

## 2. Typical Chat Workflow Architecture

### 2.1 Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React/Vue)                    │
├─────────────────────────────────────────────────────────────────┤
│  • WebSocket/SSE connection to runtime                          │
│  • Renders streaming tokens as they arrive                      │
│  • Displays tool execution progress                             │
│  • Shows HITL approval dialogs                                  │
│  • Stores run_id and thread_id for resume                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ WebSocket / SSE / REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Runtime (FastAPI)                         │
├─────────────────────────────────────────────────────────────────┤
│  • POST /threads/{id}/runs → Create run                         │
│  • GET /runs/{id}/stream → SSE stream                          │
│  • WS /runs/{id}/stream/ws → WebSocket                         │
│  • POST /runs/{id}/input → HITL input                          │
│  • POST /runs/{id}/resume → Resume from checkpoint ← NEW       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Task Queue (PostgreSQL)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Worker (execute_run.py)                   │
├─────────────────────────────────────────────────────────────────┤
│  • Claims pending runs from queue                               │
│  • Executes workflow with streaming callbacks                   │
│  • Persists events to stream_chunks                            │
│  • Creates checkpoints after each node ← NEW                   │
│  • Polls run_input_events for HITL                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Workflow Structure for Chat

```python
# Typical agentic chat workflow

from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools import HumanFeedbackTool, PythonTool

agent = Agent(
    id="chat-agent",
    name="Assistant",
    llm=OpenAI(model="gpt-4o", streaming=True),  # Streaming enabled
    tools=[
        PythonTool(),
        HumanFeedbackTool(input_method="stream"),  # HITL via streaming
    ],
    max_loops=15,  # Multiple reasoning loops
)

flow = Flow(
    nodes=[agent],
    checkpoint_config=CheckpointConfig(
        enabled=True,
        backend=PostgresCheckpointBackend(...),
        checkpoint_mid_agent_loop=True,  # Checkpoint during long loops
    ),
)
```

---

## 3. Streaming Events with Checkpoints

### 3.1 Event Types

| Event | Purpose | Checkpoint Impact |
|-------|---------|-------------------|
| `streaming_chunk` | LLM token output | Not checkpointed (ephemeral) |
| `tool_call_started` | Tool execution start | Checkpoint captures loop state |
| `tool_call_completed` | Tool result | Checkpoint after tool completes |
| `approval_requested` | HITL prompt | Checkpoint with PENDING_INPUT |
| `approval_received` | User response | Clear pending, continue |
| `run_completed` | Final answer | Checkpoint status = COMPLETED |
| **`checkpoint_created`** | **NEW: Checkpoint saved** | For client tracking |

### 3.2 New Checkpoint Events for UI

```python
# New streaming events for checkpoint visibility

class CheckpointCreatedEvent(BaseStreamingEvent):
    """Emitted when checkpoint is created."""
    event: str = "checkpoint_created"
    data: CheckpointCreatedData

class CheckpointCreatedData(BaseModel):
    checkpoint_id: str
    status: str  # active, pending_input, completed
    node_id: str | None  # Node that triggered checkpoint
    loop_num: int | None  # For agents, which loop
    completed_nodes: int  # Count of completed nodes
    total_nodes: int  # Total nodes in flow

# Example event payload
{
    "event": "checkpoint_created",
    "data": {
        "checkpoint_id": "ckpt_abc123",
        "status": "active",
        "node_id": "chat-agent",
        "loop_num": 3,
        "completed_nodes": 0,
        "total_nodes": 1
    }
}
```

### 3.3 Event Sequence for Agent Chat

```
User sends: "Research and summarize AI trends"
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Loop 1: Planning                                          │
├──────────────────────────────────────────────────────────┤
│ → streaming_chunk: "Let me search for information..."     │
│ → tool_call_started: {tool: "TavilySearch"}              │
│ → tool_call_completed: {result: "Found 5 articles..."}   │
│ → checkpoint_created: {loop_num: 1, status: "active"}    │ ← Checkpoint
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Loop 2: Processing (HITL)                                 │
├──────────────────────────────────────────────────────────┤
│ → streaming_chunk: "I found several articles..."          │
│ → tool_call_started: {tool: "HumanFeedback"}             │
│ → approval_requested: "Should I include market data?"    │
│ → checkpoint_created: {status: "pending_input"}          │ ← Checkpoint
│                                                           │
│ [WORKFLOW PAUSES - Waiting for user]                     │
│                                                           │
│ User sends: "Yes, include market data"                   │
│ → approval_received                                       │
│ → checkpoint_created: {status: "active"}                 │ ← Checkpoint
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Loop 3-5: Research and synthesis                          │
├──────────────────────────────────────────────────────────┤
│ → [multiple tool calls and streaming chunks]             │
│ → checkpoint_created after each loop                     │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│ Final: Answer                                             │
├──────────────────────────────────────────────────────────┤
│ → streaming_chunk: "Based on my research..."             │
│ → streaming_chunk: [full response tokens]                │
│ → run_completed: {status: "success"}                     │
│ → checkpoint_created: {status: "completed"}              │ ← Final
└──────────────────────────────────────────────────────────┘
```

---

## 4. Frontend Integration

### 4.1 React Hook Example

```typescript
// hooks/useDynamiqChat.ts

import { useState, useCallback, useRef, useEffect } from 'react';

interface Checkpoint {
  id: string;
  status: 'active' | 'pending_input' | 'completed' | 'failed';
  loopNum?: number;
  nodeId?: string;
}

interface UseDynamiqChatReturn {
  messages: Message[];
  isLoading: boolean;
  isWaitingForInput: boolean;
  currentCheckpoint: Checkpoint | null;
  sendMessage: (content: string) => Promise<void>;
  sendInput: (content: string) => Promise<void>;
  resumeFromCheckpoint: (checkpointId?: string) => Promise<void>;
}

export function useDynamiqChat(threadId: string): UseDynamiqChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isWaitingForInput, setIsWaitingForInput] = useState(false);
  const [currentCheckpoint, setCurrentCheckpoint] = useState<Checkpoint | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const connectToStream = useCallback((newRunId: string) => {
    const ws = new WebSocket(`${WS_URL}/runs/${newRunId}/stream/ws`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.event) {
        case 'streaming_chunk':
          // Append tokens to current message
          setMessages(prev => appendChunk(prev, data.data));
          break;
          
        case 'tool_call_started':
          // Show tool execution indicator
          setMessages(prev => addToolStart(prev, data.data));
          break;
          
        case 'tool_call_completed':
          // Update tool result
          setMessages(prev => updateToolResult(prev, data.data));
          break;
          
        case 'approval_requested':
          // Show HITL dialog
          setIsWaitingForInput(true);
          setMessages(prev => addApprovalRequest(prev, data.data));
          break;
          
        case 'checkpoint_created':
          // Track checkpoint for potential resume
          setCurrentCheckpoint({
            id: data.data.checkpoint_id,
            status: data.data.status,
            loopNum: data.data.loop_num,
            nodeId: data.data.node_id,
          });
          
          // Store in localStorage for browser recovery
          localStorage.setItem(
            `checkpoint_${threadId}`, 
            JSON.stringify(data.data)
          );
          break;
          
        case 'run_completed':
        case 'run_failed':
          setIsLoading(false);
          setIsWaitingForInput(false);
          break;
      }
    };

    ws.onclose = () => {
      // Handle disconnection - checkpoint allows resume
      if (isLoading) {
        console.log('Disconnected during run, can resume from checkpoint');
      }
    };

    return ws;
  }, [threadId, isLoading]);

  const sendMessage = useCallback(async (content: string) => {
    setIsLoading(true);
    setMessages(prev => [...prev, { role: 'user', content }]);

    // Create new run
    const response = await fetch(`${API_URL}/threads/${threadId}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        workflow_uri: 'workflows/chat-agent.yaml',
        input: { input: content },
      }),
    });

    const run = await response.json();
    setRunId(run.id);
    
    // Connect to stream
    connectToStream(run.id);
  }, [threadId, connectToStream]);

  const sendInput = useCallback(async (content: string) => {
    if (!runId) return;
    
    // Send via WebSocket for lowest latency
    wsRef.current?.send(JSON.stringify({
      type: 'input',
      content,
    }));
    
    setIsWaitingForInput(false);
    setMessages(prev => addUserInput(prev, content));
  }, [runId]);

  const resumeFromCheckpoint = useCallback(async (checkpointId?: string) => {
    if (!runId) return;
    
    setIsLoading(true);
    
    // Resume from checkpoint
    const response = await fetch(`${API_URL}/runs/${runId}/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        checkpoint_id: checkpointId || currentCheckpoint?.id,
      }),
    });

    const resumedRun = await response.json();
    
    // Reconnect to stream
    connectToStream(resumedRun.id);
  }, [runId, currentCheckpoint, connectToStream]);

  // Check for recoverable checkpoint on mount
  useEffect(() => {
    const savedCheckpoint = localStorage.getItem(`checkpoint_${threadId}`);
    if (savedCheckpoint) {
      const checkpoint = JSON.parse(savedCheckpoint);
      if (checkpoint.status === 'pending_input') {
        setCurrentCheckpoint(checkpoint);
        setIsWaitingForInput(true);
        // Optionally auto-resume
        // resumeFromCheckpoint(checkpoint.id);
      }
    }
  }, [threadId]);

  return {
    messages,
    isLoading,
    isWaitingForInput,
    currentCheckpoint,
    sendMessage,
    sendInput,
    resumeFromCheckpoint,
  };
}
```

### 4.2 Chat Component Example

```tsx
// components/Chat.tsx

import { useDynamiqChat } from '../hooks/useDynamiqChat';

export function Chat({ threadId }: { threadId: string }) {
  const {
    messages,
    isLoading,
    isWaitingForInput,
    currentCheckpoint,
    sendMessage,
    sendInput,
    resumeFromCheckpoint,
  } = useDynamiqChat(threadId);

  return (
    <div className="chat-container">
      {/* Status Bar */}
      <div className="status-bar">
        {isLoading && (
          <span className="status-indicator">
            Processing... 
            {currentCheckpoint?.loopNum && (
              <span>(Loop {currentCheckpoint.loopNum})</span>
            )}
          </span>
        )}
        
        {currentCheckpoint?.status === 'pending_input' && (
          <span className="status-indicator warning">
            Waiting for your input
          </span>
        )}
      </div>

      {/* Messages */}
      <div className="messages">
        {messages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} />
        ))}
      </div>

      {/* HITL Dialog */}
      {isWaitingForInput && (
        <ApprovalDialog
          prompt={messages[messages.length - 1]?.content}
          onApprove={() => sendInput('APPROVE')}
          onReject={() => sendInput('REJECT')}
          onFeedback={(text) => sendInput(text)}
        />
      )}

      {/* Resume Banner (shown if disconnected with pending checkpoint) */}
      {!isLoading && currentCheckpoint?.status === 'pending_input' && (
        <div className="resume-banner">
          <p>Your previous session is waiting for input.</p>
          <button onClick={() => resumeFromCheckpoint()}>
            Resume Conversation
          </button>
        </div>
      )}

      {/* Input */}
      <ChatInput 
        onSend={sendMessage}
        disabled={isLoading}
      />
    </div>
  );
}
```

---

## 5. Checkpoint Visualization for UI

### 5.1 Progress Indicator

```tsx
// Visual representation of agent loops with checkpoints

function AgentProgress({ 
  currentLoop, 
  maxLoops, 
  checkpoints 
}: AgentProgressProps) {
  return (
    <div className="agent-progress">
      <div className="loop-bar">
        {Array.from({ length: maxLoops }, (_, i) => (
          <div 
            key={i}
            className={classNames('loop-segment', {
              'completed': i < currentLoop,
              'current': i === currentLoop,
              'has-checkpoint': checkpoints.some(c => c.loopNum === i),
            })}
          >
            {checkpoints.find(c => c.loopNum === i) && (
              <CheckpointMarker 
                checkpoint={checkpoints.find(c => c.loopNum === i)!}
              />
            )}
          </div>
        ))}
      </div>
      <span className="loop-label">
        Loop {currentLoop + 1} of {maxLoops}
      </span>
    </div>
  );
}
```

### 5.2 Checkpoint History (Debug/Admin View)

```tsx
// Admin view showing checkpoint history for debugging

function CheckpointHistory({ runId }: { runId: string }) {
  const { data: checkpoints } = useQuery({
    queryKey: ['checkpoints', runId],
    queryFn: () => fetch(`/runs/${runId}/checkpoints`).then(r => r.json()),
  });

  return (
    <div className="checkpoint-history">
      <h3>Checkpoint History</h3>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Status</th>
            <th>Node</th>
            <th>Loop</th>
            <th>Time</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {checkpoints?.map(cp => (
            <tr key={cp.id}>
              <td>{cp.id.slice(0, 8)}...</td>
              <td>
                <StatusBadge status={cp.status} />
              </td>
              <td>{cp.node_id || '-'}</td>
              <td>{cp.loop_num ?? '-'}</td>
              <td>{formatTime(cp.created_at)}</td>
              <td>
                <button onClick={() => resumeFrom(cp.id)}>
                  Resume
                </button>
                <button onClick={() => viewDetails(cp.id)}>
                  Details
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

---

## 6. Multi-Device Scenarios

### 6.1 Start on Desktop, Continue on Mobile

```
Desktop Browser                    Mobile App
      │                                 │
      │ 1. User starts chat             │
      ▼                                 │
┌─────────────┐                        │
│ Create Run  │                        │
│ Connect WS  │                        │
└─────────────┘                        │
      │                                 │
      │ 2. Agent requests approval      │
      ▼                                 │
┌─────────────┐                        │
│ HITL Dialog │                        │
│ shown       │                        │
└─────────────┘                        │
      │                                 │
      │ 3. User closes laptop           │
      │ (WebSocket disconnects)         │
      │                                 │
      │ Checkpoint saved:               │
      │ status=pending_input            │
      │                                 │
      │                                 │ 4. User opens mobile app
      │                                 ▼
      │                          ┌─────────────┐
      │                          │ App detects │
      │                          │ pending run │
      │                          └─────────────┘
      │                                 │
      │                                 │ 5. Resume from checkpoint
      │                                 ▼
      │                          ┌─────────────┐
      │                          │ Approval    │
      │                          │ re-shown    │
      │                          └─────────────┘
      │                                 │
      │                                 │ 6. User approves
      │                                 ▼
      │                          ┌─────────────┐
      │                          │ Run         │
      │                          │ completes   │
      │                          └─────────────┘
```

### 6.2 Implementation

```python
# Mobile app flow for detecting and resuming pending runs

async def check_pending_runs(user_id: str) -> list[PendingRun]:
    """Check for runs waiting for user input."""
    async with get_db_context() as session:
        # Find runs with pending_input checkpoints for this user
        result = await session.execute(
            select(Run, Checkpoint)
            .join(Checkpoint, Run.id == Checkpoint.run_id)
            .join(Thread, Run.thread_id == Thread.id)
            .where(
                Thread.user_id == user_id,
                Checkpoint.status == "pending_input",
            )
            .order_by(Checkpoint.created_at.desc())
        )
        
        return [
            PendingRun(
                run_id=row.Run.id,
                thread_id=row.Run.thread_id,
                checkpoint_id=row.Checkpoint.id,
                prompt=row.Checkpoint.pending_input_context.get("prompt"),
                created_at=row.Checkpoint.created_at,
            )
            for row in result
        ]
```

---

## 7. Agent Checkpoint State (Detailed)

### 7.1 What Gets Checkpointed Per Loop

```python
# During Agent._run_agent loop

class Agent(BaseAgent):
    def _run_agent(self, input_message, history_messages, config, **kwargs):
        # ... setup ...
        
        for loop_num in range(1, self.max_loops + 1):
            # STATE AT START OF LOOP:
            # - self._prompt.messages: Full conversation so far
            # - self._intermediate_steps: {loop_num: step_data}
            # - self._tool_cache: {(action, input): result}
            # - self._run_depends: Tracing dependencies
            
            # Execute LLM call
            llm_result = self._run_llm(...)
            
            # Parse and execute tool
            if action and self.tools:
                tool_result = self._run_tool(...)
                
                # Observation added to conversation
                self._prompt.messages.append(
                    Message(role=MessageRole.USER, content=f"Observation: {tool_result}")
                )
            
            # ============ CHECKPOINT POINT ============
            # After each loop, if checkpoint_mid_agent_loop is enabled:
            if should_checkpoint_mid_loop:
                self._emit_checkpoint_event(
                    loop_num=loop_num,
                    status=CheckpointStatus.ACTIVE,
                )
            # ==========================================
        
        return final_answer
```

### 7.2 Agent Checkpoint State Model

```python
# Extended NodeCheckpointState for Agent

class AgentCheckpointState(NodeCheckpointState):
    """Extended state for Agent nodes."""
    
    # Conversation state
    prompt_messages: list[dict]  # Full message history
    
    # Loop state
    current_loop: int
    total_loops: int
    
    # Reasoning trace
    intermediate_steps: dict[int, dict]  # {loop_num: step_data}
    
    # Efficiency (avoid re-executing tools)
    tool_cache: dict[str, Any]
    
    # Context
    call_context: dict | None  # user_id, session_id, metadata
    
    # Memory integration
    history_offset: int  # Where user messages start (after system)
    
    # Streaming state
    last_streamed_content: str | None  # For resume mid-token (optional)
```

### 7.3 Resume from Mid-Loop

```python
def restore_agent_from_checkpoint(agent: Agent, state: AgentCheckpointState):
    """Restore agent to exact loop state."""
    
    # Restore conversation
    agent._prompt.messages = [
        deserialize_message(m) for m in state.prompt_messages
    ]
    
    # Restore reasoning trace
    agent._intermediate_steps = state.intermediate_steps
    
    # Restore tool cache (skip re-executing successful tools)
    agent._tool_cache = {
        ToolCacheEntry(**k): v 
        for k, v in state.tool_cache.items()
    }
    
    # Restore tracing
    agent._run_depends = state.run_depends
    
    # Restore context
    agent._current_call_context = state.call_context
    agent._history_offset = state.history_offset
    
    # KEY: Set resume point
    agent._resume_from_loop = state.current_loop
    agent._is_resumed = True
```

---

## 8. Performance Considerations for UI

### 8.1 Checkpoint Frequency

| Setting | When to Use | UI Impact |
|---------|-------------|-----------|
| **After each node** | Short workflows (1-3 nodes) | Minimal, good for HITL |
| **After each agent loop** | Agent chats | Progress visibility |
| **On failure only** | High-throughput, non-HITL | No overhead |

### 8.2 Checkpoint Size Optimization

```python
# For chat workflows, conversation can grow large

class AgentCheckpointOptimizer:
    """Optimize agent checkpoint size for UI responsiveness."""
    
    @staticmethod
    def optimize_for_ui(state: AgentCheckpointState) -> AgentCheckpointState:
        """Reduce checkpoint size while preserving resume capability."""
        
        # Truncate very long tool outputs in cache
        optimized_cache = {}
        for key, value in state.tool_cache.items():
            if isinstance(value, str) and len(value) > 10000:
                # Keep first and last parts
                optimized_cache[key] = value[:5000] + "\n...[truncated]...\n" + value[-5000:]
            else:
                optimized_cache[key] = value
        
        state.tool_cache = optimized_cache
        
        # Don't checkpoint streaming content (ephemeral)
        state.last_streamed_content = None
        
        return state
```

### 8.3 Streaming Latency

```python
# Ensure checkpoints don't block streaming

async def save_checkpoint_async(checkpoint: FlowCheckpoint):
    """Non-blocking checkpoint save."""
    
    # Fire-and-forget save (log errors but don't block)
    asyncio.create_task(_save_checkpoint_background(checkpoint))

async def _save_checkpoint_background(checkpoint: FlowCheckpoint):
    try:
        await backend.asave(checkpoint)
        logger.debug(f"Checkpoint saved: {checkpoint.id}")
    except Exception as e:
        logger.warning(f"Checkpoint save failed (non-blocking): {e}")
```

---

## 9. Summary

**Key Takeaways for UI Integration:**

1. **Checkpoint events stream to UI** - Frontend knows checkpoint state in real-time
2. **HITL and checkpoints are synergistic** - Checkpoint captures "waiting for input" state
3. **Multi-device support works** - Resume from any device via checkpoint
4. **Agent loop state preserved** - Can resume from exact loop iteration
5. **Non-blocking saves** - Checkpoints don't add latency to streaming
6. **localStorage for browser recovery** - Persist checkpoint ID for tab/browser recovery

**UI Should:**
- Display checkpoint progress during agent loops
- Store current checkpoint ID for recovery
- Show "Resume" option when pending checkpoint exists
- Handle reconnection gracefully with checkpoint resume

---

## 10. Alternative: Continue via New Message (No New Endpoints)

### 10.1 The Cursor/ChatGPT Pattern

The user raises an important point: in chat contexts like Cursor or ChatGPT, you can often just **send a new message** and the conversation continues naturally. This is different from explicit checkpoint resume.

**Two Distinct Patterns:**

| Pattern | Use Case | Mechanism |
|---------|----------|-----------|
| **Continue via New Message** | Normal conversation flow | Memory + Thread context |
| **Explicit Checkpoint Resume** | Crash recovery, HITL pause | Checkpoint restore |

### 10.2 When "Continue via New Message" Works

```
Scenario: Agent completes task, user sends follow-up
────────────────────────────────────────────────────

Run 1: "Research AI trends"
  └── Agent executes, returns summary
  └── Run COMPLETED

Run 2: "Now summarize the top 3" (NEW MESSAGE)
  └── Agent uses Memory to recall Run 1's context
  └── Continues naturally
  └── No checkpoint needed!
```

**This already works in Dynamiq** because:
1. **Memory** persists conversation history per user/session
2. **Threads** group related runs
3. **Agent** can access previous conversation via `memory.get_agent_conversation()`

### 10.3 When Checkpoint Resume is NEEDED

```
Scenario: Browser crash during HITL wait
────────────────────────────────────────

Run 1: "Review and approve this contract"
  └── Agent analyzes document
  └── Agent requests approval: "Should I proceed?"
  └── HITL event emitted
  └── [BROWSER CRASHES]
  └── WebSocket disconnects
  └── Run times out...

WITHOUT Checkpoint:
  └── Run marked FAILED
  └── User must restart from beginning
  └── All LLM calls re-executed ($$)

WITH Checkpoint:
  └── Checkpoint saved with PENDING_INPUT
  └── User returns, sees pending run
  └── Resume re-emits approval prompt
  └── User approves, workflow completes
```

### 10.4 Avoiding New Endpoints: Implicit Resume

**Option A: Extend Existing Endpoint**

Instead of `POST /runs/{id}/resume`, extend `POST /threads/{id}/runs`:

```python
# In app/api/v2/runs.py

@router.post("/threads/{thread_id}/runs")
async def create_run(
    thread_id: UUID,
    request: CreateRunRequest,
    # NEW: Optional auto-resume behavior
    auto_resume_pending: bool = Query(
        default=True,
        description="If thread has a PENDING_INPUT run, resume it instead of creating new"
    ),
) -> RunResponse:
    """
    Create a new run, or resume pending run if auto_resume_pending=True.
    
    Behavior:
    - If thread has a run with status=PENDING_INPUT and auto_resume_pending=True:
      - Resume that run instead of creating new
      - Use the new message as HITL input
    - Otherwise: Create new run normally
    """
    
    # Check for pending run
    if auto_resume_pending:
        pending_run = await get_pending_run(db, thread_id)
        if pending_run:
            # Treat the new input as HITL response
            await save_hitl_input(
                pending_run.id, 
                event_type="continuation",
                data={"content": request.input.get("input", "")}
            )
            # Resume the pending run
            return await resume_run_internal(pending_run)
    
    # Normal: create new run
    return await create_new_run(...)
```

**Option B: Smart Thread Behavior**

```python
# The runtime auto-detects context and does the right thing

async def create_run_smart(thread_id: UUID, input_data: dict):
    """
    Smart run creation that handles continuations automatically.
    
    1. If thread has PENDING_INPUT run → Resume with input as HITL response
    2. If thread has FAILED run with checkpoint → Offer to resume
    3. Otherwise → Create new run with memory context
    """
    thread = await get_thread(thread_id)
    
    # Case 1: Pending HITL
    if thread.active_run_id:
        active_run = await get_run(thread.active_run_id)
        if active_run.status == RunStatus.PENDING_INPUT:
            # User's new message is the HITL response
            return await handle_as_hitl_input(active_run, input_data)
    
    # Case 2: Recent failed run with checkpoint
    recent_failed = await get_recent_failed_run(thread_id, within_minutes=30)
    if recent_failed and recent_failed.checkpoint_id:
        # Include checkpoint context in new run
        input_data["_resume_context"] = {
            "checkpoint_id": recent_failed.checkpoint_id,
            "failed_at": recent_failed.failed_at,
        }
    
    # Case 3: Normal - create with memory context
    return await create_new_run(thread_id, input_data)
```

### 10.5 Memory-Based Continuation (Already Works!)

For **normal chat continuations**, no checkpoint is needed:

```python
# Agent already does this via Memory

class Agent:
    def execute(self, input_data, ...):
        # Memory retrieves conversation history
        user_id = input_data.get("user_id")
        session_id = input_data.get("session_id")
        
        if self.memory and (user_id or session_id):
            # Get previous conversation
            history_messages = self._retrieve_memory(input_data)
            # Agent sees full context from previous runs
            ...
```

**This means:** For simple "continue the conversation" cases, **you don't need checkpoints at all** - Memory handles it.

### 10.6 When to Use What

| Scenario | Solution | New Endpoint? |
|----------|----------|---------------|
| Normal follow-up message | Memory (existing) | ❌ No |
| HITL input (workflow running) | `run_input_events` (existing) | ❌ No |
| HITL resume (workflow paused) | Auto-resume in existing endpoint | ❌ No (Option A) |
| Crash recovery | Explicit checkpoint resume | ✅ Yes (or Option B) |
| Time-travel to past state | Explicit checkpoint | ✅ Yes |

### 10.7 Recommended Approach

**For Chat Workflows (Majority Case):**
1. Use Memory for conversation continuity (existing)
2. Extend `POST /threads/{id}/runs` with `auto_resume_pending=True` (Option A)
3. No new endpoints for basic use cases

**For Advanced Use Cases:**
1. Add `/runs/{id}/checkpoints` for debugging/time-travel (optional)
2. Add `/runs/{id}/resume` for explicit control (optional)

### 10.8 Implementation Priority

| Feature | Priority | Endpoint Impact |
|---------|----------|-----------------|
| Memory-based continuation | ✅ Already exists | None |
| HITL input during run | ✅ Already exists | None |
| Auto-resume pending (Option A) | High | Extend existing |
| Checkpoint visibility | Medium | New GET endpoint |
| Explicit resume | Low | New POST endpoint |

**Bottom Line:** For 90% of chat use cases, **no new endpoints are needed**. Checkpoints provide the safety net for the 10% of edge cases (crashes, long pauses).

---

**Previous:** [08-TESTING-MIGRATION.md](./08-TESTING-MIGRATION.md)  
**Index:** [README.md](./README.md)
