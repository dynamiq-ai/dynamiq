# RFC-001-04: Node Analysis

**Status:** Final Draft v7.0  
**Created:** January 6, 2026  
**Part:** 4 of 9

---

## 1. Overview

This document provides a systematic analysis of every node type in Dynamiq, categorizing them by checkpoint requirements and providing implementation guidance.

**Principle:** Only checkpoint what's needed for resume. Node definitions (tools, LLM config) are already in the workflow - we only save runtime state.

---

## 2. Node Categories

| Category | Description | Examples | State Complexity |
|----------|-------------|----------|------------------|
| **A: Stateless** | No internal runtime state | Converters, Validators | None |
| **B: Simple** | 1-2 fields to save | BaseLLM, Rankers | Low |
| **C: Complex** | Multiple fields, custom serialization | Agent, Orchestrators, Map | High |
| **D: External** | External resource connections | E2B, HumanFeedback, MCP | Medium-High |

---

## 3. Category A: Stateless Nodes

These nodes have no internal runtime state that needs checkpointing. The Flow captures their input/output.

### 3.1 List of Stateless Nodes

| Node Type | Reason |
|-----------|--------|
| **Converters** (PyPDF, Unstructured, LLMTextExtractor) | Conversion is atomic |
| **Embedders** (OpenAI, Mistral, etc.) | Embedding calls are atomic |
| **Retrievers** (Pinecone, Qdrant, Chroma, etc.) | Query is atomic, vector store is external |
| **Writers** (Pinecone, Qdrant, Chroma, etc.) | Write is atomic (success/fail) |
| **Validators** (ValidJSON, ValidPython, etc.) | Pure validation, no state |
| **Splitters** (DocumentSplitter) | Splitting is atomic |
| **Audio** (WhisperSTT, ElevenLabs) | Processing is atomic |
| **Image** (ImageGeneration, ImageEdit) | Generation is atomic |
| **Operators** (Choice, Pass) | Pure evaluation |
| **Simple Tools** (Tavily, Firecrawl, HTTP, SQL) | API calls are atomic |
| **Python** (Code Executor) | Execution is atomic |

### 3.2 Default Implementation

All stateless nodes inherit the default behavior:

```python
# From CheckpointMixin
def get_checkpoint_state(self) -> dict:
    return {}  # No internal state

def restore_from_checkpoint(self, state: dict) -> None:
    pass  # Nothing to restore
```

---

## 4. Category B: Simple State Nodes

### 4.1 BaseLLM / OpenAI / Anthropic / etc.

| Field | Save? | Reason |
|-------|-------|--------|
| `_is_fallback_run` | ✅ | Track if fallback was triggered |
| `client` | ❌ | Reconstructed from connection |
| `connection` | ❌ | Configuration |

```python
class BaseLLM(ConnectionNode):
    """LLM with simple checkpoint support."""
    
    def get_checkpoint_state(self) -> dict:
        return {"is_fallback_run": self._is_fallback_run}
    
    def restore_from_checkpoint(self, state: dict) -> None:
        self._is_fallback_run = state.get("is_fallback_run", False)
        self._is_resumed = True
```

### 4.2 LLMDocumentRanker

| Field | Save? | Reason |
|-------|-------|--------|
| `_run_depends` | ✅ | Tracing dependencies |

```python
class LLMDocumentRanker(Node):
    def get_checkpoint_state(self) -> dict:
        return {"run_depends": self._run_depends.copy()}
    
    def restore_from_checkpoint(self, state: dict) -> None:
        self._run_depends = state.get("run_depends", [])
        self._is_resumed = True
```

---

## 5. Category C: Complex State Nodes

### 5.1 Agent (ReAct Agent)

**The most complex node.** Agent maintains conversation history, intermediate reasoning steps, tool cache, and loop state. This is critical for agentic chat workflows where an agent may go through multiple reasoning loops before providing an answer.

#### State Analysis (Comprehensive)

Based on analysis of `dynamiq/nodes/agents/agent.py` and `dynamiq/nodes/agents/base.py`:

| Field | Save? | Type | Reason |
|-------|-------|------|--------|
| **Conversation State** | | | |
| `_prompt.messages` | ✅ | List[Message] | Full conversation with LLM (system, user, assistant, observations) |
| `_history_offset` | ✅ | Int | Index where user messages start (after system prompt) |
| **Loop State** | | | |
| `_current_loop` | ✅ | Int | Current iteration for resume |
| `max_loops` | ❌ | Int | Configuration (already in workflow def) |
| **Reasoning Trace** | | | |
| `_intermediate_steps` | ✅ | Dict[int, dict] | `{loop_num: {input_data, model_observation, final_answer}}` |
| **Tool Execution Cache** | | | |
| `_tool_cache` | ✅ | Dict[ToolCacheEntry, Any] | `{(action, action_input): result}` - Skip re-execution |
| **Tracing** | | | |
| `_run_depends` | ✅ | List[dict] | Dependencies for tracing continuity |
| **Prompt State** | | | |
| `system_prompt_manager._prompt_variables` | ✅ | Dict | Runtime variables merged into prompt |
| `system_prompt_manager._prompt_blocks` | ⚠️ | Dict | Only if modified at runtime |
| **Context** | | | |
| `_current_call_context` | ✅ | Dict | `{user_id, session_id, metadata}` for memory isolation |
| **NOT Saved (Configuration)** | | | |
| `llm` | ❌ | BaseLLM | Configuration - reconstructed from workflow |
| `tools` | ❌ | List[Node] | Configuration - reconstructed from workflow |
| `memory` | ❌ | Memory | External backend - accessed via context |
| `file_store` | ❌ | FileStore | External backend |
| `streaming` | ❌ | StreamingConfig | Configuration |
| `inference_mode` | ❌ | InferenceMode | Configuration |

#### Implementation

```python
from dynamiq.nodes.agents.base import Agent as BaseAgent
from dynamiq.nodes.agents.utils import ToolCacheEntry
from dynamiq.prompts import Message, MessageRole, VisionMessage, VisionMessageTextContent

class Agent(BaseAgent):
    """Agent with comprehensive checkpoint support."""
    
    def get_checkpoint_state(self) -> dict:
        """
        Extract all state needed for resume.
        
        Critical for agentic chat workflows where an agent may execute
        15+ loops with tool calls before providing a final answer.
        
        Returns:
            Dictionary with serializable state. All nested objects
            are converted to primitive types for JSON serialization.
        """
        return {
            # === Conversation History ===
            # Full message sequence: system → user → assistant → observation → ...
            "prompt_messages": [
                self._serialize_message(msg) 
                for msg in self._prompt.messages
            ],
            
            # Where user messages start (skip system prompt on resume)
            "history_offset": self._history_offset,
            
            # === Reasoning Trace ===
            # {loop_num: {input_data, model_observation: {initial, tool_using, tool_input, tool_output}, final_answer}}
            "intermediate_steps": self._intermediate_steps.copy(),
            
            # === Tool Execution Cache ===
            # Avoid re-executing identical tool calls on resume
            "tool_cache": self._serialize_tool_cache(),
            
            # === Tracing Dependencies ===
            "run_depends": self._run_depends.copy(),
            
            # === Loop State (Critical for Chat Workflows) ===
            "current_loop": getattr(self, "_current_loop", 0),
            
            # === Prompt Manager State ===
            "prompt_variables": (
                self.system_prompt_manager._prompt_variables.copy()
                if self.system_prompt_manager else {}
            ),
            
            # === User/Session Context (Memory Isolation) ===
            # {user_id, session_id, metadata} - enables memory per-user
            "call_context": (
                self._current_call_context.copy() 
                if self._current_call_context else None
            ),
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        """
        Restore agent state from checkpoint.
        
        After restore, agent will continue from the exact loop iteration
        where it was checkpointed, with full conversation history intact.
        This is critical for HITL workflows where user may close browser
        mid-conversation.
        
        Args:
            state: Dictionary from get_checkpoint_state()
        """
        # Restore conversation history
        self._prompt.messages = [
            self._deserialize_message(m) 
            for m in state.get("prompt_messages", [])
        ]
        
        # Restore history offset for memory/summarization
        self._history_offset = state.get("history_offset", 2)
        
        # Restore reasoning trace (for tracing continuity)
        self._intermediate_steps = state.get("intermediate_steps", {})
        
        # Restore tool cache (skip re-executing successful calls)
        self._tool_cache = self._deserialize_tool_cache(
            state.get("tool_cache", {})
        )
        
        # Restore tracing dependencies
        self._run_depends = state.get("run_depends", [])
        
        # === CRITICAL: Set resume loop ===
        # Agent._run_agent will start from this loop instead of 1
        self._resume_from_loop = state.get("current_loop", 0)
        
        # Restore prompt manager variables
        if self.system_prompt_manager:
            self.system_prompt_manager._prompt_variables = state.get(
                "prompt_variables", {}
            )
        
        # Restore user/session context (for memory isolation)
        self._current_call_context = state.get("call_context")
        
        # Mark as resumed (tools may behave differently)
        self._is_resumed = True
    
    def _serialize_message(self, msg) -> dict:
        """
        Convert Message to serializable dict.
        
        Handles both regular Messages and VisionMessages (with image content).
        """
        if hasattr(msg, 'model_dump'):
            data = msg.model_dump()
            # Ensure role is string, not enum
            if 'role' in data and hasattr(data['role'], 'value'):
                data['role'] = data['role'].value
            return data
        return {"role": str(msg.role), "content": str(msg.content)}
    
    def _deserialize_message(self, data: dict):
        """
        Reconstruct Message from dict.
        
        Detects VisionMessage vs regular Message by content type.
        """
        # Handle role enum
        if 'role' in data and isinstance(data['role'], str):
            data['role'] = MessageRole(data['role'])
        
        # VisionMessage has list content
        if "content" in data and isinstance(data["content"], list):
            return VisionMessage(**data)
        
        return Message(**data)
    
    def _serialize_tool_cache(self) -> dict:
        """
        Serialize tool cache for checkpointing.
        
        The cache key is (action_name, action_input_json).
        Values are tool results (typically strings or dicts).
        """
        serialized = {}
        cache = getattr(self, "_tool_cache", {})
        
        for key, value in cache.items():
            # ToolCacheEntry is a namedtuple-like, convert to string key
            if isinstance(key, ToolCacheEntry):
                cache_key = f"{key.action}::{json.dumps(key.action_input, sort_keys=True)}"
            else:
                cache_key = str(key)
            
            # Ensure value is serializable
            try:
                json.dumps(value)
                serialized[cache_key] = value
            except (TypeError, ValueError):
                # Large or non-serializable results: store truncated
                if isinstance(value, str) and len(value) > 10000:
                    serialized[cache_key] = value[:10000] + "...[truncated]"
        
        return serialized
    
    def _deserialize_tool_cache(self, data: dict) -> dict:
        """
        Restore tool cache from checkpoint.
        
        Reconstructs ToolCacheEntry keys.
        """
        cache = {}
        for key_str, value in data.items():
            if "::" in key_str:
                action, input_json = key_str.split("::", 1)
                try:
                    action_input = json.loads(input_json)
                    cache[ToolCacheEntry(action=action, action_input=action_input)] = value
                except json.JSONDecodeError:
                    pass
        return cache
```

#### Agent Loop Resume Logic

The key modification is in `Agent._run_agent()` to support resuming from a specific loop:

```python
# In Agent._run_agent() - modifications for checkpoint resume

def _run_agent(
    self,
    input_message: Message | VisionMessage,
    history_messages: list[Message] | None = None,
    config: RunnableConfig | None = None,
    **kwargs,
) -> str:
    """Modified _run_agent with checkpoint resume support."""
    
    # === CHECKPOINT RESUME: Determine start loop ===
    start_loop = 1
    if getattr(self, "_is_resumed", False):
        start_loop = getattr(self, "_resume_from_loop", 1)
        
        # On resume, conversation history is already restored
        # Skip the initial message setup
        if not self._prompt.messages:
            # Only set up messages if not restored from checkpoint
            system_message = Message(
                role=MessageRole.SYSTEM,
                content=self.generate_prompt(...),
            )
            if history_messages:
                self._prompt.messages = [system_message, *history_messages, input_message]
            else:
                self._prompt.messages = [system_message, input_message]
    else:
        # Fresh run - set up messages normally
        system_message = Message(...)
        self._prompt.messages = [system_message, input_message]
    
    # === MAIN LOOP ===
    for loop_num in range(start_loop, self.max_loops + 1):
        # Track current loop for checkpointing
        self._current_loop = loop_num
        
        # ... existing LLM call and tool execution logic ...
        
        # Execute tool if action parsed
        if action and self.tools:
            # Check tool cache first (restored from checkpoint)
            tool_cache_key = ToolCacheEntry(action=action, action_input=action_input)
            cached_result = self._tool_cache.get(tool_cache_key)
            
            if cached_result and self._is_resumed:
                # Use cached result, skip tool execution
                tool_result = cached_result
                logger.info(f"Agent {self.name}: Using cached tool result for {action}")
            else:
                # Execute tool
                tool_result, tool_files = self._run_tool(...)
            
            # Add observation to conversation
            observation = f"\nObservation: {tool_result}\n"
            self._prompt.messages.append(
                Message(role=MessageRole.USER, content=observation, static=True)
            )
        
        # === CHECKPOINT AFTER LOOP ===
        # Emit checkpoint event for UI tracking and persistence
        if self._should_emit_loop_checkpoint(config):
            self._emit_checkpoint_event(
                loop_num=loop_num,
                config=config,
                **kwargs,
            )
    
    # Max loops handling...
    return final_answer

def _should_emit_loop_checkpoint(self, config: RunnableConfig) -> bool:
    """Check if should emit checkpoint event after this loop."""
    # Check if streaming is enabled (for UI visibility)
    if not self.streaming.enabled:
        return False
    
    # Check if checkpoint_mid_agent_loop is enabled in flow config
    flow_config = getattr(config, 'checkpoint_config', None)
    if flow_config and getattr(flow_config, 'checkpoint_mid_agent_loop', False):
        return True
    
    return False

def _emit_checkpoint_event(self, loop_num: int, config: RunnableConfig, **kwargs):
    """Emit checkpoint event for UI tracking."""
    self.stream_content(
        content={
            "type": "checkpoint_requested",
            "loop_num": loop_num,
            "current_loop": self._current_loop,
            "max_loops": self.max_loops,
        },
        source=self.name,
        step="checkpoint",
        config=config,
        **kwargs,
    )
```

**Key Points:**
1. `_resume_from_loop` determines where to start the loop
2. `_tool_cache` avoids re-executing successful tool calls
3. Conversation history (`_prompt.messages`) is restored, so no re-prompting
4. `_emit_checkpoint_event` allows UI to track progress

---

### 5.2 GraphOrchestrator

Orchestrates multiple states with agents, maintaining conversation and context across states.

#### State Analysis

| Field | Save? | Reason |
|-------|-------|--------|
| `_chat_history` | ✅ | Conversation across states |
| `context` | ✅ | Shared context dict |
| `_run_depends` | ✅ | Tracing |
| `_current_state_id` | ✅ | Resume state |
| `_completed_states` | ✅ | Progress tracking |
| `states` | ❌ | Configuration |
| `manager` | ❌ | Configuration |

#### Implementation

```python
class GraphOrchestrator(Orchestrator):
    """Graph orchestrator with checkpoint support."""
    
    # Track state for checkpointing
    _current_state_id: str | None = None
    _completed_states: list[str] = []
    
    def get_checkpoint_state(self) -> dict:
        return {
            "chat_history": self._chat_history.copy(),
            "context": self.context.copy(),
            "run_depends": self._run_depends.copy(),
            "current_state_id": self._current_state_id,
            "completed_states": self._completed_states.copy(),
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        self._chat_history = state.get("chat_history", [])
        self.context = state.get("context", {})
        self._run_depends = state.get("run_depends", [])
        self._current_state_id = state.get("current_state_id")
        self._completed_states = state.get("completed_states", [])
        self._is_resumed = True
```

---

### 5.3 LinearOrchestrator

Executes tasks sequentially, tracking results and progress.

#### State Analysis

| Field | Save? | Reason |
|-------|-------|--------|
| `_results` | ✅ | Task results by task_id |
| `_chat_history` | ✅ | Conversation history |
| `_run_depends` | ✅ | Tracing |
| `_current_task_index` | ✅ | Resume task |
| `_tasks` | ✅ | Parsed task list |
| `agents` | ❌ | Configuration |

#### Implementation

```python
class LinearOrchestrator(Orchestrator):
    """Linear orchestrator with checkpoint support."""
    
    _current_task_index: int = 0
    _tasks: list = []
    
    def get_checkpoint_state(self) -> dict:
        return {
            "results": self._results.copy(),
            "chat_history": self._chat_history.copy(),
            "run_depends": self._run_depends.copy(),
            "current_task_index": self._current_task_index,
            "tasks": [
                task.model_dump() for task in self._tasks
            ] if self._tasks else [],
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        from dynamiq.nodes.orchestrators.types import Task
        
        self._results = state.get("results", {})
        self._chat_history = state.get("chat_history", [])
        self._run_depends = state.get("run_depends", [])
        self._current_task_index = state.get("current_task_index", 0)
        
        task_data = state.get("tasks", [])
        self._tasks = [Task(**t) for t in task_data] if task_data else []
        self._is_resumed = True
```

---

### 5.4 Map Node

Executes a workflow in parallel across a list of inputs. Critical for efficient checkpointing.

#### State Analysis

| Field | Save? | Reason |
|-------|-------|--------|
| `_completed_iterations` | ✅ | Dict: index → result |
| `_total_iterations` | ✅ | Total count |

#### Implementation

```python
class Map(Node):
    """Map node with iteration-level checkpoint support."""
    
    def get_checkpoint_state(self) -> dict:
        return {
            "completed_iterations": getattr(self, "_completed_iterations", {}),
            "total_iterations": getattr(self, "_total_iterations", 0),
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        self._completed_iterations = state.get("completed_iterations", {})
        self._total_iterations = state.get("total_iterations", 0)
        self._is_resumed = True
    
    def execute(
        self, 
        input_data: MapInputSchema, 
        config: RunnableConfig = None, 
        **kwargs
    ):
        """
        Execute map with resume support.
        
        On resume, only executes iterations not in _completed_iterations.
        """
        input_list = input_data.input
        self._total_iterations = len(input_list)
        
        # Determine which iterations to execute
        if getattr(self, "_is_resumed", False) and hasattr(self, "_completed_iterations"):
            # Only execute pending iterations
            pending = [
                (i, data) for i, data in enumerate(input_list)
                if i not in self._completed_iterations
            ]
        else:
            # Fresh execution
            pending = list(enumerate(input_list))
            self._completed_iterations = {}
        
        # Execute pending iterations in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._execute_single, 
                    index, data, config, kwargs
                ): index
                for index, data in pending
            }
            
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                    self._completed_iterations[index] = result
                except Exception as e:
                    self._completed_iterations[index] = {"error": str(e)}
        
        # Reconstruct ordered results
        results = [
            self._completed_iterations[i] 
            for i in range(self._total_iterations)
        ]
        return {"output": results}
    
    def _execute_single(self, index, data, config, kwargs):
        """Execute single iteration."""
        return self.execute_workflow(index, data, config, kwargs)
```

---

## 6. Category D: External Resource Nodes

### 6.1 E2BInterpreterTool

Connects to E2B sandbox for code execution. Sandbox may expire between checkpoint and resume.

#### State Analysis

| Field | Save? | Reason |
|-------|-------|--------|
| `sandbox_id` | ✅ | For reconnection attempt |
| `installed_packages` | ✅ | Reinstall if sandbox expired |
| `uploaded_file_names` | ✅ | Reference only |
| `_sandbox` | ❌ | External resource |
| `files` | ❌ | Can re-upload |

#### Implementation

```python
class E2BInterpreterTool(ConnectionNode):
    """E2B tool with reconnection support."""
    
    def get_checkpoint_state(self) -> dict:
        return {
            "sandbox_id": (
                self._sandbox.sandbox_id if self._sandbox else None
            ),
            "installed_packages": self.installed_packages.copy(),
            "uploaded_file_names": [
                getattr(f, 'name', '') for f in (self.files or [])
            ],
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        """
        Restore E2B state.
        
        Attempts to reconnect to existing sandbox. If sandbox expired,
        a new one will be created on next execute() with packages reinstalled.
        """
        from e2b_code_interpreter import Sandbox
        
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            try:
                # Attempt reconnection
                self._sandbox = Sandbox.connect(
                    sandbox_id, 
                    api_key=self.connection.api_key
                )
                logger.info(f"Reconnected to E2B sandbox: {sandbox_id}")
            except Exception as e:
                # Sandbox expired or unavailable
                logger.warning(
                    f"Could not reconnect to sandbox {sandbox_id}: {e}. "
                    "Will create new sandbox on next execute."
                )
                self._sandbox = None
        
        # Restore package list for reinstallation
        self.installed_packages = state.get("installed_packages", [])
        self._is_resumed = True
```

---

### 6.2 HumanFeedbackTool

Requests human input via streaming. Critical for HITL workflows.

#### State Analysis

| Field | Save? | Reason |
|-------|-------|--------|
| `_pending_prompt` | ✅ | Question asked |
| `_request_timestamp` | ✅ | When asked |
| `input_queue` | ❌ | Runtime resource |

#### Implementation

```python
class HumanFeedbackTool(Node):
    """Human feedback tool with checkpoint support."""
    
    # Internal state for checkpointing
    _pending_prompt: str | None = None
    _request_timestamp: datetime | None = None
    
    def get_checkpoint_state(self) -> dict:
        return {
            "pending_prompt": self._pending_prompt,
            "request_timestamp": (
                self._request_timestamp.isoformat() 
                if self._request_timestamp else None
            ),
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        """
        Restore human feedback state.
        
        On resume, the tool will re-emit the pending prompt via streaming
        so the user sees the approval request again.
        """
        self._pending_prompt = state.get("pending_prompt")
        ts = state.get("request_timestamp")
        self._request_timestamp = (
            datetime.fromisoformat(ts) if ts else None
        )
        self._is_resumed = True
    
    def execute(self, input_data, config, **kwargs):
        """Execute with resume support."""
        if self._is_resumed and self._pending_prompt:
            # Re-emit the approval event
            self._emit_approval_event(
                prompt=self._pending_prompt,
                is_resumed=True,  # Flag for client
            )
        else:
            # Normal execution
            self._pending_prompt = input_data.get("prompt", "Approve?")
            self._request_timestamp = datetime.utcnow()
            self._emit_approval_event(prompt=self._pending_prompt)
        
        # Wait for input from queue
        response = self._wait_for_input()
        
        # Clear pending state
        self._pending_prompt = None
        self._request_timestamp = None
        
        return response
```

---

### 6.3 MCPServer / MCPTool

Model Context Protocol server connection. Needs tool rediscovery on resume.

#### State Analysis

| Field | Save? | Reason |
|-------|-------|--------|
| `server_name` | ✅ | Identification |
| `discovered_tool_names` | ✅ | Validation |
| `session` | ❌ | External connection |
| `client` | ❌ | External connection |

#### Implementation

```python
class MCPServer:
    """MCP server with reconnection support."""
    
    def get_checkpoint_state(self) -> dict:
        return {
            "server_name": self.name,
            "discovered_tool_names": [
                t.name for t in self._mcp_tools
            ],
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        """
        Restore MCP server state.
        
        Tools will be rediscovered on next init_components call.
        We save discovered names for validation that server has same tools.
        """
        self._discovered_tool_names = state.get("discovered_tool_names", [])
        self._is_resumed = True
```

---

### 6.4 Browser Tools (Stagehand)

Browser automation tools maintain session state.

#### State Analysis

| Field | Save? | Reason |
|-------|-------|--------|
| `session_id` | ✅ | For session restoration |
| `current_url` | ✅ | Navigation state |
| `page_state` | ⚠️ | Complex, may not be restorable |
| `browser` | ❌ | External resource |

#### Implementation

```python
class StagehandTool(ConnectionNode):
    """Browser tool with limited checkpoint support."""
    
    def get_checkpoint_state(self) -> dict:
        """
        Note: Full browser state cannot be restored.
        We save what we can for logging/debugging.
        """
        return {
            "session_id": getattr(self, "_session_id", None),
            "current_url": getattr(self, "_current_url", None),
            "navigation_history": getattr(self, "_navigation_history", []),
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        """
        Restore what we can. Browser session likely expired.
        
        On next execute, will create new session and potentially
        navigate to last known URL.
        """
        self._session_id = state.get("session_id")
        self._current_url = state.get("current_url")
        self._navigation_history = state.get("navigation_history", [])
        self._is_resumed = True
        self._needs_session_recreation = True
```

---

## 7. Complex Type Serialization (Critical)

### 7.1 Message Types

The Agent's `_prompt.messages` contains `Message` and `VisionMessage` objects that need special serialization:

```python
# dynamiq/prompts/prompts.py

class Message(BaseModel):
    """Regular text message."""
    content: str
    role: MessageRole  # Enum: USER, SYSTEM, ASSISTANT
    metadata: dict | None = None
    static: bool = False  # Exclude from serialization

class VisionMessage(BaseModel):
    """Message with images."""
    content: list[VisionMessageTextContent | VisionMessageImageContent]
    role: MessageRole
    static: bool = False

class VisionMessageTextContent(BaseModel):
    type: VisionMessageType = VisionMessageType.TEXT
    text: str

class VisionMessageImageContent(BaseModel):
    type: VisionMessageType = VisionMessageType.IMAGE_URL
    image_url: VisionMessageImageURL  # Contains url and detail level
```

**Serialization Strategy:**

```python
def _serialize_message(msg: Message | VisionMessage) -> dict:
    """Serialize message for checkpoint."""
    if isinstance(msg, VisionMessage):
        return {
            "type": "vision",
            "role": msg.role.value,  # Convert enum to string
            "content": [
                c.model_dump() for c in msg.content
            ],
        }
    else:
        return {
            "type": "text",
            "role": msg.role.value,
            "content": msg.content,
            "metadata": msg.metadata,
        }

def _deserialize_message(data: dict) -> Message | VisionMessage:
    """Deserialize message from checkpoint."""
    role = MessageRole(data["role"])  # Convert string to enum
    
    if data["type"] == "vision":
        content = []
        for c in data["content"]:
            if c["type"] == "text":
                content.append(VisionMessageTextContent(**c))
            else:
                content.append(VisionMessageImageContent(**c))
        return VisionMessage(role=role, content=content)
    else:
        return Message(
            role=role,
            content=data["content"],
            metadata=data.get("metadata"),
        )
```

### 7.2 ToolCacheEntry (Agent Tool Cache)

The Agent's `_tool_cache` uses `ToolCacheEntry` as keys:

```python
# dynamiq/nodes/agents/utils.py

class ToolCacheEntry:
    """Hashable key for tool cache."""
    action: str        # Tool name
    action_input: Any  # Tool input (usually dict or str)
    
    def __hash__(self):
        # Uses frozen dict for hashing
        return hash((self.action, self._freeze(self.action_input)))
```

**Serialization Strategy:**

```python
def _serialize_tool_cache(cache: dict) -> dict:
    """Serialize tool cache with string keys."""
    result = {}
    for entry, value in cache.items():
        # Convert ToolCacheEntry to string key
        key = f"{entry.action}::{json.dumps(entry.action_input, sort_keys=True)}"
        result[key] = value
    return result

def _deserialize_tool_cache(data: dict) -> dict:
    """Deserialize tool cache, reconstructing ToolCacheEntry keys."""
    result = {}
    for key, value in data.items():
        action, input_str = key.split("::", 1)
        action_input = json.loads(input_str)
        entry = ToolCacheEntry(action=action, action_input=action_input)
        result[entry] = value
    return result
```

### 7.3 AgentIntermediateStep

Reasoning trace per loop:

```python
class AgentIntermediateStep(BaseModel):
    input_data: str | dict
    model_observation: AgentIntermediateStepModelObservation
    final_answer: str | dict | None = None

class AgentIntermediateStepModelObservation(BaseModel):
    initial: str | dict | None = None
    tool_using: str | dict | list | None = None
    tool_input: str | dict | list | None = None
    tool_output: Any = None
    updated: str | dict | None = None
```

**Serialization:** Already Pydantic models, use `.model_dump()` directly.

### 7.4 GraphState Context

The GraphOrchestrator's context and chat history:

```python
# Context: dict[str, Any] - must be JSON-serializable
# Chat history: list[dict[str, str]] with role/content

def _validate_context_serializable(context: dict) -> dict:
    """Ensure context values are JSON-serializable."""
    import json
    try:
        json.dumps(context)
        return context
    except TypeError as e:
        # Convert non-serializable values
        return _deep_serialize(context)
```

---

## 7b. File Handling Strategy (Critical)

Files appear in multiple places in Dynamiq workflows. This section documents how to handle each type.

### 7b.1 File Types in Workflows

| Location | Type | Checkpoint Strategy |
|----------|------|---------------------|
| **Agent `files` parameter** | `list[BytesIO]` | Store metadata only, re-upload on resume |
| **Agent `images` parameter** | `list[str\|bytes\|BytesIO]` | URLs: save as-is; bytes: reference only |
| **VisionMessage images** | Base64 data URLs or URLs | Save URL/data URL string directly |
| **E2B sandbox files** | Uploaded to `/home/user/input` | Save filenames, re-upload if sandbox recreated |
| **FileStore** | External storage | Save file IDs/paths, not content |
| **Tool outputs** | `BytesIO` in results | Convert to base64 or external reference |

### 7b.2 Agent Files - Input Files

```python
# Agent receives files via input
class AgentInputSchema(BaseModel):
    files: list[io.BytesIO | bytes] | None = None
    images: list[str | bytes | io.BytesIO] | None = None
```

**Checkpoint Strategy:**

```python
def get_checkpoint_state(self) -> dict:
    """Agent checkpoint - handle files."""
    state = {
        "messages": [...],
        "intermediate_steps": [...],
        # Files: save metadata only (content is large)
        "input_files_metadata": [
            {
                "name": getattr(f, "name", f"file_{i}"),
                "size": len(f.getvalue()) if hasattr(f, "getvalue") else None,
            }
            for i, f in enumerate(self._input_files or [])
        ],
        # Images: URLs save directly, bytes save as reference
        "input_images_metadata": [
            {
                "type": "url" if isinstance(img, str) else "bytes",
                "value": img if isinstance(img, str) else f"image_{i}_ref",
            }
            for i, img in enumerate(self._input_images or [])
        ],
    }
    return state
```

**Resume Strategy:**
- Files must be re-provided by caller on resume
- OR: Store file content in external storage (S3/blob) with reference

### 7b.3 VisionMessage Images

VisionMessages can contain images as:
1. **URLs** - Save directly in checkpoint
2. **Base64 data URLs** - Save directly (already string)
3. **BytesIO** - Converted to base64 before message creation

```python
class VisionMessageImageURL(BaseModel):
    url: str  # Can be "https://..." or "data:image/png;base64,..."
    detail: VisionDetail = VisionDetail.AUTO

def _serialize_vision_content(content: VisionMessageImageContent) -> dict:
    """Serialize vision content - URL is already a string."""
    return {
        "type": "image_url",
        "image_url": {
            "url": content.image_url.url,  # Already string (URL or base64)
            "detail": content.image_url.detail.value,
        }
    }
```

**Note:** Base64 images can be large (MBs). Consider:
- Extracting to external storage for large images
- Setting `checkpoint_max_image_size` threshold

### 7b.4 E2B Sandbox Files

Files uploaded to E2B sandbox live in `/home/user/input`:

```python
class E2BInterpreterTool(ConnectionNode):
    files: list[io.BytesIO] | None = None
    _sandbox: Sandbox | None = None
    
    def get_checkpoint_state(self) -> dict:
        return {
            "sandbox_id": self._sandbox.sandbox_id if self._sandbox else None,
            "installed_packages": self.installed_packages.copy(),
            # Save file metadata for reference
            "uploaded_files": [
                {
                    "name": getattr(f, "name", f"file_{i}"),
                    "sandbox_path": f"/home/user/input/{getattr(f, 'name', f'file_{i}')}",
                }
                for i, f in enumerate(self.files or [])
            ],
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        # If sandbox still exists, files are still there
        # If sandbox expired, files need to be re-uploaded
        self._uploaded_files_metadata = state.get("uploaded_files", [])
        self._needs_file_reupload = True  # Check on next execute
```

### 7b.5 FileStore (Agent File Storage)

Agent's `file_store` uses the Dynamiq `FileStore` abstraction:

```python
# dynamiq/storages/file/base.py
class FileStore(abc.ABC, BaseModel):
    """Abstract base class for file storage implementations.
    
    Supports: in-memory, file system, cloud storage (S3, GCS, Azure), etc.
    """
    def store(self, file_path, content, ...) -> FileInfo: ...
    def retrieve(self, file_path) -> bytes: ...
    def list_files(self, directory, ...) -> list[FileInfo]: ...

# Current implementation
class InMemoryFileStore(FileStore):
    """In-memory file storage (lost on process restart!)"""
    
# Future implementations (not yet in codebase)
# - S3FileStore
# - GCSFileStore  
# - AzureBlobFileStore
# - LocalFileStore
```

**IMPORTANT: FileStore IS Part of Agent State - MUST Checkpoint**

The FileStore is the agent's **working directory**. Files created during execution (tool outputs, intermediate results) live here. This IS execution state.

| FileStore Backend | What to Checkpoint | Restore Behavior |
|-------------------|-------------------|------------------|
| **InMemoryFileStore** | ✅ **FULL FILE CONTENTS** (base64) | Restore entire virtual FS |
| **S3FileStore** (future) | ✅ S3 keys + bucket info | Files already in S3 |
| **LocalFileStore** (future) | ✅ File paths | Verify files exist on resume |

### InMemoryFileStore - MUST Checkpoint Contents

```python
# dynamiq/storages/file/in_memory.py - ADD checkpoint methods

class InMemoryFileStore(FileStore):
    _files: dict[str, dict[str, Any]] = {}  # The virtual file system
    
    def get_checkpoint_state(self) -> dict:
        """Checkpoint the ENTIRE virtual file system."""
        import base64
        
        files_state = {}
        for path, file_data in self._files.items():
            files_state[path] = {
                "content_base64": base64.b64encode(file_data["content"]).decode(),
                "size": file_data["size"],
                "content_type": file_data["content_type"],
                "created_at": file_data["created_at"].isoformat(),
                "metadata": file_data.get("metadata", {}),
            }
        
        return {
            "type": "in_memory",
            "file_count": len(files_state),
            "total_size": sum(f["size"] for f in files_state.values()),
            "files": files_state,
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        """Restore the ENTIRE virtual file system."""
        import base64
        from datetime import datetime
        
        self._files = {}
        for path, file_data in state.get("files", {}).items():
            self._files[path] = {
                "content": base64.b64decode(file_data["content_base64"]),
                "size": file_data["size"],
                "content_type": file_data["content_type"],
                "created_at": datetime.fromisoformat(file_data["created_at"]),
                "metadata": file_data.get("metadata", {}),
            }
        
        logger.info(f"Restored {len(self._files)} files from checkpoint")
```

### S3FileStore (Future) - The Sync Pattern

The FileStore abstraction makes S3 feel "local" to the agent:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Agent                                    │
│   file_store.store("report.csv", data)  ← "local" operation     │
│   content = file_store.retrieve("report.csv")  ← "local" read   │
└────────────────────────────┬────────────────────────────────────┘
                             │ FileStore Abstraction
┌────────────────────────────▼────────────────────────────────────┐
│                      S3FileStore                                 │
│   store() → Upload to S3                                        │
│   retrieve() → Download from S3                                 │
│   _file_registry: dict[path, S3Key]  ← Tracks what exists       │
└─────────────────────────────────────────────────────────────────┘
                             │
                     ┌───────▼───────┐
                     │      S3       │
                     │  (persistent) │
                     └───────────────┘
```

**What to checkpoint:** The registry of what files exist (not content!)

```python
class S3FileStore(FileStore):
    bucket: str
    prefix: str
    # Registry: local_path → S3 key mapping
    _file_registry: dict[str, dict] = {}  # path → {s3_key, size, content_type, ...}
    
    def store(self, file_path: str, content: bytes, ...) -> FileInfo:
        """Upload to S3 and register."""
        s3_key = f"{self.prefix}/{file_path}"
        self._s3_client.upload(self.bucket, s3_key, content)
        
        # Track in registry
        self._file_registry[file_path] = {
            "s3_key": s3_key,
            "size": len(content),
            "content_type": content_type,
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        return FileInfo(...)
    
    def retrieve(self, file_path: str) -> bytes:
        """Download from S3."""
        if file_path not in self._file_registry:
            raise FileNotFoundError(...)
        s3_key = self._file_registry[file_path]["s3_key"]
        return self._s3_client.download(self.bucket, s3_key)
    
    def get_checkpoint_state(self) -> dict:
        """Checkpoint the FILE REGISTRY (not content - it's in S3)."""
        return {
            "type": "s3",
            "bucket": self.bucket,
            "prefix": self.prefix,
            "file_registry": self._file_registry.copy(),  # What files exist
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        """Restore the registry - files are still in S3."""
        self.bucket = state["bucket"]
        self.prefix = state["prefix"]
        self._file_registry = state.get("file_registry", {})
        # Agent can now retrieve() any file - content is in S3
```

### LocalFileStore (Future) - Similar Pattern

```python
class LocalFileStore(FileStore):
    base_path: Path
    _file_registry: dict[str, dict] = {}
    
    def get_checkpoint_state(self) -> dict:
        return {
            "type": "local",
            "base_path": str(self.base_path),
            "file_registry": self._file_registry.copy(),
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        self._file_registry = state.get("file_registry", {})
        # Verify files still exist on disk
        for path, info in self._file_registry.items():
            full_path = self.base_path / path
            if not full_path.exists():
                logger.warning(f"File missing on resume: {full_path}")
```

### The Key Insight

| Backend | What's Checkpointed | Why |
|---------|---------------------|-----|
| **InMemoryFileStore** | Full contents (base64) | Contents lost on crash |
| **S3FileStore** | File registry (path → S3 key mapping) | Contents persist in S3 |
| **LocalFileStore** | File registry (paths) | Contents on disk (verify on resume) |

**The checkpoint preserves the agent's "view" of its file system, not necessarily the raw bytes.**

### Agent Integration

```python
class Agent(Node):
    file_store: FileStoreConfig = Field(...)
    
    def get_checkpoint_state(self) -> dict:
        state = {
            # ... messages, intermediate_steps, tool_cache ...
        }
        
        # CRITICAL: Checkpoint the virtual file system
        if self.file_store.enabled and self.file_store.backend:
            state["file_store_state"] = self.file_store.backend.get_checkpoint_state()
        
        return state
    
    def restore_from_checkpoint(self, state: dict) -> None:
        # ... restore other state ...
        
        # CRITICAL: Restore the virtual file system
        if "file_store_state" in state and self.file_store.backend:
            self.file_store.backend.restore_from_checkpoint(state["file_store_state"])
```

### Size Optimization

```python
# For large file stores, compress before checkpointing
import gzip

def get_checkpoint_state(self) -> dict:
    files_data = self._serialize_all_files()
    
    # Compress if > 1MB
    if self._total_size > 1_000_000:
        compressed = gzip.compress(json.dumps(files_data).encode())
        return {
            "type": "in_memory",
            "compressed": True,
            "data_base64": base64.b64encode(compressed).decode(),
        }
    return {"type": "in_memory", "compressed": False, "files": files_data}
```

**Key Points:**
1. **InMemoryFileStore MUST checkpoint full contents** - files are ephemeral!
2. **S3/cloud backends** - checkpoint the FILE REGISTRY (what exists), not content
3. **The abstraction hides sync** - agent sees "local" files, backend handles S3 uploads/downloads
4. **On resume:** Registry tells agent what files exist; content fetched on-demand from S3
5. **For production** - recommend S3FileStore to avoid large checkpoint payloads

### Checkpoint Workflow Visualization

```
CHECKPOINT (S3FileStore):
┌────────────────────────────────────────┐
│ Agent State                            │
│   messages: [...]                      │
│   intermediate_steps: [...]            │
│   file_store_state:                    │
│     type: "s3"                         │
│     bucket: "my-bucket"                │
│     file_registry:                     │
│       "report.csv": {s3_key: "..."}   │  ← Only metadata
│       "chart.png": {s3_key: "..."}    │
└────────────────────────────────────────┘
        │
        ▼ Save to checkpoint backend
┌────────────────────────────────────────┐
│        Checkpoint Storage              │
│        (Postgres/Redis/File)           │
└────────────────────────────────────────┘

                    Content stays in S3 ─────► ┌─────────────┐
                                               │     S3      │
                                               │ (unchanged) │
                                               └─────────────┘

RESUME:
1. Load checkpoint → Restore file_registry
2. Agent calls file_store.retrieve("report.csv")
3. S3FileStore looks up registry → s3_key
4. Downloads from S3 → Returns bytes to agent
5. Agent continues as if nothing happened
```

### Edge Case: In-Flight Operations

What if checkpoint happens DURING a file operation?

```python
class S3FileStore(FileStore):
    _pending_uploads: dict[str, bytes] = {}  # Files being uploaded
    
    def store(self, file_path: str, content: bytes) -> FileInfo:
        self._pending_uploads[file_path] = content
        try:
            # Upload to S3
            self._s3_client.upload(...)
            # Only add to registry AFTER successful upload
            self._file_registry[file_path] = {...}
        finally:
            del self._pending_uploads[file_path]
    
    def get_checkpoint_state(self) -> dict:
        # Include pending uploads in checkpoint!
        return {
            "type": "s3",
            "file_registry": self._file_registry.copy(),
            "pending_uploads": {
                path: base64.b64encode(content).decode()
                for path, content in self._pending_uploads.items()
            },
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        # Resume any interrupted uploads
        for path, content_b64 in state.get("pending_uploads", {}).items():
            content = base64.b64decode(content_b64)
            self.store(path, content)  # Retry the upload
```

### 7b.6 Tool Outputs with Binary Data

Tools can return binary data in results:

```python
# Tool output with file
return {
    "content": "Analysis complete",
    "files": {
        "chart.png": BytesIO(png_bytes),  # Binary!
    }
}
```

**Checkpoint Strategy:**

```python
def _serialize_tool_output(output: dict) -> dict:
    """Serialize tool output, handling binary files."""
    result = {}
    for key, value in output.items():
        if key == "files" and isinstance(value, dict):
            # Convert BytesIO to references or base64
            result["files"] = {}
            for fname, fdata in value.items():
                if isinstance(fdata, io.BytesIO):
                    # Option 1: Base64 (small files)
                    if fdata.getbuffer().nbytes < MAX_INLINE_FILE_SIZE:
                        result["files"][fname] = {
                            "type": "base64",
                            "data": base64.b64encode(fdata.getvalue()).decode(),
                        }
                    # Option 2: External storage reference (large files)
                    else:
                        file_id = _store_to_external(fdata)
                        result["files"][fname] = {
                            "type": "external",
                            "storage_id": file_id,
                        }
                else:
                    result["files"][fname] = value
        else:
            result[key] = value
    return result
```

### 7b.7 Configuration Constants

```python
# dynamiq/checkpoint/config.py

# Maximum size for inline base64 files in checkpoint (bytes)
CHECKPOINT_MAX_INLINE_FILE_SIZE = 1024 * 1024  # 1MB

# Maximum total checkpoint size before compression
CHECKPOINT_COMPRESSION_THRESHOLD = 10 * 1024 * 1024  # 10MB

# External storage for large files (future)
CHECKPOINT_EXTERNAL_STORAGE_ENABLED = False
```

### 7b.8 Summary: File Handling Decision Tree

```
Is it a file in the checkpoint?
│
├─ URL string (http:// or data:) → Save directly ✅
│
├─ BytesIO/bytes < 1MB → Base64 encode, save inline ✅
│
├─ BytesIO/bytes > 1MB → Options:
│   ├─ Save metadata only, require re-upload on resume
│   ├─ Store in external storage (S3), save reference
│   └─ Compress and save (gzip)
│
├─ FileStore content → DON'T checkpoint (external) ✅
│
└─ E2B sandbox file → Save metadata, re-upload if sandbox expired ✅
```

---

## 8. Runtime Integration Insights (Critical)

From analyzing `runtime/app/worker/execute_run.py`, key patterns for checkpoint integration:

### 8.1 HITL Done Events (Critical)

The runtime uses `done_events` to signal HITL tools to stop waiting:

```python
# From execute_run.py - cleanup pattern
for node_id, done_event in hitl_done_events.items():
    try:
        done_event.set()  # Unblock waiting tools
    except Exception as e:
        logger.warning(f"Error setting HITL done_event: {e}")
```

**Checkpoint Implication:** When checkpointing during HITL wait, we must:
1. Save the pending state
2. NOT signal done_event (workflow pauses, doesn't terminate)
3. On resume, recreate done_event and reconnect to input_queue

### 8.2 WorkflowExecutionContext

Runtime tracks workflow creation time for connection refresh:

```python
class WorkflowExecutionContext:
    def __init__(self, workflow, workflow_uri, ...):
        self.creation_time = datetime.now(timezone.utc)
    
    async def check_and_recreate_workflow(self) -> bool:
        """Recreate workflow if connections expired."""
        latest_update_time = AppCtx.wf_connection_expiration_manager...
        if latest_update_time > self.creation_time:
            # Connections updated since creation, recreate
            self.workflow = await init_workflow_by_uri(...)
```

**Checkpoint Implication:** Include `creation_time` in checkpoint metadata to handle long pauses.

### 8.3 Recursive HITL Node Collection

Runtime recursively finds HITL nodes across all node types:

```python
def _collect_hitl_nodes_recursive(node, input_queue, done_events, nodes_override):
    """Recursively collect HITL nodes, handling:
    - Map (recurse into inner node)
    - ReActAgent (recurse into tools)
    - Orchestrators (recurse into agents/states)
    - GraphOrchestrator (recurse into state tasks)
    """
```

**Checkpoint Implication:** Checkpoint must capture state at all levels of nesting.

### 8.4 Approval Event Format

HITL input uses specific format:

```python
approval_msg = ApprovalStreamingInputEventMessage(
    entity_id=None,  # Matched by tool
    data={"feedback": feedback, "is_approved": is_approved},
    event=APPROVAL_EVENT,
)
sync_queue.put(approval_msg.to_json())
```

**Checkpoint Implication:** Pending approval state should be serialized in compatible format.

---

## 9. Summary Table

| Node | Category | Fields to Save | Resume Strategy |
|------|----------|----------------|-----------------|
| **Converters, Validators** | A | None | N/A |
| **Embedders, Retrievers** | A | None | N/A |
| **Writers, Splitters** | A | None | N/A |
| **Audio, Image nodes** | A | None | N/A |
| **Choice, Pass** | A | None | N/A |
| **Simple Tools** | A | None | N/A |
| **BaseLLM** | B | `_is_fallback_run` | Direct restore |
| **Agent** | C | Messages, steps, cache, loop | Resume from loop |
| **GraphOrchestrator** | C | History, context, state | Resume from state |
| **LinearOrchestrator** | C | Results, task index | Resume from task |
| **Map** | C | Completed iterations | Only run pending |
| **E2BInterpreterTool** | D | Sandbox ID, packages | Try reconnect |
| **HumanFeedbackTool** | D | Pending prompt | Re-emit prompt |
| **MCPServer** | D | Tool names | Rediscover tools |
| **Browser Tools** | D | Session ID, URL | Recreate session |

---

## 10. Critical Implementation Notes

### 10.1 Order of Operations on Resume

1. **Load checkpoint** from backend
2. **Validate** checkpoint version and integrity
3. **Restore Flow state** (completed_node_ids, results)
4. **Restore Node states** in dependency order:
   - Stateless nodes: skip
   - Simple nodes: direct restore
   - Complex nodes: restore with validation
   - External nodes: attempt reconnection
5. **Reinitialize runtime resources** (queues, events, callbacks)
6. **Resume execution** from checkpoint position

### 10.2 Error Handling on Restore

```python
def restore_node_with_fallback(node, state):
    """Restore node state with graceful fallback."""
    try:
        if hasattr(node, 'restore_from_checkpoint'):
            node.restore_from_checkpoint(state)
        else:
            # Apply generic state restoration
            for key, value in state.items():
                if hasattr(node, key):
                    setattr(node, key, value)
    except Exception as e:
        logger.warning(f"Failed to restore state for {node.id}: {e}")
        # Node will re-execute from scratch
        node._is_resumed = False
```

### 10.3 Testing Requirements

| Scenario | Test Case |
|----------|-----------|
| Agent mid-loop | Checkpoint at loop 5 of 15, resume, verify continuation |
| Map partial | Checkpoint with 3 of 10 iterations done |
| E2B expired | Checkpoint with sandbox_id, wait for expiry, resume |
| HITL pending | Checkpoint during approval wait, close browser, resume |
| Orchestrator mid-state | Checkpoint during GraphOrchestrator execution |
| Nested HITL | Agent inside Map with HumanFeedbackTool |

---

**Previous:** [03-RUNTIME-INTEGRATION.md](./03-RUNTIME-INTEGRATION.md)  
**Next:** [05-DATA-MODELS.md](./05-DATA-MODELS.md)
