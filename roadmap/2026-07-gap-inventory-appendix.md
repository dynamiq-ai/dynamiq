# Appendix: Verified Gap Inventory & Dimension Detail (July 2026)

Companion to [`2026-07-framework-gap-analysis.md`](./2026-07-framework-gap-analysis.md). Every gap below was claimed by a cross-framework comparison agent and then **adversarially verified** by an independent agent instructed to refute it by searching the Dynamiq codebase (source, examples, docs, tests). **Confirmed** = capability is absent; **Partial** = present but materially weaker than the competitor bar. 0 of 74 verified claims were refuted outright; low-severity items were catalogued without a verification pass.


## Agent abstractions & reasoning (`reasoning_agents`)

### Gaps

#### [HIGH — confirmed] No per-step or dynamic model control inside a single agent

*Who sets the bar: langgraph, langchain, crewai, deepagents*

LangGraph accepts a callable (state, runtime) -> model for context-dependent model selection per step, LangChain's ModelRequest.override swaps model/tools/settings for a single call, CrewAI supports a dedicated planning LLM separate from the execution LLM, and DeepAgents-code ships ConfigurableModelMiddleware plus a reasoning_effort selector. A Dynamiq Agent has exactly one LLM for every loop iteration (base.py _run_llm calls self.llm.run), with the only alternative being the error-triggered FallbackConfig secondary model. Users cannot route cheap steps to cheap models or use a stronger model for planning/final synthesis — a common cost-optimization ask.

> **Verification:** The Agent class declares exactly one LLM (llm: BaseLLM, base.py line 238) and every loop-step call funnels through _run_llm (base.py lines 1256-1288), which unconditionally invokes self.llm.run — including the ReAct step (agent.py:1729), the final-answer attempt after max loops (agent.py:2091), and even history summarization (components/history_manager.py uses self.llm). Exhaustive greps across dynamiq/, examples/, docs/, and tests/ found no callable-model type, no per-phase LLM field (SummarizationConfig and LongTermMemoryConfig carry no LLM), no runtime/input-schema LLM override, no selector/router/middleware, and BaseLLM._build_completion_params hardcodes "model": self.model (llms/base.py:905) so not even a per-call model kwarg exists. The only alternate model is the error-class-triggered FallbackConfig secondary LLM (llms/base.py:91-118, 1149-1232), exactly as the claim states; different models are only reachable by composing separate agent nodes (SubAgentTool, GraphOrchestrator manager), which is multi-agent delegation, not per-step model selection within one Agent run.

#### [MEDIUM — confirmed] No shipped plan-and-execute strategy or pluggable planner abstraction

*Who sets the bar: crewai, adk*

CrewAI's new default AgentExecutor is an explicit plan -> execute-step -> observe -> replan loop with isolated per-step messages and configurable reasoning effort, and ADK ships pluggable planners (BuiltInPlanner with native thinking config, PlanReActPlanner). Dynamiq has no planner abstraction or alternative executor: planning is limited to TodoWriteTool/ThinkingTool prompt-level task tracking inside the single ReAct loop, matching the LangChain/DeepAgents todo approach but with no upgrade path to structured plan-and-execute for long-horizon tasks.

> **Verification:** The Agent (dynamiq/nodes/agents/agent.py) has no planner plug-in or alternative executor: its only executor-level knob is InferenceMode (DEFAULT/XML/FUNCTION_CALLING/STRUCTURED_OUTPUT), a tool-invocation format for the single ReAct loop, and grepping dynamiq/ for planner/plan_and_execute/replan yields only a Neo4j Cypher query-planner comment; in-loop planning support is exactly TodoWriteTool, ThinkingTool, and prompt blocks (TODO_TOOLS_INSTRUCTIONS, REACT_BLOCK_MULTI_TOOL_PLANNING). The closest near-miss is the multi-agent layer's AgentManager with plan/assign/final actions used by GraphOrchestrator.get_next_state_by_manager, but its "plan" prompt performs one-shot next-state selection in a user-defined state graph — no structured multi-step plan, no replan step, no per-step message isolation — and the plan-and-execute-style Linear/Adaptive orchestrators from older versions are absent in v0.59.0 (orchestrators/__init__.py exports only GraphOrchestrator). Plan-and-execute workflows must be hand-built on GraphOrchestrator, as examples/use_cases/gpt_researcher does with a user-defined planner state, so the framework capability claimed missing is genuinely absent.

#### [MEDIUM — confirmed] No built-in reflection/self-critique loop

*Who sets the bar: deepagents, crewai, adk*

DeepAgents' RubricMiddleware runs an isolated grader sub-agent that scores the transcript against rubric criteria and re-injects revision feedback until satisfied; CrewAI's PlannerObserver performs post-step LLM observation by default; ADK ships ReflectAndRetryToolPlugin for tool-failure reflection. Dynamiq has no runtime critic/grader/reviser machinery — its reflection 'example' (examples/components/agents/agents/reflection_agent_wf.py) is plain role prompting on the standard Agent. Quality-critical deployments must hand-roll evaluator-optimizer loops as multi-node workflows.

> **Verification:** Exhaustive search of dynamiq/ (agents, orchestrators, evaluations, validators, tools, detectors) found no class implementing a runtime critic/grader loop that scores agent output against criteria and forces revisions: the Agent's only self-correction is parse-error recovery on malformed XML/function-call output (agent.py ~617-648), dynamiq/evaluations/ (LLMEvaluator, metric evaluators) are standalone offline scoring utilities never invoked by agent runtime, and dynamiq/nodes/validators/ are deterministic format checkers with no revision loop. The reflection example is confirmed as a plain Agent with a role string and no critique mechanism, and the closest thing to an evaluator-optimizer loop is examples/components/agents/orchestrators/graph_orchestrator/code_assistant.py, where the generate/validate/reflect cycle is entirely hand-rolled from user-defined functions on GraphOrchestrator — exactly the hand-rolling the claim describes. No class named or resembling Reflect/Critic/Rubric/Grader exists anywhere in source, examples, docs, tests, or pyproject extras.

#### [MEDIUM — confirmed] No agent-level output guardrails with automatic retry

*Who sets the bar: crewai, langchain*

CrewAI attaches guardrail/guardrail_max_retries directly to the Agent, auto-compiles string guardrails into LLM-executed checks, and ships a HallucinationGuardrail faithfulness checker; LangChain implements the same via after_model middleware with retry. Dynamiq has validator nodes (regex/JSON/choices/Python) and detector nodes (PII, prompt injection, LlamaGuard) usable as separate workflow nodes, but nothing can be attached to the Agent to validate the final answer semantically and retry with feedback — the only automatic answer retry is JSON-schema parse coercion for response_format.

> **Verification:** The Agent classes in dynamiq/nodes/agents/base.py and agent.py expose no guardrail field, output-validation hook, or any mechanism to run a user-supplied predicate or LLM check on the final answer with automatic retry; the only "guardrail" strings in the repo are a regex write-blocker in cypher_executor.py and the standalone LlamaGuard detector node. Validators (BaseValidator and subclasses) and detectors are standalone workflow nodes with zero integration into the agent loop, callbacks are observational-only, ErrorHandling retries only on exceptions, and a FaithfulnessEvaluator exists only as an offline evaluation metric. The agent loop's automatic final-answer retries are exactly ParsingError recovery from _coerce_to_response_format (parse-only JSON coercion, no schema validation) plus one minor extra the claim omits: an OutputFileNotFoundError retry when declared output files are missing — a built-in mechanical check, not a user-supplied semantic guardrail, so the core claim is accurate.

#### [CRITICAL — partial] No request-modifying middleware/hook system on the agent loop

*Who sets the bar: langchain, adk, crewai, langgraph, deepagents*

All five competitors let user code intercept and modify the agent loop: LangChain's AgentMiddleware (wrap_model_call, before/after_model, wrap_tool_call, jump_to flow control), ADK's 8 before/after model/tool/agent callbacks, CrewAI's before/after llm/tool hooks plus httpx interceptors, LangGraph's pre/post_model_hook nodes, and DeepAgents' middleware-everything assembly. Dynamiq's callback system (dynamiq/callbacks/base.py) is purely observational — on_node_start/end/error/stream — and cannot rewrite messages, prune tools, swap the model, veto a tool call, or short-circuit the loop; any customization beyond config flags requires subclassing the ~2500-LOC Agent monolith. In 2026 this is the standard extension mechanism through which teams add guardrails, routing, PII scrubbing, and custom summarization, so its absence blocks power users and the ecosystem of drop-in extensions competitors enjoy.

> **Verification:** The core gap is real: callbacks in dynamiq/callbacks/base.py are purely observational (Node.run even passes them a dict copy at node.py:1151, so mutation cannot propagate), _run_llm (agents/base.py:1256) calls self.llm.run directly with no hook to rewrite messages/tools/model, and exhaustive greps for middleware/interceptor/wrap_model/before_model/jump_to across source, examples, docs, and tests found nothing. However, the claim overstates that nothing can veto or modify a tool call without subclassing: every Node carries an ApprovalConfig (node.py:273, types/feedback.py), and agent tool execution goes through tool.run → get_approved_data_or_origin (node.py:935-966, invoked from agents/base.py:1681), which gates the tool call before execution and lets the responder cancel it or rewrite whitelisted mutable_data_params — including programmatically via FeedbackMethod.STREAM streaming events; prompt customization without subclassing also exists via set_block/system_prompt_manager (base.py:615-621). So model-request interception and loop redirection are genuinely absent, but limited pre-execution tool-call interception/veto exists, making the claim partially rather than fully correct.

#### [MEDIUM — partial] No callable/provider-style dynamic instructions recomputed per model call

*Who sets the bar: adk, langchain, langgraph*

ADK accepts InstructionProvider callables (plus MCP-sourced instructions) evaluated per invocation with session-state templating, LangChain's @dynamic_prompt computes the system prompt from state at every model call, and LangGraph's prompt can be a callable of state. Dynamiq's role/instructions are strings with Jinja templates resolved once against runtime input params at run start — the system prompt cannot react to evolving loop state (e.g., remaining budget, accumulated findings) mid-run, and cannot be produced by arbitrary user code.

> **Verification:** The claim's core is accurate: role/instructions are pydantic `str | None` fields (base.py:307-316) that cannot be callables, no InstructionProvider/dynamic-prompt mechanism exists anywhere in dynamiq/, examples/, docs/, or tests/, and the system message is rendered exactly once per run as a static Message (agent.py:1163-1193, variables merged from input at base.py:829) — in fact role/instructions are `{% raw %}`-wrapped (manager.py:324-344), making them even more static than claimed. However, the assertion that the prompt "cannot react to evolving loop state (e.g., remaining budget) mid-run" is overstated: `_inject_state_into_messages` (agent.py:1659-1692, called before every LLM call at line 1713) injects fresh AgentState — current/max loop progress and a live-reloaded todo list (agent.py:2121-2144) — into the last user message at each model call, and a new system message is issued on max-loops exhaustion (agent.py:2069-2092). This is a framework-fixed, non-user-programmable weaker form of state-reactive prompting via message injection, so the gap for callable/user-defined dynamic system prompts is real, but loop budget and task-progress state do reach the model every call.

### Low-severity (unverified)

- **No per-tool call limits or model-call budgets beyond max_loops** — LangChain ships ToolCallLimitMiddleware (per-tool or global, thread/run scoped) and ModelCallLimitMiddleware; CrewAI has max_rpm throttling per agent. Dynamiq caps the overall loop with max_loops and limits sub-agent delegation via SubAgentTool.max_calls, but ordinary tools can be called without limit within the loop and there is no RPM throttle, weakening fine-grained cost/runaway controls.
- **No prompt/agent optimization or training tooling** — ADK ships GEPA reflective prompt evolution against eval sets ('adk optimize') and CrewAI has a training loop that injects human-feedback-derived instructions into prompts ('crewai train'). Dynamiq has no evaluation-driven prompt optimization or feedback-training capability in the library; prompt quality iteration is fully manual.
- **No few-shot example machinery** — ADK provides ExampleTool, BaseExampleProvider and a Vertex retrieval-backed example store; LangChain-core has few-shot prompt templates with semantic example selectors. Dynamiq has no abstraction for curating or retrieving few-shot examples into agent prompts — users must paste examples into the role/instructions strings by hand.

### Dynamiq strengths on this dimension

- Four switchable inference modes (DEFAULT text-ReAct, XML, FUNCTION_CALLING, STRUCTURED_OUTPUT) over one identical agent config, with litellm capability checks (supports_response_schema, Bedrock quirk detection) that validate and auto-correct the mode per model — the broadest model-portability story of the six; competitors ship at most two loop styles (CrewAI react-text vs native-FC)
- Model-aware prompt system with per-model-family prompt overrides shipped in-repo (prompts/overrides/gpt.py, gemini.py via AgentPromptManager/registry) — same harness-per-model idea as DeepAgents' HarnessProfile; no other competitor versions its prompts per model family
- Typed loop recovery is unusually robust: ReactStep recovery outcomes, per-inference-mode corrective guidance on parse failures, echo-back of malformed LLM output, and a max-loops fallback answer synthesis that is still coerced to response_format
- Mid-loop checkpointing with pending-tool-call capture and replay (AgentIterativeCheckpointMixin, agents/checkpoint.py) — ReAct loops are resumable at the agent level, including replaying an interrupted tool call after resume
- First-class multimodal agent inputs on AgentInputSchema (images, videos gated by per-model is_video_input_supported, files) — richer than any competitor's agent-level input schema
- ToolParams runtime parameter injection (global/by_name/by_id) merged into tool inputs and hidden from the LLM — secure runtime credential/config injection without prompt exposure
- 28 LLM provider nodes on a litellm base with a patchable model_registry.json (capabilities, context windows, costs synced into litellm) plus FallbackConfig error-class-triggered secondary LLM at the node level
- Declarative YAML round-trip (serializers/loaders/yaml.py + dumpers/yaml.py) covering full agent/workflow configs — in-repo parity with ADK's YAML agents and ahead of LangChain/LangGraph/DeepAgents, which have no declarative config path
- Skills system (Anthropic-skills-style instruction packages from filesystem or platform registry, optionally ingested into the sandbox) — matches CrewAI's SKILL.md loader and DeepAgents' SkillsMiddleware; LangChain/LangGraph/ADK have no equivalent

### At parity

- Structured output enforcement: pydantic/JSON-schema response_format that coexists with tools in FUNCTION_CALLING mode (schema embedded in function schemas, analogous to LangChain's ToolStrategy and ADK's set_model_response), parse-failure retry with corrective instructions, and capability-based mode auto-correction — par with ADK/CrewAI, only slightly behind LangChain's union-schema/AutoStrategy depth
- Todo-list planning: TodoWriteTool plus AgentState todo serialization into observations is equivalent to LangChain's TodoListMiddleware, DeepAgents' write_todos, and CrewAI's todo tracking
- Run-start dynamic instructions: Jinja-templated role/input_message resolved against runtime params is comparable to ADK's {var} session-state templating and CrewAI's system/prompt/response templates (all weaker than per-step callables, noted as a gap)
- Prompt caching: Anthropic cache_control injection points plus cache read/creation token accounting roughly matches ADK's cache-stable static_instruction tier and DeepAgents' cache_control preservation
- Reasoning-model support: thinking_enabled/budget_tokens on the litellm base is at par with ADK's BuiltInPlanner ThinkingConfig and Anthropic thinking support elsewhere
- Loop budget awareness: max_loops with behaviour_on_max_loops RAISE|RETURN and loop-progress state block is comparable to LangGraph's RemainingSteps/IsLastStep managed values
- Execution timeouts: node-level error_handling.timeout_seconds matches CrewAI's max_execution_time
- Model capability metadata: patchable model_registry.json (supports_function_calling/supports_response_schema, context windows, costs) is roughly at par with LangChain's ModelProfile/model-profiles package


## Multi-agent orchestration (`multi_agent`)

### Gaps

#### [HIGH — confirmed] No cross-process agent interop: no A2A (client or server), no agent-as-MCP-server, no remote-agent node

*Who sets the bar: adk, crewai, langgraph*

ADK ships deep native A2A (RemoteA2aAgent consumption, to_a2a() server exposure, agent cards, task stores) plus to_mcp_server(); CrewAI has bidirectional A2A with streaming/push notifications and a UI extension; LangGraph has RemoteGraph for mounting a remotely deployed graph as a local subgraph. Dynamiq's MCP support is strictly client-side (nodes/tools/mcp.py MCPServer node imports tools FROM external servers), and there is no primitive for consuming a remote agent as a node/tool or exposing a Dynamiq agent over any standard protocol — the OSS package contains no HTTP server at all (no FastAPI/Starlette/uvicorn); the CLI only deploys to Dynamiq's commercial platform. As A2A becomes the enterprise interop standard in 2026, teams that need to federate agents across vendors or processes cannot do it with Dynamiq OSS.

> **Verification:** Exhaustive grep across dynamiq/, examples/, docs/, tests/, and pyproject.toml found zero A2A material (no a2a/AgentCard/jsonrpc/.well-known/task-store matches; the only 'a2a' hits are hash substrings in uv.lock) and no a2a-sdk dependency. dynamiq/nodes/tools/mcp.py imports only `from mcp import ClientSession` — its MCPServer node is a client that imports tools FROM external MCP servers over SSE/stdio/streamable-HTTP; there is no mcp.server/FastMCP/to_mcp_server usage anywhere, and the package source contains no fastapi/starlette/uvicorn (those appear only in the `examples` dev dependency-group for hand-rolled example servers). The only agent-composition primitive, SubAgentTool in agent_tool.py, wraps in-process agent instances/factories; clients/dynamiq.py is a tracing-only client, and the CLI (cli/commands/service.py) deploys Docker archives to Dynamiq's commercial platform via /v1/services — so every element of the claimed gap is confirmed.

#### [HIGH — confirmed] No peer-to-peer handoff or swarm primitive

*Who sets the bar: adk, langgraph, langchain*

ADK's transfer_to_agent tool with disallow_transfer_to_parent/peers flags supports swarm-style lateral transfer; LangGraph's Command (including Command.PARENT, returnable from tools) is a full state-update-plus-handoff mechanism across graph boundaries; the langgraph-swarm ecosystem builds on it. In Dynamiq, control always returns to the delegating agent: SubAgentTool calls are request/response, and delegate_final only lets a parent pass a child's answer verbatim to its own caller — it never transfers the conversation or task to a peer that continues independently. Handoff-style architectures (popularized by OpenAI Agents SDK and Claude-style routing) cannot be expressed without contorting the GraphOrchestrator.

> **Verification:** Exhaustive search (handoff, handover, takeover, transfer_to, swarm, goto, Command, route, peer, delegat*, etc.) found no agent-initiated control-transfer primitive. SubAgentTool (dynamiq/nodes/tools/agent_tool.py) is strictly synchronous call-and-return, and delegate_final (agent.py/_should_delegate_final, base.py delegation_allowed, secondary_instructions.py) only returns a child tool's output verbatim as the parent's final answer up the existing call stack — the prompt itself says "returned verbatim as the final output". The only orchestrator in v0.59.0 is GraphOrchestrator, where transitions are decided externally by static edges, condition callables, or the manager LLM — agents in states cannot even mutate context, and the concierge example builds swarm-style routing by hand with a regex-parsing conditional edge and hub-return edges, confirming handoffs require exactly the GraphOrchestrator contortion the claim describes; the only "handoff/takeover" hits in the codebase are shared browser-session arbitration and human browser takeover, not agent control transfer.

#### [HIGH — confirmed] No concurrent branches, joins, or dynamic fan-out inside the LLM-routed graph orchestrator

*Who sets the bar: langgraph, adk, crewai*

LangGraph executes independent branches in parallel per BSP super-step, offers Send for dynamic map-reduce dispatch with custom per-branch state, and defer=True nodes for fan-in joins; ADK has ParallelAgent with isolated branches plus a workflow engine with JoinNode, ctx.run_node() dynamic spawning and max_concurrency; CrewAI has async tasks and and_()/or_() trigger combinators. Dynamiq's GraphOrchestrator maintains a single current state and executes even multiple tasks within one state sequentially in a for loop (graph_state.py execute); parallelism exists only in the separate deterministic DAG layer (thread-pooled Flow, Map node, ParallelToolCallsTool), which is not LLM-routable. Users building fan-out/fan-in agent teams inside the stateful graph must drop down to the DAG layer and lose dynamic routing.

> **Verification:** Every sub-assertion checks out in v0.59.0: run_flow in graph.py advances a single _current_state_id through a sequential loop, _get_next_state returns exactly one GraphState (and add_edge even overwrites next_states to a single destination, precluding unconditional fan-out), and GraphState.execute in graph_state.py runs multiple tasks in a plain sequential for loop while thread-pool executors exist only in the DAG layer (flows/flow.py, Map operator, ParallelToolCallsTool). Exhaustive greps across dynamiq/, examples/, tests/, and docs/ for Send/spawn/dispatch/barrier/join/defer/parallel/concurrent found no concurrency or join primitive anywhere in the orchestrators package, and GraphOrchestrator is the only concrete orchestrator. Corroborating the gap, Dynamiq's own gpt-researcher port implements run_parallel_research as a sequential list comprehension. Minor non-refuting nuances: merge_contexts plus per-task deepcopy gives sequential isolated-branch-with-merge semantics within one state, and ParallelToolCallsTool is LLM-routable parallelism but only for tool calls inside one agent's ReAct loop, not graph states.

#### [MEDIUM — confirmed] Untyped dict graph state without typed schemas or reducer/channel semantics

*Who sets the bar: langgraph, adk, crewai*

LangGraph state is generically typed (StateT/InputT/OutputT) with a channel library of reducers (Topic, BinaryOperatorAggregate, barrier channels) governing concurrent writes; ADK's workflow engine attaches per-node pydantic input/output/state schemas; CrewAI flows support typed Pydantic state. Dynamiq's GraphOrchestrator context is a plain dict merged with lossless-merge validation — no compile-time typing, no declarative merge policies for concurrent writers, and validation failures surface at runtime. This matters more as gap #3 gets fixed, since reducers are what make parallel branch writes safe.

> **Verification:** GraphOrchestrator.context is declared as a plain dict[str, Any] (graph.py line 63) merged via dict-union, and GraphState.merge_contexts (graph_state.py lines 75-93) is an imperative loop that raises OrchestratorError at runtime whenever two task contexts write different values to the same key — exactly as claimed. Exhaustive greps across dynamiq/, examples/, docs/, tests/, and pyproject extras for reducer/aggregator/channel/state_schema/merge_policy/Annotated/operator.add and related synonyms found no API to declare a typed context schema or per-key reducer functions; the only StateInputSchema types the envelope (context: dict[str, Any]) not the contents, and dynamiq.utils.merge is a trivial last-write-wins {**a, **b}. Examples and tests all manipulate context as an untyped dict, confirming the gap; the only conceivable workaround is subclassing GraphState to override merge_contexts, which is not a declared API.

#### [MEDIUM — confirmed] No event-driven flow composition or external event triggers in OSS

*Who sets the bar: crewai, adk*

CrewAI's Flow engine is event-driven at its core (@start/@listen/@router decorators, or_()/and_() combinators, CEL runtime conditions, declarative YAML flows), and ADK ships /trigger/pubsub and /trigger/eventarc endpoints with concurrency control. Dynamiq flows are strictly invocation-driven: DAG dependency ordering plus the Choice operator, with no listener/trigger decorators, no pub-sub channel primitives, and no OSS endpoints for external event ingestion (event-driven execution is deferred to the commercial platform). Teams wanting reactive, trigger-based agent pipelines must build the plumbing themselves.

> **Verification:** Exhaustive grepping of dynamiq/, examples/, docs/, and tests/ for listener/trigger/webhook/pubsub/eventarc/event-bus vocabulary found no event-listener or trigger-based flow composition API: Flow in dynamiq/flows/flow.py runs only via explicit run_sync/run_async calls driving a graphlib.TopologicalSorter loop over NodeDependency edges, and the only 'trigger' hits are LLM fallback triggers and cancellation. Callbacks (on_node_end/on_flow_end) are observability hooks, the Choice operator and GraphOrchestrator.add_conditional_edge are within-run conditional routing, and StreamingConfig.input_queue only feeds input into an already-running node — none fire nodes off completion or external events. No webhook/pub-sub endpoints ship in the package: fastapi/sse-starlette are examples-group-only dependencies used in user-written example servers that explicitly call wf.run, and the CLI's service command deploys to the commercial platform API, consistent with event-driven execution being deferred there.

### Low-severity (unverified)

- **No background/asynchronous subagent dispatch** — DeepAgents' AsyncSubAgentMiddleware gives agents five tools to start, poll, update (mid-flight instruction injection), cancel and list long-running remote subagent tasks whose registry persists in agent state. Dynamiq subagent calls via SubAgentTool block the parent's loop until completion; ParallelToolCallsTool parallelizes within one step but offers no fire-and-forget dispatch, status polling, or cancellation of a running subagent from the parent. Long-horizon delegation (hours-long research or coding tasks) has no first-class pattern.
- **No adapters for embedding third-party framework agents** — CrewAI ships LangGraphAgentAdapter and OpenAIAgentAdapter so foreign agents participate in crews with tool and structured-output conversion; ADK has LangGraphAgent wrapping compiled LangGraph graphs. Dynamiq has no adapter layer for LangGraph, OpenAI Agents SDK, or other frameworks, raising migration friction for teams with existing agent investments who might otherwise adopt Dynamiq for orchestration.

### Dynamiq strengths on this dimension

- Shared execution environments across an agent team: share_sandbox_with_subagents (ALL/AUGMENT scopes with per-subagent isolated working-dir views) and share_browser_session_with_subagents (one live browser page with page-control locking so parent/subagent handoffs don't deadlock) in dynamiq/nodes/agents/shared_session.py — no competitor in this set has team-shared sandbox or browser-session machinery
- SubAgentTool's three instantiation modes (instance reuse, callable factory for fresh isolated agents per call, and dict blueprints resolved through the YAML loader) give per-invocation agent isolation that is fully serializable — richer than ADK's AgentTool or deepagents' task tool specs
- Full YAML serialization round-trip of entire multi-agent systems (workflows, orchestrators, agents, nested subagent blueprints) via Workflow.from_yaml_file/to_yaml_file — broader declarative coverage than CrewAI's declarative flows and absent in LangGraph/LangChain/deepagents
- delegate_final verbatim handoff of a child agent's answer (skipping parent re-summarization and truncation) is a practical answer-fidelity feature the tool-return-based delegation of CrewAI/ADK lacks
- PlanApprovalConfig human approval gate over a manager agent's task plan with an editable task list (types/feedback.py) — a supervisor-specific HITL primitive competitors only approximate with generic interrupts
- Checkpoint/resume depth for long multi-agent runs: per-iteration resume of agent ReAct loops and orchestrator state transitions plus HITL-timeout pending-tool-call replay, with pluggable filesystem/in-memory/PostgreSQL backends (dynamiq/checkpoints/, nodes/agents/checkpoint.py) — iteration-level resume inside a node is finer-grained than ADK's node-level replay
- ParallelToolCallsTool works even for non-function-calling LLM modes (emulated) while using native parallel function calling when available — parallel subagent fan-out is not gated on provider FC support

### At parity

- Agent-as-tool delegation (SubAgentTool with auto-wrapping of plain Agents) is at par with ADK's AgentTool, CrewAI's generated delegation tools, LangChain tool-wrapped agents, and deepagents' task tool
- Supervisor/manager pattern: AgentManager with pluggable plan/assign/final actions plus GraphAgentManager LLM routing matches CrewAI's hierarchical process manager and what LangGraph users assemble from primitives (langgraph-supervisor being a separate repo)
- Deterministic sequential/parallel/conditional composition: dependency-ordered DAG Flow with thread/process executor pools, Choice condition DSL, and Map fan-out is comparable to ADK's Sequential/Parallel agents and LangChain LCEL sequence/parallel/branch/router
- LLM-driven dynamic routing: GraphOrchestrator conditional edges (Python or FunctionTool callables) plus manager-decided next-state selection is comparable to CrewAI @router methods and ADK transfer-based routing (for routing, not for lateral handoff)
- Hierarchical teams via nesting: orchestrators and agents are Nodes, so graphs nest in DAGs and agents own SubAgentTools of agents recursively — equivalent to LangGraph subgraph composition and ADK sub_agents trees
- Checkpointed multi-agent runs with pluggable persistence backends are at par with LangGraph checkpointers and ADK workflow node checkpointing/replay
- Subagent streaming with entity_id/source attribution on events is roughly at par with competitors' flattened nested-agent event streams (LangChain's typed subagent stream handles are somewhat ahead)
- Map-over-list fan-out with per-item cloned sub-workflows and a thread pool covers the common map-reduce case that LangGraph's Send and CrewAI's kickoff_for_each address (though Send remains more general — see the graph-level fan-out gap)


## Tools & MCP (`tools_mcp`)

### Gaps

#### [HIGH — confirmed] No OpenAPI-spec-to-tools generator

*Who sets the bar: adk*

ADK converts OpenAPI specs, Apigee API Hub entries, GCP Application Integration connectors, and Google API discovery docs into full toolsets with per-operation auth. Dynamiq only offers HttpApiCall, a generic single-endpoint HTTP tool, so wiring an internal REST API with 30 operations means hand-authoring 30 tool configs. For enterprise buyers whose internal APIs all have Swagger/OpenAPI specs, spec import is a frequent evaluation checkbox; MCP and Pipedream only partially mitigate because most internal APIs expose neither.

> **Verification:** Exhaustive search of the Dynamiq v0.59.0 repo found zero references to OpenAPI, Swagger, or OAS in any source, docs, examples, tests, dependencies (pyproject.toml/uv.lock), or git history — no spec-parsing code or libraries exist. HttpApiCall in dynamiq/nodes/tools/http_api_call.py targets a single resolved URL per invocation (url = input_data.url or self.url or self.connection.url) and generates no per-operation tools. The only dynamic tool-generation mechanisms are MCPServer.get_mcp_tools() (one MCPTool per MCP server tool) and the Pipedream Connect action runner, which are exactly the partial mitigations the claim already acknowledges and do not parse OpenAPI specs. The claim is accurate: wiring a 30-operation internal REST API requires hand-authoring 30 HttpApiCall configs.

#### [MEDIUM — confirmed] No MCP server exposure (agents/workflows as MCP tools)

*Who sets the bar: adk, langgraph*

ADK ships to_mcp_server() to publish any agent as an MCP server, and LangGraph Server mounts /mcp routes so deployments are consumable from Claude Desktop, IDEs, and other agents. Dynamiq's MCP support (nodes/tools/mcp.py) is client-only. As MCP becomes the default interop surface in 2026, Dynamiq agents cannot be plugged into the growing ecosystem of MCP hosts without custom glue.

> **Verification:** Dynamiq's MCP support is client-only: dynamiq/nodes/tools/mcp.py's confusingly named MCPServer class is a ClientSession-based discovery wrapper (execute() raises NotImplementedError) that consumes external MCP servers as tools, and the only mcp-library imports in package source are mcp.client.* transports in dynamiq/connections/connections.py. There is no to_mcp_server/serve_mcp function, no mcp.server import, and no HTTP/route-mounting code anywhere in the dynamiq package or its CLI that would publish an agent or workflow as an MCP server. The only FastMCP usage in the repo is two standalone toy demo servers (math, weather) under examples/, which serve as targets for the client-side examples and do not expose Dynamiq agents — a minor technical inexactness in the claim's "no FastMCP anywhere" wording, but the claimed capability gap is fully real.

#### [MEDIUM — confirmed] MCP client lacks OAuth and dynamic per-request auth

*Who sets the bar: adk, crewai, deepagents*

ADK's McpToolset supports auth_scheme/auth_credential OAuth and per-request header_provider callables; DeepAgents ships full MCP OAuth login flows with token management; CrewAI adds connection/discovery/execution timeouts, retries, and schema caching. Dynamiq's MCPSse/MCPStreamableHTTP connections accept only a static headers dict set at construction, so connecting to the growing population of OAuth-protected remote MCP servers (Notion, Linear, GitHub remote MCP, etc.) requires users to obtain and refresh tokens out-of-band.

> **Verification:** MCPSse (lines 1515-1535) and MCPStreamableHTTP (lines 1538-1558) in dynamiq/connections/connections.py accept only a static headers dict plus timeouts, and their connect() methods pass headers directly to the MCP SDK's sse_client/streamablehttp_client without ever using the SDK's `auth` parameter (the hook for OAuthClientProvider); dynamiq/nodes/tools/mcp.py (MCPTool/MCPServer) contains no auth, token-refresh, or header-provider logic at all. Exhaustive greps across dynamiq/, examples/, tests/, and pyproject.toml for oauth, refresh_token, header_provider, auth_scheme, auth_credential, OAuthClientProvider, token_storage, pkce, mcp.client.auth, and callable-header patterns found nothing MCP-related; the only OAuth-named class, PipedreamOAuth2, is a static env-var bearer token for Pipedream's REST API, not MCP. The sole "OAuth" MCP example (use_mcp_adapter_tool.py, use_remote_server_oauth) delegates the flow to the external `npx mcp-remote` proxy via MCPStdio, confirming that OAuth-protected remote MCP servers require out-of-band token handling exactly as the claim states.

#### [MEDIUM — confirmed] Agents cannot use provider-native server-side tools

*Who sets the bar: adk, langchain*

LangChain agents accept provider dict tools and map Anthropic server tools (web_search, code_execution, computer use, text editor) and OpenAI Responses built-ins; ADK wires Gemini-native google_search, url_context, and BuiltInCodeExecutor. Dynamiq's Agent requires tools to be Node instances (tools: list[Node]), so provider-executed tools — which are often cheaper, faster, and better integrated than third-party equivalents — are unusable in the agent loop (raw dicts are accepted only at the bare LLM node level with no agent-loop handling of server-tool result blocks).

> **Verification:** Agent.tools is exactly `tools: list[Node] = []` at dynamiq/nodes/agents/base.py:241, and the claim's grep for web_search/computer_use/code_execution/text_editor across dynamiq/nodes/llms/ and dynamiq/nodes/agents/ returns zero hits (the only repo-wide hits are an ActionType classification enum in nodes/types.py and the client-side code_interpreter tool). The agent loop builds only client-executed function schemas from Node input schemas (schema_generator.generate_function_calling_schemas takes list[Node]) and the LLM response handler extracts only message.content and message.tool_calls — no server_tool_use/web_search_tool_result/grounding/annotation block handling anywhere; searches for web_search_preview, computer_use_preview, google_search, url_context, googleSearch, anthropic-beta, litellm.responses, etc. across dynamiq/, examples/, docs/, tests/, and pyproject.toml found nothing. Raw dict tools are indeed accepted only at the bare LLM node (`tools: list[Tool | dict]` in nodes/llms/base.py:251, passed through in _get_response_format_and_tools), and all shipped web-search/code-exec/computer-use capabilities are third-party client-side Node tools (Exa, Tavily, ScaleSerp, E2B/Daytona sandboxes, Cua/E2B desktop), so provider-native server-side tools are unusable in the agent loop exactly as claimed.

#### [MEDIUM — confirmed] No deferred tool loading or LLM-based tool selection for large catalogs

*Who sets the bar: langchain, adk*

LangChain ships ProviderToolSearchMiddleware (defer_loading + provider-native tool search so large catalogs don't send every schema each turn) and LLMToolSelectorMiddleware (small-LLM pre-selection with max_tools caps); ADK has a skills registry with list/search/load tools for dynamically loadable capabilities. Dynamiq sends every attached tool's schema on every LLM call. This gap is amplified by Dynamiq's own Pipedream and MCP breadth: the more tools a Dynamiq agent gets, the worse the token cost, with no built-in mitigation beyond manual include/exclude lists on MCPServer.

> **Verification:** The claim's exact grep returns nothing in dynamiq/ source, and the agent loop confirms the behavior: _init_prompt_blocks calls schema_generator.generate_function_calling_schemas(self.tools) over every attached tool, and _run_react_llm_step passes that full schema list (tools=fc_tools) into every LLM call, with the _run_extra_tools/_shared_sandbox_tools overlays only ever adding tools (LTM, shared sandbox), never selecting a subset. The only catalog mitigation is the static, manual include_tools/exclude_tools filter on MCPServer (mcp.py), exactly as the claim states; broad synonym sweeps (lazy/defer/max_tools/tool_retriev/filter_tools/catalog/top_k) and pyproject extras found no deferred-schema or per-turn tool-selection mechanism. The only near-miss is dynamiq/skills/ plus SkillsTool (list/get of skill instructions/scripts), which lazily loads prompt-level capabilities ADK-style but has no search and does not defer or reduce tool schemas — it is itself one more always-serialized tool — so the core gap stands.

#### [MEDIUM — confirmed] No adapters for LangChain/CrewAI community tools

*Who sets the bar: adk, crewai*

ADK ships LangchainTool and CrewaiTool wrappers, and CrewAI has from_langchain/to_langchain converters, letting users pull from the ~700-integration langchain-community catalog for free. Dynamiq has no such adapter, so its effective catalog is its ~35 built-in tools plus MCP and Pipedream. Users evaluating frameworks often check whether their existing LangChain tools port over; in Dynamiq each one must be re-wrapped as a Node by hand.

> **Verification:** Exhaustive search of source, examples, docs, tests, dependency files, and full git history found no LangChain or CrewAI tool adapter: grep for langchain in dynamiq/ matches only two docstring comments in the splitters package plus one docstring example query string in firecrawl_search.py, and from_langchain/to_langchain/LangchainTool/CrewaiTool/langchain_core/langchain_community appear nowhere in the repo. pyproject.toml has no langchain/crewai dependency in any group or extra (the only lock-file mentions are transitive opentelemetry-instrumentation packages pulled in by agentops for examples), and no commit in git history ever touched such an adapter. The only generic extension paths are FunctionTool (manual re-wrapping of a Python callable), MCP, and Pipedream nodes, exactly as the claim's context states; the claim's sole inaccuracy is the trivial third grep hit outside splitters/, which does not affect the capability question.

#### [HIGH — partial] No programmatic tool-call interception (hooks/middleware/wrappers)

*Who sets the bar: adk, crewai, langchain, langgraph, deepagents*

Every competitor lets code intercept a tool call to mutate arguments, block execution, or override results: CrewAI before/after tool hooks (block via HookAborted, mutate args, override results), LangChain wrap_tool_call middleware chains, LangGraph wrap_tool_call interceptors, ADK plugins and on_tool_error callbacks. Dynamiq's callbacks (on_node_start/end/error) are observational tracing hooks, and the only interception point is ApprovalConfig, which requires a human feedback round-trip. This blocks common patterns like programmatic arg sanitization, PII scrubbing at the tool boundary, custom per-tool retry logic, and policy-based tool blocking without a human in the loop.

> **Verification:** The claim's specific assertions are accurate: dynamiq/callbacks/base.py has only observational on_node_*/on_workflow_* callbacks whose return values node.py ignores (dispatch even passes dict copies), no before_tool/after_tool/wrap_tool hook exists anywhere, there is no way to programmatically replace a tool's result (OutputTransformer is jsonpath-only), and ApprovalConfig is the only per-call block/mutate point and requires a console or streaming feedback round-trip. However, the headline "no API lets user code programmatically mutate tool arguments" is overstated: Node.inputs()/input_mapping (node.py:606-624, 1930-1980) accepts Python callables that compute/override named input fields and is applied via transform_input on every tool.run() including agent-invoked runs (agents/base.py:1681 -> node.py:1148), enabling programmatic per-argument sanitization at the tool boundary; ToolParams (agents/base.py:159, applied in _run_tool) merges caller-supplied arg overrides per tool name/id at run time; and per-tool retry/backoff/timeout exists via ErrorHandling config (node.py execute_with_retry). These are limited, wiring-oriented mechanisms — no clean blocking contract, no result override, no dynamic wrap-style hook — so the gap is real but narrower than claimed.

### Low-severity (unverified)

- **No generic per-tool usage limits or rate limiting** — CrewAI has max_usage_count on any tool and LangChain has ToolCallLimitMiddleware (thread/run-scoped per-tool or global limits). Dynamiq's max_calls exists only on SubAgentTool, so there is no way to cap how often an agent invokes an expensive search or scrape tool per run, and no rate limiter exists at the tool level.
- **@function_tool does not parse docstrings into per-argument schema descriptions** — LangChain's @tool parses Google/NumPy-style docstrings into per-arg descriptions (with validation), and ADK's FunctionTool derives declarations from docstrings. Dynamiq's @function_tool builds the input schema purely from the signature's type hints and defaults, appending the raw docstring to the tool description — so argument-level guidance never reaches the model unless the author hand-writes a pydantic schema. Minor polish gap that affects tool-call accuracy for decorator-authored tools.

### Dynamiq strengths on this dimension

- Pipedream bridge (nodes/tools/pipedream.py): thousands of SaaS actions exposed as schema-typed tools with dynamically built pydantic schemas from action props and per-user OAuth (PipedreamOAuth2) — long-tail integration coverage no audited OSS competitor matches without a paid platform (CrewAI's 16 'apps' require the paid AMP platform)
- Uniform tool-level controls for free: every tool inherits ErrorHandling (timeout, max_retries, backoff, RAISE|RETURN), Redis-backed result caching (CacheConfig with TTL/namespace), and ApprovalConfig human gates from the Node base class — first-party per-tool result caching that LangChain (none first-party), ADK, LangGraph (policy exists but no bundled tools), and DeepAgents all lack
- Desktop computer-use and AI browser breadth: CuaDesktopTool (cloud/Docker desktop), E2BDesktopTool (VNC desktop), and Stagehand (Browserbase/Steel with live-view URLs for user takeover) — richer than CrewAI core, LangChain, LangGraph, and DeepAgents (which have no desktop computer-use at all); only ADK is comparable
- MCP JSON-Schema-to-pydantic conversion (_SchemaModelBuilder in nodes/tools/mcp.py) handles allOf merging, ref dedup, and unions — more robust than typical adapter-level conversion, reducing broken schemas from real-world MCP servers
- Sandbox subsystem (dynamiq/sandboxes/): E2B/Daytona/E2BDesktop backends with shell execution, file upload/collect/store lifecycle, per-agent sandbox views, and public preview URLs, plus graduated in-process options (RestrictedPython, PythonMonty) — with rate-limit-aware creation retries built in
- Agent-loop tool error recovery: recoverable ToolExecutionExceptions are fed back to the model as observations for self-correction, tool outputs are truncated to configurable token caps, and oversized outputs are persisted to the sandbox instead of blowing the context window
- Skills system (SkillsTool + BaseSkillRegistry with sandbox ingestion of SKILL.md/scripts) for dynamically loadable capabilities — only ADK has an equivalent skills registry among the six frameworks
- Multi-warehouse SQL (MySQL/PostgreSQL/Snowflake/Redshift/Databricks via one SQLExecutor) and graph-DB CypherExecutor (Neo4j/Neptune/Apache AGE) — graph query tooling none of the other five ship

### At parity

- Tool authoring basics: @function_tool decorator (sync+async, signature-derived schemas) plus class-based Node with pydantic input_schema is functionally equivalent to LangChain's @tool/StructuredTool, ADK's FunctionTool, and CrewAI's @tool/BaseTool (minus docstring arg parsing, noted as a low gap)
- MCP client transports: stdio, SSE, and streamable HTTP with include/exclude tool filtering and per-tool node expansion — at par with CrewAI's native client and ADK's McpToolset on transports (auth and resiliency extras are the noted gap)
- Per-tool human approval: ApprovalConfig with editable inputs and feedback methods is comparable to ADK's require_confirmation round-trip and LangChain's HumanInTheLoopMiddleware approve/edit/reject
- Code-interpreter tools: E2BInterpreterTool/DaytonaInterpreterTool match CrewAI's E2B/Daytona tools and ADK's container/cloud executors for mainstream use (ADK's six-backend range, including gVisor-on-GKE, is broader at the extreme)
- Web search/scrape vendor coverage: Tavily, Exa, Firecrawl, Jina, ZenRows, ScaleSerp cover the mainstream vendors, though CrewAI's ~15-vendor scraping/browser lineup is broader
- Harness-style agent-utility tools (TodoWriteTool, ThinkingTool, HumanFeedbackTool, RememberFact/RecallFacts, file read/write/search/list, SummarizerTool, ContextManagerTool) match the DeepAgents core-tool surface
- Per-tool retry/backoff/timeout config is equivalent in capability to LangChain's ToolRetryMiddleware and LangGraph's RetryPolicy/TimeoutPolicy, just expressed as node config rather than middleware


## Memory & state (`memory_state`)

### Gaps

#### [HIGH — confirmed] No typed state schemas or reducer-based state merging for workflow/orchestrator state

*Who sets the bar: langgraph, langchain, adk, crewai*

LangGraph (TypedDict/Pydantic/dataclass schemas with Annotated reducer channels), LangChain 1.x (composable per-middleware typed state slices), ADK (pydantic state_schema validated at graph build), and CrewAI (Flow[StateModel] generic with typed Pydantic state) all give users typed, validated, composable state with declarative merge semantics. Dynamiq's GraphOrchestrator context is a plain dict[str, Any] whose parallel-branch merge simply raises on conflicts instead of applying reducers. This is the state-modeling DX that LangGraph normalized; users evaluating orchestration frameworks in 2026 expect it, and its absence makes complex parallel/branching graph state error-prone in Dynamiq.

> **Verification:** All three assertions verify verbatim: StateInputSchema.context is dict[str, Any] (graph_state.py line 22) and GraphOrchestrator.context is dict[str, Any] = {} (graph.py line 63); merge_contexts (graph_state.py lines 75-93) raises OrchestratorError on any key whose values differ across parallel-task contexts, with no reducer hook or merge-strategy option. Exhaustive searches across dynamiq/, examples/, tests/, docs/, and pyproject.toml for state_schema, TypedDict, Annotated, reducer, merge_strategy, context_schema, StateModel, operator.add, channels, and Flow generics found nothing — the only Pydantic state models are per-node input_schema validation, checkpoint serialization states, and agent-internal loop tracking, none of which is a user-definable typed schema for shared graph state. Minor nuances (merge_contexts tolerates identical duplicate values; per-node input schemas are Pydantic-validated) do not amount to the claimed capability.

#### [HIGH — confirmed] Long-term memory has no automatic extraction, scoring, or consolidation intelligence

*Who sets the bar: crewai, adk (cloud), langgraph (TTL + langmem ecosystem)*

CrewAI's unified memory runs LLM analysis on remember() (inferring scope, categories, importance), scores recall with a tunable composite of 0.5 semantic + 0.3 recency-decay + 0.2 importance, consolidates near-duplicates in background flows, and does LLM-driven multi-round adaptive recall; ADK's Vertex Memory Bank auto-generates and consolidates memories from session events and builds user profiles (cloud). Dynamiq's long-term memory is a flat Fact store written only when the agent explicitly calls RememberFactTool, retrieved by pure embedding similarity with no importance or recency weighting, no automatic extraction from transcripts, no consolidation beyond near-duplicate semantic upsert, and no per-item TTL/retention (which LangGraph's store has). The good substrate (5 vector backends) exists, but the intelligence layer that competitors headline is missing.

> **Verification:** Exhaustive search confirms the gap: backend.remember() is called only from RememberFactTool (the sole write path; agent wiring in base.py/agent.py just registers the remember/recall tools when ltm_enabled), the Fact schema has no importance field and metadata is stored verbatim with no LLM analysis, recall is pure cosine similarity (pgvector ORDER BY embedding distance; multi-query merge by max score) with no recency or importance term, the only consolidation is the synchronous >=0.85 semantic upsert inside remember() that the claim already concedes, and TTL exists only in the cache and checkpoint subsystems — never in any memory backend. Minor nuances that do not refute the claim: the tool wiring lives mainly in nodes/agents/base.py rather than agent.py, a TimeWeightedDocumentRanker exists but only as an unwired RAG document-ranker node, and RecallFactsTool accepts multiple LLM-supplied query phrasings in one call (a weak, single-round form of multi-angle recall).

#### [MEDIUM — confirmed] No state-edit or rewind API on checkpoints before resume

*Who sets the bar: langgraph, adk*

LangGraph offers update_state(values, as_node) and bulk_update_state to surgically edit any historical checkpoint as if a node had written it, and ADK's Runner.rewind_async computes reverse state and artifact deltas to roll a session back. Dynamiq can resume/fork from any checkpoint ID, but the only way to alter state first is to manually mutate a FlowCheckpoint object and pass it to resume_from — no first-class, validated edit API. This weakens human-in-the-loop correction workflows ('fix the agent's wrong tool output and continue') that are a common production pattern.

> **Verification:** Exhaustive grepping for update_state/bulk_update/rewind/edit/patch/as_node/delta/fork/rollback and enumeration of every checkpoint-related method confirms Dynamiq has no validated state-edit API: the flow's public checkpoint surface is only list/get_latest/get_pending_inputs/delete/clear_all plus resume_from on run_sync/run_async, and backends expose only raw save/load/update/delete/get_chain persistence. FlowCheckpoint offers in-memory mutators (mark_node_complete, mark_pending_input) but no schema validation, node-attributed edit semantics, or reverse-delta computation, and the APPEND-mode parent_checkpoint_id chain supports only read/resume time travel. The repo's own tests (test_iterative_checkpoint.py lines 236-247) demonstrate the claimed sole mechanism: hand-mutating FlowCheckpoint fields, re-saving via backend.save, then passing it to resume_from.

#### [MEDIUM — confirmed] Checkpoints are full JSON snapshots — no delta/incremental storage

*Who sets the bar: langgraph, deepagents*

LangGraph's DeltaChannel stores per-checkpoint deltas plus periodic snapshots (with dedicated Postgres/SQLite support), and DeepAgents applies a delta reducer to messages cutting checkpoint growth from O(N^2) to O(N). Dynamiq's APPEND mode (the time-travel mode) serializes the complete FlowCheckpoint — including full message history — as JSON on every save, so long-running agents with per-node and mid-loop checkpointing enabled accumulate quadratic storage. This is the classic checkpoint-bloat failure mode LangGraph explicitly engineered around.

> **Verification:** The claim is accurate: in APPEND mode _save_checkpoint_unlocked (checkpoint.py lines 453-490) model_copies the entire FlowCheckpoint and every backend serializes it whole — PostgreSQL via json.dumps(payload) at postgresql.py line 194 into one JSONB column, FileSystem via json.dump(to_dict()), InMemory via deep copy — and the agent/orchestrator iteration states embed the full prompt message list / chat_history in each snapshot's internal_state. Exhaustive greps across dynamiq/, examples/, docs/, tests/, and pyproject for delta, diff, incremental, patch, reducer, DeltaChannel, dedup, compression, snapshot-interval, etc. found nothing checkpoint-related; the CheckpointBackend interface has no partial-write method. The only mitigations are retention pruning (max_checkpoints=50 default, max_ttl_minutes via cleanup_by_flow) and exclude_node_ids, which cap how many full snapshots are kept but do not reduce per-snapshot size or write cost — so the per-save O(history) serialization and quadratic growth within the retained chain is real, merely bounded in total by retention.

#### [MEDIUM — confirmed] Thin checkpoint-backend ecosystem: no SQLite or Redis saver, no third-party conformance suite

*Who sets the bar: langgraph, adk*

LangGraph ships InMemory, SQLite (sync+async), Postgres (sync+async+shallow) savers plus a published conformance test-suite package that spawned an ecosystem of Redis/MongoDB/etc. savers; ADK has five session-service backends including Firestore and a multi-dialect SQLAlchemy service. Dynamiq has exactly three checkpoint backends (InMemory, Filesystem, PostgreSQL) — notably no SQLite even though its conversation memory has one — and no conformance kit to encourage community backends. Local-dev durability defaults to files, and teams on Redis/MySQL/Firestore have no first-party path.

> **Verification:** The backends directory contains exactly base.py, in_memory.py, filesystem.py, and postgresql.py — three concrete backends (InMemory, FileSystem, PostgreSQL) and nothing else; exhaustive greps for sqlite/redis/mysql/mongo/firestore/dynamodb across dynamiq/, examples/, docs/, and tests/ find those technologies only in the separate memory (sqlite.py, dynamo_db.py) and cache (redis.py) subsystems, which implement different interfaces than the CheckpointBackend ABC. No conformance/compliance kit is published: pyproject extras are only 'cua' and 'monty', and the wheel ships only the dynamiq/ package. The one nuance is an internal BackendTestMixin ("Shared tests for all checkpoint backends") in tests/unit/checkpoints/test_backends.py, but it is not distributed, not importable from the installed package, and not documented for third-party backend authors — so the claim that no conformance suite is published stands.

### Low-severity (unverified)

- **No at-rest encryption for checkpointed state** — LangGraph provides an EncryptedSerializer (AES via pycryptodome) so checkpoints — which contain full conversation history and tool outputs — can be encrypted at rest in any backend. Dynamiq serializes checkpoints as plaintext JSON with no encryption hook, leaving sensitive conversation state exposure to whatever the database offers. Relevant for enterprise/regulated deals where agent state contains PII.
- **No OSS tooling to inspect or migrate checkpoints and memory** — CrewAI ships Textual-based TUIs for browsing memory records and checkpoints plus CLI replay/log-tasks-outputs commands, and ADK has an 'adk migrate session' CLI for moving sessions between backends. Dynamiq's CLI covers platform deployment (org/project/service) only; inspecting checkpoints or memory in OSS means writing SQL or reading JSON files, with observability positioned in the paid platform. This adds friction to debugging time-travel and memory behavior for OSS users.

### Dynamiq strengths on this dimension

- Broadest first-party conversation-memory backend set: 8 in-repo backends (InMemory, SQLite, PostgreSQL, DynamoDB, Pinecone, Qdrant, Weaviate, plus the managed Dynamiq platform backend) with ALL/RELEVANT/BOTH retrieval strategies — LangChain/LangGraph ship conversation persistence in separate packages, ADK has no vector-search session memory, DeepAgents uses files.
- Only framework with in-repo embedding-based semantic long-term memory usable fully in OSS: user-scoped Fact store across 5 backends (InMemory, PGVector, Pinecone, Qdrant, Weaviate) with semantic-upsert dedup (CREATED/UPDATED/UNCHANGED) and auto-registered RememberFact/RecallFacts agent tools — ADK's OSS memory is keyword matching, its semantic memory is paid Vertex; LangChain/LangGraph defer to external langmem; DeepAgents has only file-based memory.
- Mid-agent-loop checkpoint granularity: IterativeCheckpointMixin persists loop iteration counts, pending tool calls, and message history so a ReAct agent resumes at the exact iteration, with per-tool CheckpointState classes (SubAgentTool, LLM, code interpreter, summarizer, context manager) — matches LangGraph-level durability and exceeds ADK and DeepAgents, which checkpoint only at session/graph level.
- Rich declarative checkpoint triggers (checkpoint_on_start/after_node/on_failure/on_cancel/mid_agent_loop/on_input_timeout) configurable at flow and per-run level — more granular policy control than LangGraph's every-superstep model without custom code.
- OSS-to-managed continuity in one interface: the Dynamiq platform memory backend is a drop-in peer of the OSS backends, so moving from self-hosted to cloud-managed memory is a config change rather than an architecture change (ADK's equivalent requires adopting Vertex services with a different feature set).

### At parity

- Time-travel and forking: APPEND-mode snapshots plus resume_from any checkpoint ID or FlowCheckpoint object is functionally at par with LangGraph's replay-then-fork from checkpoint_id and CrewAI's Crew.fork (though Dynamiq records no explicit branch/parent lineage metadata like CrewAI's parent_id chain).
- Durable Postgres-backed checkpointing with indices and fetch modes is at par with LangGraph's PostgresSaver and ADK's DatabaseSessionService for core persistence and crash recovery.
- Cross-session continuity: first-class session_id/user_id on AgentInputSchema plus persistent memory and checkpoint backends is at par with ADK's session services and LangGraph's thread_id-scoped checkpointers.
- Agent-facing memory tools (remember/recall with proactive-use prompting) are at par with CrewAI's Recall/Remember tools and ADK's load_memory/preload_memory tools.
- Persisted agent plan state (AgentState todos checkpointed across loop iterations) is at par with LangChain's PlanningState middleware and DeepAgents' todo/filesystem state.


## Context engineering (`context_mgmt`)

### Gaps

#### [HIGH — confirmed] No pluggable middleware/hook pipeline for custom context transforms

*Who sets the bar: langchain, deepagents, adk, langgraph*

LangChain 1.x middleware, DeepAgents' middleware stack, ADK's plugin before_model callbacks, and LangGraph's pre_model_hook + message reducers all let users inject arbitrary context policies (custom trimming, tool-result editing, PII redaction, cache-breakpoint placement) around each model call. Dynamiq's compaction and truncation logic is hardcoded inside Agent/HistoryManagerMixin (dynamiq/nodes/agents/agent.py, components/history_manager.py); the only extension points are node-level input/output transformers and tracing callbacks, neither of which can rewrite the in-loop message history. Users who need a context policy Dynamiq didn't anticipate must fork or subclass Agent internals.

> **Verification:** Exhaustive search (middleware, hook, pre_model/before_model, interceptor, Callable fields, message filters, docs and examples) found no user-pluggable mechanism to rewrite the agent's in-loop message history or per-call request: the ReAct loop hardcodes message assembly and calls llm.run directly (agent.py:1694-1795, base.py:1256-1288), callbacks are fire-and-forget with return values ignored (node.py:1804-1823), and InputTransformer/OutputTransformer are declarative JSONPath mappings that the agent's internal llm.run(prompt=...) bypasses. Compaction lives in HistoryManagerMixin with a thresholds-only SummarizationConfig, summarization is dispatched via an isinstance(ContextManagerTool) check, and tool-output truncation is fixed logic in process_tool_output_for_agent with only size/on-off knobs — customization requires subclassing Agent, ContextManagerTool, or BaseLLM (get_messages/update_completion_params), exactly as claimed. Minor caveats that fall short of refutation: the LLM node passes its live prompt_messages list to on_node_execute_run callbacks before the litellm call (an incidental, undocumented in-place mutation channel used only for tracing), and Anthropic cache-breakpoint placement is a config knob (cache_control.cache_injection_point_index), so one item in the claim's context (cache breakpoints) is partially covered by static config rather than a hook.

#### [HIGH — confirmed] Prompt-cache management is minimal, Anthropic-only, and compaction is cache-oblivious

*Who sets the bar: adk, crewai, langchain, deepagents*

ADK manages explicit Gemini context caches (fingerprinting, TTL, invalidation, a performance analyzer, static_instruction prefix stability); CrewAI sets cache breakpoints for Anthropic, Gemini, and Bedrock; LangChain ships AnthropicPromptCachingMiddleware; DeepAgents installs caching by default and deliberately orders middleware so prompt mutations don't invalidate cache prefixes. Dynamiq has only an opt-in AnthropicCacheControl on the Anthropic LLM node (a single static cache_control injection point at message index -2, dynamiq/nodes/llms/anthropic.py) plus passive cache-token usage accounting. Nothing protects the cacheable prefix: auto-compaction rewrites the message list in place, silently invalidating provider caches — a direct token-cost hit for long-running agents.

> **Verification:** Exhaustive greps across dynamiq/, examples/, docs/, tests/, and pyproject.toml (cache_control, cachePoint, cached_content, CachedContent, genai, breakpoint, prompt_caching, prefix/fingerprint terms) found exactly one prompt-cache mechanism: the Anthropic node's opt-in AnthropicCacheControl (default None), which injects a single cache_control_injection_points entry at message index -2 in update_completion_params. The Gemini, Bedrock, and VertexAI nodes are bare BaseLLM subclasses with no cache configuration, there is no Gemini cached-content lifecycle (no google-genai dependency, no TTL/invalidation/fingerprinting), and everything else cache-named is unrelated (Redis node-output caching in dynamiq/cache/, tool-result reuse via _tool_cache, passive cache-token usage/pricing accounting in base.py and model_registry.json). Auto-compaction (_compact_history in history_manager.py, triggered by is_token_limit_exceeded in agent.py) rewrites self._prompt.messages in place with no cache-prefix awareness or mutation-ordering safeguards — the message 'static' flag only controls Jinja templating, not prefix stability — so every sub-claim is confirmed.

#### [HIGH — confirmed] Oversized-output offload requires a paid remote sandbox; no local or persistent VFS backend

*Who sets the bar: deepagents, langchain*

DeepAgents' BackendProtocol offers StateBackend, FilesystemBackend (local disk), StoreBackend (persistent), and CompositeBackend routing; LangChain has checkpointer-persisted state VFS and filesystem variants. Dynamiq's FileStore ships only InMemoryFileStore (dynamiq/storages/file/ contains just base.py and in_memory.py), and the flagship auto-offload of oversized tool outputs (process_tool_output_with_sandbox_persistence) targets only a sandbox backend — E2B or Daytona, both paid third-party services. Without a sandbox attached, outputs over the threshold are destructively truncated, and in-memory working files vanish at process exit, so the differentiated offload story degrades sharply in local/default deployments.

> **Verification:** The gap is real: dynamiq/storages/file/ contains only base.py and in_memory.py, and InMemoryFileStore (dict-backed, "Files are lost when the process terminates") is the sole concrete FileStore in the repo — no local-disk, S3, or database implementation exists anywhere (boto3 appears only in the DynamoDB conversation-memory backend). process_tool_output_with_sandbox_persistence offloads only when a sandbox is attached (guard at utils.py lines 1200-1201), and the only caller in base.py passes sandbox=self.sandbox_backend with save_tool_output_to_sandbox gated on the sandbox existing; the file_store backend is never an offload target, and the only shipped sandboxes are E2B/E2B-Desktop/Daytona (third-party services). One nuance: without a sandbox, destructive truncation occurs only above tool_output_max_length (~256K chars by default), not at the 7,000-char offload threshold — outputs between those sizes enter context in full — but above that ceiling truncation is destructive with the full content persisted nowhere, so the claimed capability is genuinely absent.

#### [MEDIUM — confirmed] Compaction is destructive: evicted history is not offloaded anywhere retrievable

*Who sets the bar: deepagents, adk*

DeepAgents writes the full evicted conversation (including extracted media) to /conversation_history/{thread_id}.md on its virtual filesystem, so the agent can dereference details later via read_file; ADK retains raw events in the session service and only swaps in compaction records at prompt-build time. When Dynamiq compacts (auto or via ContextManagerTool), the summarized messages are simply deleted from the prompt — only the summary, the verbatim 'notes' field, and the pinned original input survive. Any detail the summarizer dropped is permanently lost to the run, which matters for long-horizon tasks where the agent later needs an exact value it saw 50 steps ago.

> **Verification:** Dynamiq's _compact_history (history_manager.py) simply truncates self._prompt.messages and re-inserts only the LLM summary, the verbatim ContextManagerTool 'notes' field, the pinned original request, and the recent preserved tail — evicted messages are written nowhere, and the framework's own tool description and system prompt tell the agent to save critical info in 'notes' BEFORE compaction because "previous messages will be removed". Memory persistence (_save_history_to_memory) snapshots only the post-compaction prompt at run end and replace_messages overwrites the session scope, so evicted content never reaches memory either; checkpoints serialize prompt state for resume only and are not agent-readable. The closest adjacent mechanisms do not refute the claim: large tool outputs can be dumped to sandbox files at execution time, but that path is opt-in via is_output_persisted_in_sandbox_allowed which defaults to False and is never enabled anywhere in the codebase, and the in-run _tool_cache holds verbatim results only in a process-local dict requiring an exact (action, action_input) replay — neither is a compaction-time write to a file store, sandbox file, or session log. No DeepAgents-style conversation-history file or ADK-style raw-event retention exists anywhere in dynamiq/, examples/, docs/, or tests/.

#### [MEDIUM — confirmed] No reactive recovery from provider context-overflow errors

*Who sets the bar: crewai, deepagents*

CrewAI catches LLMContextLengthExceededError and summarizes the transcript in chunks; DeepAgents' ContextOverflowError fallback both summarizes and clips the trailing tool-message batch. Dynamiq's compaction is purely proactive (token counting against thresholds before the call); if a request still overflows — token-count drift, a single huge message, provider-side counting differences — the run fails. The LLM fallback system only triggers on rate-limit/connection/any errors and swaps models rather than shrinking context.

> **Verification:** Repo-wide greps for ContextWindowExceeded/ContextOverflow/context-length terms return zero hits; dynamiq/nodes/llms/base.py imports only APIConnectionError, BudgetExceededError, InternalServerError, RateLimitError, ServiceUnavailableError, and Timeout from litellm.exceptions, and FallbackTrigger (lines 85-88) contains exactly ANY/RATE_LIMIT/CONNECTION with indicator lists covering only rate-limit and connection strings — fallback re-runs the same input on another model without shrinking context, and the only reactive retry in the LLM node (_recover_completion_params) handles unsupported sampling params, not context length. The agent loop catches only OutputFileNotFoundError/ParsingError/ActionParsingException, while _run_llm (agents/base.py:1281-1283) raises a plain ValueError on any LLM failure that propagates and fails the run. Compaction is purely proactive: _try_summarize_history fires on is_token_limit_exceeded(), a litellm token_counter threshold check (disabled by default via SummarizationConfig.enabled=False), never in response to a provider overflow error — so the claim is fully confirmed.

#### [MEDIUM — partial] No lightweight in-place context editing (clearing old tool results/args)

*Who sets the bar: langchain, deepagents, adk*

LangChain's ContextEditingMiddleware/ClearToolUsesEdit prunes stale tool results (mirroring Anthropic's clear_tool_uses_20250919) without touching the rest of the conversation and without an LLM call; DeepAgents' TruncateArgsSettings truncates historical tool-call args as a cheaper pre-compaction pass; ADK's ContextFilterPlugin trims to last-N invocations. Dynamiq's only way to shrink live history is full summarization-based compaction, which costs an LLM call and rewrites the conversation. A zero-cost 'clear old tool uses' pass is the standard first line of defense in 2026.

> **Verification:** The strict core of the claim holds: Dynamiq has no retroactive, selective clearing/replacement/truncation of historical tool results or tool-call args in the live in-loop history — the only mid-run shrink path is LLM-summarization compaction (ContextManagerTool -> _compact_history; SummarizationConfig has no truncate-only mode, and no clear_tool_uses/context_management equivalent exists anywhere). However, the claim's framing that summarization is Dynamiq's ONLY defense is overstated: a default-on, zero-LLM-cost tool-output truncation pass (tool_output_truncate_enabled, process_tool_output_for_agent/prepare_tool_output with 64k-token cap and optional sandbox offload plus preview) bounds every tool result as it enters history, and cross-turn history is filtered at zero cost via MemorySaveMode.INPUT_OUTPUT (drops all tool trace from persisted history) and memory_limit/MemoryRetrievalStrategy (last-N-message filtering at run start, loosely analogous to ADK's last-N-invocation trimming). These are preventive/at-turn-boundary mechanisms rather than Anthropic-style retroactive context editing, so the capability exists only in a limited, weaker form.

### Low-severity (unverified)

- **Compaction triggers less expressive than competitors** — LangChain and DeepAgents support compound trigger clauses (OR-of-AND over tokens/messages/fraction units); ADK offers dual triggers (invocation-count sliding window with overlap plus observed-token threshold with event retention). Dynamiq's SummarizationConfig supports a token threshold and a fraction-of-context ratio but no message-count trigger, no AND/OR composition, and no sliding-window/interval trigger. The essentials are covered, so this is polish rather than a functional hole.

### Dynamiq strengths on this dimension

- Verbatim 'notes' preservation on compaction (ContextManagerTool) directly addresses the lost-identifier failure mode of summarization, plus pinned original input (_pinned_input) that survives repeated compactions — no competitor combines both
- Agent-invocable mid-run compaction (ContextManagerTool, proactively prompted via CONTEXT_MANAGER_INSTRUCTIONS) with automatic threshold-based fallback in the same code path — only DeepAgents matches the agent-triggered pattern; ADK, CrewAI, and LangGraph have no agent-invoked compaction tool
- Automatic dump-to-sandbox of oversized tool outputs with preview-plus-file-path returned to the model, on by default when a sandbox is attached (ToolOutputSandboxPersistenceConfig) — only DeepAgents has a comparable built-in default; ADK, CrewAI, and LangGraph do not
- Per-subagent sandbox views: each subagent gets an isolated working directory inside a shared sandbox session (sandboxes/base.py create_view, shared_session.py), combining context quarantine with controlled artifact sharing — a more granular filesystem-isolation model than DeepAgents' single shared VFS or ADK's branch filtering
- Skills with progressive disclosure built into the OSS agent (dynamiq/skills/ + SkillsTool: name+description indexed in the prompt, full SKILL.md loaded on demand, automatic ingestion into sandbox paths, filesystem and platform registry backends) — parity with DeepAgents' SkillsMiddleware and absent from ADK, CrewAI, LangChain core, and LangGraph
- Chunked summarization with MERGE_SUMMARIES_PROMPT and a separate cheaper summarizer LLM option (SummarizerTool), with orphan tool-message pairing protection at the compaction boundary

### At parity

- Proactive threshold-based auto-summarization preserving recent messages within a token budget — par with LangChain SummarizationMiddleware and DeepAgents core compaction, ahead of CrewAI (reactive only) and LangGraph (none built-in)
- Fraction-of-model-context trigger backed by a shipped model registry with per-model context windows (context_usage_ratio + model_registry.json) — par with LangChain's model-profiles-driven fraction triggers
- Subagent context isolation via fresh-history factories (SubAgentTool) returning a synthesized result to the parent — par with DeepAgents task subagents, ADK AgentTool, and LangGraph subgraph schemas
- Artifact/file passing between steps (BytesIO flows through input/output selectors, shared sandbox paths, FileStore-equipped FileRead/Write/Search/List tools, _requested_output_files) — par with ADK's artifact service and LangGraph state channels for intra-run hand-off
- Node/workflow output caching keyed on entity id + hashed inputs with Redis backend (WorkflowCacheManager) — par with LangGraph CachePolicy (which additionally offers in-memory/SQLite backends)
- Provider prompt-cache usage accounting (cache_read/cache_creation input tokens in BaseLLMUsageData) — par with competitors' usage surfacing
- Memory-side trimming of rehydrated history (memory_limit + retrieval strategies) — par with ADK's GetSessionConfig partial session loading
- Tool-output token truncation with configurable limits (tool_output_max_length) — par with DeepAgents' eviction thresholds and stronger than CrewAI/LangGraph which lack per-tool-output budgeting


## Streaming & realtime (`streaming_realtime`)

### Gaps

#### [HIGH — confirmed] No bidirectional realtime voice/video agent runtime

*Who sets the bar: Google ADK*

Google ADK ships production-depth live agents over the Gemini Live API: mixed audio/video/text input queues, VAD and activity signaling, input/output transcription, interruption handling, speech translation, session resumption, and even TTS-driven voice-agent evaluation. Dynamiq has nothing in this space — its only audio capabilities are batch HTTP Whisper STT and ElevenLabs TTS nodes that do not even use those providers' streaming endpoints, and there is no OpenAI Realtime or Gemini Live integration anywhere. In 2026 voice agents are a major deal category; teams building them cannot use Dynamiq at all, and since none of ADK's realtime stack works with non-Gemini models, a provider-agnostic realtime layer (matching Dynamiq's 28-provider litellm story) would be a genuine wedge rather than a me-too feature.

> **Verification:** Exhaustive greps across dynamiq/, examples/, docs/, and tests/ for realtime terms (realtime, Gemini Live, BidiGenerateContent, google.genai, client.beta.realtime, VAD, turn_detection, server_vad, silero, webrtc, livekit, pipecat, pyaudio, pcm16, session_resumption, barge-in) found no bidirectional realtime audio/video runtime, no OpenAI Realtime or Gemini Live client code, and no VAD or live-session abstraction. The only audio capabilities are three batch HTTP nodes — WhisperSTT (POST /v1/audio/transcriptions), ElevenLabsTTS (POST /v1/text-to-speech/{voice_id}), and ElevenLabsSTS (POST /v1/speech-to-speech/{voice_id}, ElevenLabs' batch voice-conversion, not a conversation loop) — with no use of streaming endpoints ("stream" never appears in dynamiq/nodes/audio/). WebSocket code in callbacks/streaming.py and examples serves text-token streaming only, and pyproject.toml carries no realtime audio dependencies (no google-genai, no audio I/O libraries; its only extras are cua and monty).

#### [HIGH — confirmed] No built-in streaming server or resumable stream delivery

*Who sets the bar: Google ADK, LangGraph, DeepAgents (via langgraph dev)*

ADK bundles an api_server exposing /run_sse and a /run_live websocket, and LangGraph's SDK ships SSE and WebSocket transports with cursor-based resumable streams and join_stream to re-attach to in-flight runs (DeepAgents rides on langgraph dev). Dynamiq offers only copy-paste FastAPI examples — there is no importable server, no CLI serve command, and a client disconnect permanently loses events because no replay/reconnect machinery exists. The practical answer today is 'build it yourself or buy the Dynamiq platform', which is a friction point exactly at the moment a prototype becomes a deployed streaming endpoint.

> **Verification:** The dynamiq package ships no HTTP/SSE/WebSocket server: all FastAPI/WS/SSE code lives under examples/, and fastapi/uvicorn/websockets/sse-starlette appear only in the pyproject `examples` dev dependency group, not core deps or extras. The `dynamiq` CLI has no serve command — its `service` group is a REST client that deploys user-built Docker apps to the hosted Dynamiq platform (/v1/services), matching the 'build it yourself or buy the platform' characterization. Streaming is a destructive in-memory Queue iterator (StreamingQueueCallbackHandler and subclasses) with no sequence numbers on StreamingEventMessage, no replay buffer, no Last-Event-ID support, and no API to attach to an in-flight run's stream; the only resumption machinery is dynamiq/checkpoints/, which resumes workflow execution state, not a reconnecting client's event stream.

#### [MEDIUM — confirmed] Stream consumer API lags 2026 ergonomics: two modes, raw events, no transformers

*Who sets the bar: LangGraph, LangChain 1.x, CrewAI*

LangGraph offers seven composable stream modes with typed discriminated-union parts and a v3 transformer pipeline; LangChain adds typed accumulating projections (.text/.reasoning/.tool_calls) with replay-buffer multi-consumer semantics and middleware that can rewrite streams in flight (e.g. PII redaction of deltas); CrewAI gives channel-multiplexed StreamSessions with .llm()/.tools()/.flow() filters and interleaving. Dynamiq consumers get a single sync/async iterator of raw StreamingEventMessage objects and exactly two agent modes (FINAL, ALL) — filtering, accumulation, and multiplexing are hand-rolled by every application. The typed event data is there; the consumption layer around it is a generation behind.

> **Verification:** Dynamiq's entire streaming consumer surface is exactly as claimed: StreamingQueueCallbackHandler and its sync/async iterator subclasses in dynamiq/callbacks/streaming.py yield raw StreamingEventMessage objects (data typed as Any, no discriminated unions), and StreamingMode in dynamiq/types/streaming.py has only FINAL and ALL. Exhaustive greps across dynamiq/, examples/, docs/, tests/, and pyproject extras for multiplex/StreamSession/replay/tee/broadcast/subscribe/projection/accumulator/middleware/transformer found no channel-multiplexed sessions, no typed accumulating projections, no replay buffer (queues are consume-once; all "replay" hits are checkpoint tool-call replay), and no stream-transform pipeline (InputTransformer is a JSONPath node-input selector; AgentStreamingParserCallback is an internal producer-side parser). The closest primitives — per-node custom event names, source metadata, attaching multiple callback handlers, and subclassing handlers to mutate events — are exactly the hand-rolled building blocks the claim describes, not first-class consumption-layer features.

#### [MEDIUM — confirmed] No client SDK, UI streaming primitives, or published streaming protocol

*Who sets the bar: LangGraph, DeepAgents, LangChain 1.x*

LangGraph streams typed generative-UI components to a React SDK (push_ui_message), DeepAgents streams plans/diffs/tool calls into editors via the Agent Client Protocol, and LangChain is standardizing a cross-package langchain-protocol event vocabulary shared with clients. Dynamiq has no client-side SDK and no published event schema contract — frontend teams must hand-parse StreamingEventMessage JSON from a hand-built websocket, which raises the cost of every chat UI built on Dynamiq OSS and pushes differentiation entirely onto the paid platform.

> **Verification:** Exhaustive searches found zero JS/TS/React files or package.json in the repo, no matches for ACP/Agent Client Protocol/push_ui_message/generative-UI anywhere, and no versioned or published streaming event schema: StreamingEventMessage (dynamiq/types/streaming.py) has no version field, docs/tutorials contain no streaming documentation, and fastapi/websockets/sse-starlette appear only in the examples dev-dependency group of pyproject.toml. The consumption pattern is exactly as claimed: examples hand-build FastAPI websocket/SSE servers that call event.to_json() and clients that hand-parse via StreamingEventMessage.model_validate_json(). The only nuance is that Dynamiq defines typed Pydantic event-data models (StreamingThought, AgentToolInputStart/Delta, AgentToolResultEventMessageData, approval events), an implicit Python-only vocabulary — but that is precisely the "Python model JSON serialization" contract the claim already concedes, so the gap is confirmed.

### Dynamiq strengths on this dimension

- Bidirectional input streaming over the same event bus is a genuine differentiator: StreamingConfig.input_queue lets running nodes block on inbound events, so HumanFeedbackTool and ApprovalConfig do approvals/feedback over the live stream (FeedbackMethod.STREAM) — the only text-agent equivalent among competitors is ADK's BIDI mode, which is coupled to Gemini Live; LangChain/LangGraph/CrewAI route HITL through interrupts/checkpoints outside the stream
- Streaming timeouts are wired into durable execution: InputStreamingTimeoutError triggers an input-timeout checkpoint so a workflow waiting on human input can be resumed later — no competitor ties HITL stream waits to checkpoint/resume this directly
- Tool-input delta streaming is at the leading edge: incremental tool-argument events (AgentToolInputStartData/DeltaData) with stable tool_run_ids, a per-tool stream_tool_input allowlist, fc_wait_for_first_key ordering control, and JSONInnerThoughtsExtractor splitting 'thought' fields out of streamed JSON character-by-character — comparable to ADK's progressive SSE (feature-flagged) and LangGraph's ToolCallStream, ahead of CrewAI and DeepAgents
- Event coalescing (min_chunk_chars) to reduce event counts is a practical knob for high-fanout parallel-tool streams that most competitors lack
- Uniform per-node StreamingConfig means any node in a workflow DAG can stream with the same typed envelope, and token streaming works identically across all 28 litellm providers — no provider-coupled streaming tiers

### At parity

- LLM token streaming with sync and async iterator consumption — at par with all five competitors
- Typed intermediate event model covering agent reasoning, tool-call lifecycle (start/delta/error/result), and final answers — roughly at par with CrewAI's event bus, ADK's Event model, and LangGraph's typed stream parts
- Reasoning/thought streaming as distinct structured data (StreamingThought, inner-thoughts extraction) — comparable to CrewAI's LLMThinkingChunkEvent and LangChain's .reasoning projection
- Custom progress events from arbitrary nodes via run_on_node_execute_stream — conceptually equivalent to LangGraph's StreamWriter and CrewAI's scoped event sinks, though less documented as a user-facing API
- FINAL vs ALL streaming granularity selection — equivalent in spirit to competitors' final-vs-intermediate controls, though with fewer composable modes


## Human-in-the-loop & durability (`hitl_durability`)

### Gaps

#### [HIGH — confirmed] HITL waits block the executing thread; no suspend-and-return interrupt primitive

*Who sets the bar: langgraph, langchain, crewai, adk*

LangGraph's interrupt()/Command(resume), CrewAI's HumanFeedbackPending, and ADK's long-running function calls all immediately suspend the run, return control to the caller with a typed pending-approval payload, and accept the human's decision as an argument to the resume call — the natural shape for web servers where approvals take hours or days. Dynamiq instead blocks the worker thread on console input or a streaming queue until the human answers or StreamingConfig.timeout (default 600s) elapses, at which point it converts to a resumable checkpoint; on resume the prompt is re-issued through the same blocking transport rather than accepting the answer with the resume call. This makes long-lived approvals workable but operationally awkward (held threads, timeout tuning, re-prompt round-trip) compared to competitors' first-class pause semantics.

> **Verification:** Dynamiq v0.59.0 has no immediate-suspend/resume-with-decision API. Approval and feedback waits block the worker thread (input() polling loop in send_console_approval_message; queue.get polling loop in get_input_streaming_event/input_method_streaming) until the answer arrives or StreamingConfig.timeout (default 600.0s) fires, at which point on_input_timeout saves a PENDING_INPUT checkpoint and the run returns RunnableStatus.FAILURE — RunnableStatus has no pending/paused value, so control is never returned with a first-class pending-approval result. On resume, _restore_from_checkpoint clears pending_inputs and logs that nodes "will re-request approval on resume"; run_sync/run_async/Workflow.run accept only resume_from with no decision argument, and integration tests show the sanctioned pattern is pre-loading the answer into the streaming input_queue so the re-issued blocking prompt consumes it. The only softening nuances — BaseCheckpointState.approval_response (skips re-prompt only when the answer was already received before a crash) and FlowCheckpoint.pending_inputs/get_pending_inputs() (typed pending payload available only after the blocking timeout) — are already acknowledged within the claim, so they do not refute it.

#### [MEDIUM — confirmed] No conditional/dynamic approval policies and no human-substitutes-output decision

*Who sets the bar: langchain, adk, deepagents*

LangChain's HumanInTheLoopMiddleware supports per-tool `when` predicates, callable per-call descriptions, and a RespondDecision where the human's answer replaces the tool's output; ADK's require_confirmation accepts a per-call callable; DeepAgents compiles glob-scoped filesystem permission rules into interrupt predicates automatically. Dynamiq's ApprovalConfig is a static enabled bool with a Jinja message template — every call to a gated node interrupts, with no way to gate only risky argument values, and the human can approve, edit whitelisted inputs, or cancel-with-feedback but cannot supply the tool's result directly. Teams building selective approval policies (approve only writes outside a sandbox, only payments above a threshold) must hand-roll the logic.

> **Verification:** ApprovalConfig in dynamiq/types/feedback.py contains only enabled (bool), feedback_method, mutable_data_params, msg_template (Jinja), event, and accept_pattern (a plain string equality-checked against feedback text) — no callable/predicate to decide per-invocation based on tool-call arguments, and per-run NodeRunnableConfig overrides support only streaming, not approval. The gate in Node.get_approved_data_or_origin (dynamiq/nodes/node.py:954) is a static `if self.approval.enabled`, so every call to a gated node interrupts; ApprovalInputData carries only feedback/data/is_approved, and on approval only whitelisted mutable_data_params edits are merged into inputs while rejection raises NodeSkippedException with output=None — there is no decision that lets the human supply the node's output (Agent._run_tool in dynamiq/nodes/agents/base.py surfaces a rejection as a tool error string, and the separate HumanFeedbackTool returns human text only as its own output, not as a substitute for a gated tool's result). Repo-wide searches of dynamiq/, examples/, docs/, tests/, and git history for interrupt/predicate/require_confirmation/when/respond-style mechanisms found nothing further, so the claim is accurate.

#### [MEDIUM — confirmed] No scheduled/cron, trigger, or background-run surface in OSS

*Who sets the bar: adk, deepagents, langgraph (platform client), crewai (platform client)*

All frameworks push production scheduling to paid platforms, but competitors are starting to ship OSS entry points: ADK has in-repo Pub/Sub and Eventarc trigger endpoints with backoff and concurrency limits, DeepAgents ships an experimental persistent cron scheduler with agent-facing cron tools, and LangGraph/CrewAI at least ship OSS clients for platform crons/triggers. Dynamiq OSS has zero scheduling, trigger, webhook, or job-queue surface — the only entry points are in-process run_sync/run_async calls, and the CLI's org/project/service commands exist solely to deploy to the paid Dynamiq platform. Users who want a recurring or event-triggered agent must bring their own scheduler and serving layer entirely.

> **Verification:** Exhaustive greps for cron/schedule/webhook/fastapi/uvicorn/celery/apscheduler/queue/trigger synonyms across dynamiq/, examples/, docs/, and tests/ found no scheduler, webhook, HTTP trigger, or job-queue implementation: the only 'schedul' hits are a docstring word in checkpoints/checkpoint.py and a prompt example string in nodes/tools/long_term_memory.py, and the only queues are in-process streaming Queues (callbacks/streaming.py) and thread/process pools (executors/pool.py). FastAPI/uvicorn appear only in the pyproject 'examples' dependency group and in examples/ scripts where users build their own servers around run_sync/run_async, confirming bring-your-own serving. The CLI (commands/utils.py registers only org/project/service/resource-profiles/config) exclusively calls the hosted platform REST API at https://api.getdynamiq.ai, including service deploy which uploads a tarball to /v1/services/{id}/deploy — no local run, serve, or schedule command exists.

#### [MEDIUM — confirmed] Retry/timeout policies lack exception filtering, jitter, and idle-timeout semantics

*Who sets the bar: langgraph, adk, langchain*

LangGraph RetryPolicy, ADK RetryConfig, and LangChain's retry middleware all support retry_on exception-type filters and jitter; LangGraph additionally has idle_timeout with heartbeat refresh so long-running nodes are killed on stall rather than on a wall-clock budget. Dynamiq's ErrorHandling retries every exception uniformly with deterministic backoff — so non-transient failures (auth errors, validation errors, 4xx) burn the full retry budget and can duplicate side effects — and only hard wall-clock timeouts exist. For production agents this means wasted spend on unretryable errors and coarse timeout tuning for legitimately long tasks.

> **Verification:** The claim is accurate: ErrorHandling (dynamiq/nodes/node.py:85-100) has only timeout_seconds, retry_interval_seconds, max_retries, backoff_rate, and behavior — no exception-type filter and no jitter — and execute_with_retry (node.py:1327-1415) catches bare Exception and retries every error type (including TimeoutError, auth/validation errors) uniformly with deterministic backoff, exempting only CanceledException for cancellation. Case-insensitive greps for idle_timeout, heartbeat, watchdog, keepalive, liveness, and stall across the entire repo (source, tests, docs, examples) return nothing; the only timeout is a hard wall-clock one via execute_with_timeout. Exception-filtered retry with jitter does exist in Dynamiq but only hardcoded to internal rate-limit paths (SandboxCreationErrorHandling in code_interpreter.py, tenacity in sandboxes/daytona.py and sandboxes/e2b.py, and the CLI HTTP client), and the agent's RecoverableAgentException hierarchy only feeds errors back to the LLM loop — none of these provide a user-configurable retry_on/jitter policy for node execution, so the claimed gap stands.

### Low-severity (unverified)

- **Map fan-out has no partial-progress durability** — LangGraph persists successful parallel branches' writes (pending_writes) so only failed tasks re-run after a crash, and its @task results are replayed on resume. Dynamiq's flow-level checkpointing skips completed nodes and its agents persist per-iteration state, but the Map operator is a plain Node with no IterativeCheckpointMixin — a crash 900 items into a 1000-item fan-out re-executes all items on resume. Matters for large batch/ETL-style agentic workloads.
- **Single fallback model rather than an ordered fallback chain** — LangChain's ModelFallbackMiddleware and core with_fallbacks accept an ordered list of fallback models tried in sequence. Dynamiq's FallbackConfig accepts exactly one fallback LLM, so multi-provider degradation ladders (primary -> cheaper same-provider -> different provider) require nesting or custom code. Dynamiq's trigger conditions (RATE_LIMIT/CONNECTION/ANY) are a nice touch but the chain depth is 1.

### Dynamiq strengths on this dimension

- Approval gates on any node type, not just tool calls: ApprovalConfig lives on the base Node, so LLM calls, retrievers, converters, and whole sub-workflows can be human-gated, and PlanApprovalConfig gates a manager-orchestrator's task plan before execution — competitors' approval surfaces are tool-call-centric
- Edit-and-resume with a security boundary: mutable_data_params whitelists exactly which input fields the human may modify, protecting the rest of the payload — LangChain's EditDecision has no equivalent field-level allow-list
- Browser-takeover HITL (HumanFeedbackTool.is_browser_takeover): the human takes over the agent's live Stagehand browser session mid-run and hands it back — no competitor in this set ships anything comparable
- Automatic unanswered-approval-to-durable-run conversion: checkpoint_on_input_timeout_enabled turns an approval left overnight into a PENDING_INPUT checkpoint that resumes and re-prompts, with pending approval responses persisted in node checkpoint state (to_checkpoint_state) so already-received answers are not re-asked after a crash
- Finer-grained agent-loop durability than LangGraph's node semantics: per-iteration agent state including the pending tool call captured before execution (nodes/agents/checkpoint.py, agent.py replay path) means resume continues mid-loop without repeating completed iterations, whereas LangGraph re-executes an interrupted node from its start (side effects repeat unless wrapped in @task)
- Cooperative cancellation as a first-class durability feature: CancellationToken is checked throughout agents, tools, and HITL waits, and checkpoint_on_cancel preserves resumable state at the cancel point — a stop button that yields a resumable run, not a killed process
- Checkpoint-on-failure by default gives every crashed run a resume point plus forensics with zero extra configuration (checkpoint_on_failure_enabled=True)

### At parity

- Cross-restart durable HITL: persisted pending-input contexts on PostgreSQL/Filesystem backends match LangGraph checkpointer interrupts, CrewAI's SQLite pending-feedback tables, and ADK's event-sourced confirmations in outcome (approvals survive restarts), differing only in resume ergonomics (see blocking gap)
- Core durable-execution semantics: checkpoint-after-every-node with completed-node skip and output replay is at par with ADK workflow checkpoint/replay and CrewAI Crew.from_checkpoint; APPEND snapshot chains with parent_checkpoint_id plus resume_from any checkpoint and exclude_node_ids approximate LangGraph's time-travel (less ergonomic than update_state but present)
- Baseline retry/backoff/timeout per node (ErrorHandling) matches the basic tier of every competitor, minus the filtering/jitter sophistication noted in gaps
- Agent self-correction on errors: parse-failure recovery-as-observation and tool errors fed back into the loop are equivalent to LangChain's ToolErrorMiddleware and ADK's ReflectAndRetryToolPlugin
- Ask-the-user tooling: HumanFeedbackTool ask/info with pluggable input/output transports is at par with CrewAI's Flow.ask/InputProvider and LangChain's respond-decision pattern
- LLM failure fallbacks with typed triggers (RATE_LIMIT/CONNECTION/ANY) are at par with LangChain model fallback and ahead of raw LangGraph, modulo the single-model chain-depth gap
- Node-level caching to avoid re-execution (caching config + cache_wf_entity) parallels LangGraph's CachePolicy
- The OSS/commercial split itself: like LangGraph (Platform), CrewAI (AMP), and LangChain, Dynamiq delegates serving, scheduling, and managed background runs to its paid platform — the split is industry-standard even though ADK/DeepAgents are starting to erode it in OSS


## RAG & knowledge (`rag_knowledge`)

### Gaps

#### [HIGH — confirmed] No structured citation / source-attribution model

*Who sets the bar: langchain, adk*

LangChain ships a standardized Citation content block (cited_text, start/end offsets) normalized across provider-native citation formats, and ADK propagates Gemini grounding_metadata (citations, supports) into every response. Dynamiq's citations are prompt-instruction-level only: the ReAct agent prompt asks for inline Markdown reference-style citations, plus a raw passthrough of Perplexity's citations field. For enterprise RAG in 2026, verifiable source attribution is close to table stakes, and free-text Markdown citations cannot be validated, rendered, or audited programmatically.

> **Verification:** Exhaustive greps across dynamiq/, examples/, docs/, and tests/ for citation, cited_text, url_citation, grounding, annotation, attribution, offsets, and related class names found no structured citation type and no code that parses or validates citations from agent/LLM output; the only handling is the ReAct prompt text asking for Markdown reference-style citations and perplexity.py's raw `result["citations"] = response.citations` passthrough, exactly as claimed. Other provider LLM nodes (gemini, vertexai, anthropic, openai) drop grounding/annotation metadata entirely, agent output parsers extract only thought/action/answer tags, and the Document type has no citation fields. The closest adjacent capabilities — knowledge-graph edge provenance (source_doc_id, source_documents grounding fetch) and evaluation metrics that LLM-judge sentence attribution (context_recall, factual_correctness) — are retrieval provenance and post-hoc scoring, not structured, verifiable citation output.

#### [MEDIUM — confirmed] No incremental indexing / ingestion-sync API

*Who sets the bar: langchain*

LangChain's indexing API (index()/aindex() with RecordManager) does content-hash dedup, incremental re-sync of changed source documents, and cleanup modes (incremental/full/scoped_full) that delete stale chunks when sources change or disappear. Dynamiq's ingestion offers only a write-time DuplicatePolicy (none/skip/overwrite/fail) and dry-run-with-cleanup; re-running an ingestion pipeline after documents change has no mechanism to detect unchanged content or garbage-collect chunks of deleted sources. This makes keeping a production index in sync with a living corpus a manual problem.

> **Verification:** Exhaustive greps (RecordManager, content_hash, hashlib/sha256/md5, incremental/scoped_full/stale/prune/orphan/resync/reindex, cleanup, sync/Indexer class names) across dynamiq/, examples/, docs/, tests/, pyproject, and git history found no record-manager-style indexing API: content_hash exists only in dynamiq/memory/long_term (md5 fact-dedup for conversational memory), and write-time dedup is the ID-based DuplicatePolicy enum, with dry-run cleanup only deleting artifacts created during the dry run itself. The only cleanup primitives are manual caller-driven delete_documents_by_file_id(s)/delete_documents_by_filters, and dynamiq/storages/graph/base.py explicitly documents the design as "writing never deletes on its own, so to replace a document's facts the caller deletes them here, then re-writes". Two adjacent-but-weaker mechanisms exist that do not refute the claim: splitters' IdStrategy.DETERMINISTIC makes chunk IDs content-addressed (sha256 of parent_id:index:content), giving limited idempotency with SKIP but only if the caller supplies stable document IDs (converters default to uuid4 per run), with no embedding skip and no stale-chunk deletion; and the knowledge-graph writer's deterministic entity IDs make graph re-ingestion idempotent, but that targets graph stores, not vector chunks, and also never garbage-collects on its own.

#### [MEDIUM — confirmed] No local/offline embedding providers; narrower embedder catalog

*Who sets the bar: crewai, langchain*

All 8 Dynamiq embedders (OpenAI, Cohere, Bedrock, Gemini, VertexAI, Mistral, HuggingFace, WatsonX) are hosted-API calls via litellm — even the HuggingFace embedder targets the HF Inference API endpoint. CrewAI ships 18 embedding providers including local options (ollama, onnx, sentence-transformers, openclip) and newer API vendors (voyageai, jina); LangChain covers Ollama and local HuggingFace/sentence-transformers embeddings. Air-gapped, privacy-sensitive, and cost-conscious self-hosted deployments — a natural audience for a framework with 8 self-hosted vector stores — cannot embed locally with Dynamiq.

> **Verification:** Both embedder directories contain exactly the 8 hosted providers named in the claim (bedrock, cohere, gemini, huggingface, mistral, openai, vertexai, watsonx) and nothing else; components/embedders/base.py routes every call through litellm.embedding/aembedding with connection API params, never loading a model in-process, and huggingface.py hardcodes API_BASE_URL = "https://api-inference.huggingface.co/models". Repo-wide greps for ollama/sentence-transformers/fastembed/onnx/openclip found Ollama only as an LLM node plus connection (dynamiq/nodes/llms/ollama.py), a single comment link to qdrant-client's fastembed source in qdrant.py (the SparseEmbedding there is a plain dataclass — Dynamiq never computes sparse vectors itself), and pyproject.toml has no torch/transformers/fastembed/onnxruntime dependency or extra. The only nuance is that the OpenAI connection accepts a custom url (OPENAI_URL passed as api_base), so a user could manually point OpenAITextEmbedder at a local OpenAI-compatible server such as Ollama's /v1 — but that is an undocumented generic override requiring an external server, not a shipped local/in-process embedder, so the claim as stated is confirmed.

#### [MEDIUM — partial] No advanced retrieval-composition strategies (MMR, ensemble/RRF, multi-query, parent-document, self-query)

*Who sets the bar: langchain*

LangChain (core + classic) provides MMR on every vector store, EnsembleRetriever with reciprocal-rank fusion across heterogeneous retrievers, MultiQueryRetriever, ParentDocumentRetriever/MultiVectorRetriever (retrieve small chunks, return parent context), SelfQueryRetriever (LLM-generated metadata filters), and a contextual-compression pipeline. Dynamiq has none of these composition layers — its retrieval is single-store top-k plus per-store hybrid fusion and post-hoc rerankers (Cohere/LLM/recency), which only partially compensates. Teams tuning retrieval quality hit this ceiling quickly.

> **Verification:** The core gap thesis largely holds: no MMR anywhere (no mmr/max_marginal/lambda_mult/fetch_k/diversity code; rankers are only Cohere/LLM/recency), no generic EnsembleRetriever with RRF across heterogeneous retrievers (RRF appears only inside single stores for dense+sparse hybrid in Qdrant/Milvus/pgvector), no LLM MultiQueryRetriever, and no SelfQueryRetriever (retriever tools merely expose an agent-fillable filters param). But the absolute phrasing "implements none of the following anywhere" and "greps return no framework code" is contradicted in two places: dynamiq/components/splitters/base.py implements explicit "ParentDocumentRetriever-style parent chunking" (parent_chunk_size, PARENT_DOC_KEY="parent_chunk_id", _attach_parent_chunks builds larger parents and links children) — the index-time half of the parent-document feature, though no retriever ever reads parent_chunk_id at query time; and DynamiqKnowledgebaseHybridSearch (dynamiq/nodes/knowledgebases/knowledgebase_hybrid.py) genuinely combines two retriever nodes (vector + graph search) with concurrent execution, merge/dedupe, and rerank — a cross-retriever composition layer, albeit restricted to the Dynamiq-hosted knowledgebase and using rerank rather than rank fusion.

#### [MEDIUM — partial] OSS knowledge-base abstraction is cloud-gated; no agent-level knowledge attachment

*Who sets the bar: crewai, langgraph*

CrewAI lets users attach knowledge sources directly to an agent or crew (knowledge_sources=[PDFKnowledgeSource(...)]) with automatic LLM query rewriting before retrieval, all running locally in OSS; LangGraph's semantically-indexed BaseStore is OSS for Postgres/SQLite. Dynamiq's equivalent convenience layer — the knowledgebase nodes (vector/graph/hybrid search) — exists only as a client for the paid Dynamiq cloud API (POST /v1/knowledgebases/{id}/vector-search). OSS users must hand-assemble embedder+retriever pipelines and wire them as tools, a DX disadvantage that also signals 'the good parts are paid'.

> **Verification:** The claim's specific assertions are all accurate: every node in dynamiq/nodes/knowledgebases/ is a client for the hosted Dynamiq API (vector/graph POST to {connection.url}/v1/knowledgebases/{id}/vector-search|graph-search via the Dynamiq connection defaulting to https://api.getdynamiq.ai; hybrid just composes the two cloud nodes), and the Agent class in base.py has no knowledge/knowledge_sources parameter, no automatic query rewriting (memory RELEVANT/HYBRID strategies pass the raw user_query), and integrates retrieval only via memory strategies and retriever nodes wired as generic tools. However, the headline "no locally-runnable knowledge-base abstraction" overreaches: dynamiq/nodes/knowledge_graphs/ is a fully local OSS knowledge-base stack (entity extractor, writer, retriever) over self-hosted Neo4j/Apache AGE/AWS Neptune, and its KnowledgeGraphRetriever even runs an LLM query-entity-extraction pre-step before retrieval plus optional reranking and summarization; VectorStoreRetriever is likewise a pre-built embedder+retriever+reranker composition tool, so users compose sub-nodes rather than raw pipelines. The CrewAI-style agent-level convenience (knowledge_sources with automatic LLM query rewriting) is genuinely absent, but a weaker local knowledge-base layer does exist.

### Low-severity (unverified)

- **No web/SaaS/data-source knowledge loaders** — CrewAI-tools ships 16 RAG loaders (webpage, docs-site, GitHub, YouTube channel/video, MySQL, Postgres, etc.) plus Docling-based universal document parsing, and LangChain's community catalog has hundreds of source loaders. Dynamiq's converters cover file formats well (PDF/DOCX/PPTX/Excel/HTML/CSV, Unstructured API, vision-LLM extraction, Mistral OCR) but there is no converter that ingests a URL, repo, video transcript, or SQL table into Documents — web tools (Firecrawl, Tavily, Jina, ZenRows) exist as agent tools, not ingestion sources, so wiring web content into a RAG index takes custom glue.

### Dynamiq strengths on this dimension

- Broadest in-repo, self-hostable vector-store catalog of the six frameworks: 8 stores (Chroma, Elasticsearch, Milvus, OpenSearch, PGVector, Pinecone, Qdrant, Weaviate), each with matching writer and retriever nodes — vs 2 in the LangChain monorepo (Chroma, Qdrant), 2 core clients in CrewAI, 3 memory-store backends in LangGraph, and 0 in ADK and DeepAgents.
- Only framework with a complete OSS knowledge-graph RAG loop: LLM entity/ontology/triple extraction (KnowledgeGraphEntityExtractor), graph writer and retriever, 3 graph stores (Neo4j, Apache AGE, Neptune), and a CypherExecutor tool — LangChain's graph RAG is legacy-classic proxies only, and CrewAI/LangGraph/ADK/DeepAgents have none.
- Deepest chunking stack: 11 splitter strategies including Anthropic-style contextual retrieval enrichment (ContextualSplitterComponent), embedding-based semantic splitting, and AutoSplitter rule-based strategy selection — none of which exist in langchain-text-splitters or any other competitor.
- In-core rerankers (CohereReranker, LLM-as-reranker, recency ranker); CrewAI, LangGraph, ADK, and DeepAgents ship no reranker in core, and LangChain's reranking is an abstract compressor hook plus external integrations.
- Rich document-extraction options in core: vision-LLM text extraction (LLMTextExtractor), Mistral OCR parser node, Unstructured.io API converter, and per-format converters with auto-dispatch (MultiFileTypeConverter).
- Hybrid dense+sparse retrieval shipped in-repo: Qdrant sparse embeddings with IDF (BM42-style) plus hybrid search paths in pgvector, Milvus, and Weaviate stores.
- Dry-run ingestion (DryRunConfig + DryRunMixin) that tests a pipeline then auto-deletes documents/collections — an operational capability no competitor has.
- End-to-end RAG pipelines are declarable as YAML DAGs per vector store (examples/components/rag/vector_stores/dag/), fitting the low-code orchestration story.

### At parity

- Agentic RAG via retrieval-as-a-tool: the store-agnostic VectorStoreRetriever node passed to agents matches the modern LangChain v1 and ADK 'retriever is just a tool' pattern.
- Hosted embedder API coverage: 8 major providers (OpenAI, Cohere, Bedrock, Gemini, VertexAI, Mistral, HuggingFace API, WatsonX) is roughly on par with LangChain's init_embeddings 10-provider factory for cloud APIs (the gap is local models, tracked separately).
- Web-search-as-retrieval: Exa, Tavily, Firecrawl, Jina, ScaleSerp, and ZenRows tool nodes are comparable to LangChain's Exa partner retriever and ADK's Google-grounded search tools.
- Local development story: Milvus FILE deployment (Milvus Lite, local .db) gives a zero-server dev/test vector store, comparable in practice to LangChain's InMemoryVectorStore and LangGraph's InMemoryStore, though Dynamiq has no pure in-memory reference store.
- Write-time duplicate handling (DuplicatePolicy: skip/overwrite/fail) is comparable to typical per-write dedup in competitor stores, short of LangChain's record-manager incremental indexing (tracked as a gap).
- Office/file-format document conversion breadth (PDF, DOCX, PPTX, Excel, CSV, HTML, text) is roughly at par with CrewAI's knowledge sources for file-based corpora.


## Model support & multimodal (`models_multimodal`)

### Gaps

#### [HIGH — confirmed] No native provider SDK path - everything rides LiteLLM

*Who sets the bar: crewai, adk, langchain*

CrewAI re-architected to six native SDK providers (Anthropic, OpenAI incl. Responses, Azure, Bedrock, Gemini, Snowflake) with litellm demoted to optional fallback; ADK has native google-genai and Anthropic integrations; LangChain uses first-party integration packages per provider. All 27 Dynamiq provider classes are thin LiteLLM prefix wrappers, so feature velocity is capped by LiteLLM and provider betas require workarounds - the monkey-patch in anthropic.py to forward strict:true is a symptom. Users lose day-one access to provider features (new APIs, betas, SDK-level interceptors) that native-path frameworks pick up immediately.

> **Verification:** Exhaustive grep of dynamiq/ found zero native SDK completion calls (no chat.completions, messages.create, generate_content, invoke_model, converse, or responses.create anywhere) and no imports of anthropic or google-genai at all; the only SDK imports are an openai client built in connections.py that is handed to litellm.completion as its `client` param, and boto3 used solely for DynamoDB memory and OpenSearch auth, never bedrock-runtime. BaseLLM binds litellm.completion/acompletion in __init__ (base.py:336) and execute/execute_async call only those (base.py:1021, 1094); no subclass in dynamiq/nodes/llms/ overrides the dispatch — all 28 provider classes (claim says 27, immaterial) are MODEL_PREFIX/param-tweak wrappers. The claim's cited symptom is real: anthropic.py:12-68 monkey-patches LiteLLM's AnthropicConfig._map_tool_helper to forward strict:true, and pyproject.toml carries no anthropic/google-genai dependency or native-SDK extras.

#### [HIGH — confirmed] Reasoning/thinking output is dropped, not surfaced or preserved

*Who sets the bar: crewai, langchain, adk, deepagents*

Dynamiq can enable Anthropic extended thinking (thinking_enabled + budget_tokens) but the response handler returns only content and tool_calls - reasoning_content is never extracted, there are no typed reasoning blocks or thinking-chunk stream events, and thinking-block signatures are not preserved across agent tool-use turns (which Anthropic requires for thinking+tools). CrewAI extracts thinking blocks with signature preservation and emits LLMThinkingChunkEvent; LangChain has standardized ReasoningContentBlock plus adaptive thinking and redacted-thinking handling; ADK streams and re-emits Anthropic thinking. In 2026 with reasoning models dominant, paying for thinking tokens you cannot see or replay is a serious deficit.

> **Verification:** All three literal assertions verify: 'reasoning_content' has zero occurrences anywhere in the repo; _handle_completion_response (dynamiq/nodes/llms/base.py:581-618, content read at line 597) returns only 'content' and 'tool_calls'; and the agent tool-use loop's _append_assistant_message (dynamiq/nodes/agents/agent.py:826-894) replays assistant turns as Message(content, tool_calls) where the Message model has no thinking/signature field, so Anthropic thinking blocks and signatures cannot be preserved across tool-use turns despite thinking_enabled/budget_tokens being settable (base.py:248, 885-886). The only mitigations found are incidental: raw litellm stream chunks passed verbatim through callbacks expose untyped 'thinking_blocks' deltas (demonstrated in examples/components/llm/llms/thinking_streaming.py, the sole thinking_blocks reference in the repo), and registry flags supports_reasoning/supports_adaptive_thinking exist but are dormant metadata consumed by no behavior code — neither constitutes typed reasoning output, thinking-chunk events, or replay preservation.

#### [HIGH — confirmed] No OpenAI Responses API support

*Who sets the bar: crewai, langchain, deepagents, adk*

CrewAI's native OpenAICompletion supports the Responses API (text.format structured outputs, parse_tool_outputs), LangChain's ChatOpenAI supports it including the image_generation built-in tool with streaming partial images, DeepAgents defaults OpenAI to Responses via ProviderProfile, and ADK ships OpenAIResponsesLlm. Dynamiq only uses litellm chat completions, so Responses-only capabilities - encrypted/stateful reasoning items across tool calls, server-side built-in tools (web search, code interpreter, image generation) - are unreachable, precisely where OpenAI's reasoning-model roadmap is heading.

> **Verification:** Dynamiq's entire LLM layer binds only litellm.completion/acompletion (dynamiq/nodes/llms/base.py line 336) and every provider node, including OpenAI, merely rewrites Chat Completions parameters (max_completion_tokens, reasoning_effort, verbosity); repo-wide greps across source, examples, docs, and tests for litellm.responses, aresponses, responses_api, /v1/responses, previous_response_id, encrypted_content, web_search_preview, and text.format return zero hits. Adjacent capabilities exist only in non-Responses form: image generation via the Images API endpoint (litellm.image_generation), web search via third-party tool nodes (Tavily/Exa/Firecrawl/ScaleSERP), and code interpreter via a client-orchestrated cloud sandbox — none are OpenAI server-side built-in tools. The sole caveat is that litellm itself may internally bridge Responses-only models to the Responses endpoint, but that is litellm transport, not a Dynamiq code path, and exposes none of the Responses-only capabilities the claim identifies as missing.

#### [MEDIUM — confirmed] No audio input in chat messages (speech understanding)

*Who sets the bar: crewai, langchain, adk*

CrewAI's crewai-files routes audio files (mp3/wav/...) natively into provider requests, LangChain has AudioContentBlock, and ADK maps audio content through both Gemini and LiteLLM paths. Dynamiq's VisionMessage supports only text, image_url, and file (PDF/video) content types - there is no input_audio content block, so users must bolt on the separate WhisperSTT node and lose native audio understanding on Gemini/GPT-4o-audio class models.

> **Verification:** The claim is accurate: 'input_audio' appears nowhere in the repository (source, tests, examples, docs, or dependency metadata), and VisionMessage at dynamiq/prompts/prompts.py:217 accepts exactly VisionMessageTextContent | VisionMessageImageContent | VisionMessageFileContent, with the VisionMessageType enum limited to text/image_url/file and ValueError raised for anything else. Audio support exists only as detached WhisperSTT (dynamiq/nodes/audio/whisper.py) and ElevenLabs TTS (dynamiq/nodes/audio/elevenlabs.py) nodes, and agent file routing (dynamiq/nodes/agents/utils.py, base.py) detects only images and videos — audio files are never mapped into LLM message content. The sole caveat is that parse_bytes_to_base64 could incidentally embed audio bytes as a data URI inside a generic 'file' block, but this is untyped, undocumented (VisionMessageFileData's docstring covers only video/PDF), and would not produce the input_audio format required by GPT-4o-audio-class models, so it is not a real capability.

#### [MEDIUM — confirmed] No realtime bidirectional voice/video (Live API-style) sessions

*Who sets the bar: adk*

ADK wires the Gemini Live API end-to-end: bidi websocket streaming of realtime audio/video blobs, input/output audio transcription, session resumption, and a /run_live endpoint. Dynamiq has batch TTS/STT nodes but no live model connection, so real-time voice agents - a major 2026 product category - cannot be built on Dynamiq without going outside the framework.

> **Verification:** Exhaustive search of dynamiq/, examples/, docs/, and pyproject.toml found no websocket-based live model connection, no realtime audio session API, and no LiveRequestQueue/BIDI equivalent: the LLM layer (dynamiq/nodes/llms/base.py) binds only litellm completion/acompletion with token streaming, audio support is limited to batch HTTP TTS/STT/STS nodes (whisper.py, elevenlabs.py), and Workflow exposes only run_sync/run_async. The only bidirectional-ish constructs are user-authored websocket example servers relaying text streaming events and StreamingConfig.input_queue, which is consumed solely by HumanFeedbackTool for mid-run human text input — not a live model session. The 'gemini-3.1-flash-live-preview' entry in model_registry.json is mere capability metadata (supports_video_input) for batch completion, not a Live API integration, so the claimed gap is confirmed.

#### [MEDIUM — confirmed] Prompt caching is Anthropic-only

*Who sets the bar: adk, deepagents, crewai*

ADK has Gemini context caching with a cache manager and hit-rate analyzer; DeepAgents ships Bedrock and Fireworks prompt-caching middleware alongside Anthropic; CrewAI passes Azure prompt_cache_key. Dynamiq's AnthropicCacheControl (TTL, injection points, cache token accounting) is solid but exists only on the Anthropic node - users on Gemini, Bedrock-hosted Claude, or Azure get no explicit caching controls and pay full input-token cost on long agent contexts.

> **Verification:** Exhaustive greps across dynamiq/, examples/, docs/, and tests/ confirm the claim: AnthropicCacheControl in dynamiq/nodes/llms/anthropic.py (ephemeral type, 5m/1h TTL, cache_injection_point_index wired to litellm's cache_control_injection_points) is the only explicit prompt/context caching configuration; "cachePoint", "prompt_cache_key", "cached_content"/"CachedContent"/"context_cache" have zero hits anywhere, and the Gemini, Bedrock, AzureAI, VertexAI, and OpenAI nodes are thin BaseLLM wrappers with no caching fields (Bedrock inherits BaseLLM, not Anthropic, so Bedrock-hosted Claude gets no cache_control). Other cache hits are unrelated capabilities: dynamiq/cache/ is Redis node-output caching, agents' _tool_cache caches tool results, and base.py/model_registry.json only do provider-generic cache-token usage/cost accounting. The sole mitigating nuance is that BaseLLM's extra="allow" config forwards arbitrary untyped kwargs to litellm (base.py lines 881-888), so a user could manually smuggle prompt_cache_key through, but this is undocumented and untested for caching, not an explicit control.

### Low-severity (unverified)

- **No per-model profile/harness tuning registry** — DeepAgents' HarnessProfile registry applies per-model behavioral tuning (prompt overrides, tool exclusions, middleware) automatically by model id, and LangChain's model.profile (models.dev-derived) lets generic code make capability-aware decisions. Dynamiq's closest analogs are the _SAMPLING_UNSUPPORTED_* param-stripping tables and the model_registry.json metadata overlay - useful, but limited to parameter compatibility and capability flags, with no mechanism to tune prompts/tools per model.
- **No LLM response cache or client-side model rate limiting** — LangChain's BaseChatModel supports pluggable response caches and InMemoryRateLimiter; CrewAI has an LLM response cache (llms/cache.py) and an RPM controller. Dynamiq has retries/timeouts and fallback triggers but no cache-by-prompt layer to avoid repeated identical calls and no RPM/TPM throttle, which matters for cost control and rate-limit hygiene in high-fan-out multi-agent workflows.

### Dynamiq strengths on this dimension

- Only framework of the six with multimodal generation in core: ImageGeneration/ImageEdit/ImageVariation nodes across OpenAI, Gemini, VertexAI, Bedrock, AzureAI, and OpenRouter, plus ElevenLabs TTS and speech-to-speech and Whisper-compatible STT nodes (competitors have at most tool-level DALL-E or nothing)
- Declarative LLM fallback (FallbackConfig with typed ANY/RATE_LIMIT/CONNECTION triggers) whose state survives checkpoints (LLMCheckpointState) - more operationalized failover than any competitor's with_fallbacks-style approach
- Strict tool-calling depth unusual for a LiteLLM wrapper: strict_tools as bool or per-tool-name list, automatic schema down-conversion to the strict subset, and an upstream-bug workaround forwarding strict:true to Anthropic
- Video understanding with per-file video_metadata (fps, start/end offsets) forwarded to Gemini - only DeepAgents (frame extraction) and LangChain (VideoContentBlock) have comparable video-input stories
- Built-in per-call USD cost tracking (litellm cost_per_token) surfaced in node usage data including cache read/creation tokens - cost observability competitors leave to external tooling
- model_registry.json overlay plus per-instance ModelInfo overrides compensating for LiteLLM metadata gaps (e.g. supports_video_input), with automatic completion-param recovery that strips unsupported sampling params per model family (max 3 recoveries)

### At parity

- Provider breadth: 27 named providers + CustomLLM for arbitrary OpenAI-compatible endpoints is comparable to LangChain's 27-provider init_chat_model registry and exceeds ADK's native tier; the LiteLLM long tail matches CrewAI's fallback coverage
- Structured output: JSON-schema/pydantic response_format at both LLM-node and agent level, with InferenceMode strategies (DEFAULT/XML/FUNCTION_CALLING/STRUCTURED_OUTPUT) and auto-switching when response_format is set - functionally comparable to LangChain's ToolStrategy/ProviderStrategy/AutoStrategy and CrewAI's output_pydantic
- Input-side reasoning controls: Anthropic thinking budgets (thinking_enabled + budget_tokens) and OpenAI reasoning_effort/verbosity with AUTO per-model default resolution are on par with competitors' request-side knobs (the gap is output-side, see gaps)
- Anthropic prompt caching itself (ephemeral cache_control, 5m/1h TTL, configurable injection points, cache-token accounting) is at par with LangChain's and DeepAgents' Anthropic caching middleware
- Image and PDF chat inputs with agent-level media injection (images/files params) match CrewAI's crewai-files and LangChain content blocks for these two modalities
- Local model support via the Ollama node and CustomLLM (vLLM/llama.cpp OpenAI-compatible endpoints) matches everyone except LangChain's in-process HuggingFacePipeline niche
- Embedding models: 8 provider embedders shared by RAG and memory backends - fewer than LangChain's ecosystem but ahead of ADK (none) and adequate for the built-in RAG stack
- Model capability introspection via litellm get_model_info/supports_* plus the registry overlay is functionally close to LangChain's model.profile, though sourced from LiteLLM rather than models.dev


## Observability (`observability`)

### Gaps

#### [CRITICAL — confirmed] No OpenTelemetry support and no shipped third-party observability integrations

*Who sets the bar: adk, crewai, langchain, langgraph, deepagents*

ADK ships a full OTel stack (traces/metrics/logs, dual GenAI semantic conventions, auto-configured OTLP export) so any OTLP backend (Langfuse, Phoenix, Datadog, Jaeger, Grafana) works with env vars; LangChain/LangGraph/DeepAgents have LangSmith built into core plus a large ecosystem of vendor-built callback handlers; CrewAI is auto-instrumented by 18 documented vendors. Dynamiq has zero OTel code, no runtime metrics, and its only vendor 'integrations' (Langfuse, AgentOps) are copy-paste example scripts whose SDKs sit in an optional examples dependency group — and no observability vendor ships a Dynamiq instrumentor. The practical result is that a Dynamiq user cannot get traces into any standard observability backend without writing a custom callback handler, which in 2026 is disqualifying for production/enterprise adoption.

> **Verification:** Exhaustive greps of the dynamiq package (v0.59.0) found zero references to OpenTelemetry, OTLP, GenAI semantic conventions, or any observability vendor (langfuse/agentops/langsmith/weave/phoenix/datadog/mlflow) — all apparent hits were substrings like "hotel" (otel) and "summarize" (arize); the wheel ships only dynamiq/ per [tool.hatch.build.targets.wheel]. The package's sole tracing sink is DynamiqTracingClient in dynamiq/clients/dynamiq.py, which POSTs a proprietary {"runs": [...]} JSON payload to Dynamiq's own SaaS collector (collector.getdynamiq.ai/v1/traces), not OTLP, and dynamiq/callbacks/ contains no vendor integration module. Langfuse and AgentOps handlers exist only as inline classes in example scripts under examples/components/core/tracing/, with their SDKs pinned (langfuse>=2.51.2,<3.0.0; agentops>=0.3.12,<0.4.0) in the "examples" entry of [dependency-groups] — a PEP 735 dev group, so not even a pip-installable extra of the published wheel, which is slightly stronger than the claim states; docs/ has zero tracing/observability pages, and uv.lock's traceloop instrumentor list (crewai, langchain, llamaindex, haystack, agno, openai-agents) confirms no vendor ships a Dynamiq instrumentor.

#### [HIGH — confirmed] No free local run-inspection UI, TUI, or console tracer — trace inspection is gated on the paid Dynamiq platform

*Who sets the bar: adk, crewai, langchain, langgraph*

ADK ships a local, free dev UI with a sqlite-backed trace viewer, per-event debug endpoints, and graph visualization; CrewAI ships Rich console rendering of every step plus full TUI dashboards for runs/memory/checkpoints; LangChain has stdout/console/file callback handlers and set_debug globals; LangGraph launches Studio from its CLI. Dynamiq's only trace-inspection UI is the commercial Dynamiq platform reached via DynamiqTracingCallbackHandler — OSS users debugging locally get raw log lines and in-memory Run objects with no rendering surface at all, not even a verbose pretty-print callback.

> **Verification:** Every enumerated capability is absent from the shipped package: dynamiq/callbacks/ holds only base, streaming (queue/iterator handlers), tracing, and an inner-thoughts JSON splitter — no console/rich/verbose pretty-print handler; dynamiq/clients/ holds only the abstract BaseTracingClient and DynamiqTracingClient, which requires DYNAMIQ_ACCESS_KEY and POSTs runs to the commercial collector.getdynamiq.ai; dynamiq/cli/commands/ contains exactly the seven claimed modules (all remote platform management, zero grep hits for "trac"); and pyproject.toml has no rich/textual/TUI/web-UI deps, with fastapi/streamlit/chainlit confined to the unpublished dev-only "examples" dependency group. The only softening caveat is in the examples directory (not the wheel): examples/components/core/tracing/draw.py renders collected traces into a static color-coded execution-graph PNG via pygraphviz, and langfuse_tracing.py/agentops_tracing.py demo third-party observability handlers — none of which is a web UI, TUI, console callback, or CLI trace command, so the claim as stated stands.

#### [MEDIUM — confirmed] Minimal structured logging: plain-text basicConfig at import time, no JSON logs, no logger hierarchy

*Who sets the bar: adk, deepagents, crewai*

ADK has namespaced 'google_adk.*' loggers, drop-in Logging/DebugLogging plugins, and log-file management; DeepAgents' Talon emits structured JSON event records; CrewAI has typed logging events plus per-crew log files. Dynamiq's entire logging story is dynamiq/utils/logger.py: a module-level logging.basicConfig with a plain-text format string executed at import time (which also hijacks logging config of any embedding application) and a single shared logger. There is no structured/JSON output option and no per-subsystem logger namespace to filter on in production.

> **Verification:** dynamiq/utils/logger.py is exactly as claimed: logging.basicConfig with a plain-text format string runs at import time, and a single module-level logger (namespace "dynamiq.utils.logger") is imported by 149 files across the package; it is actually more invasive than claimed, since lines 22-23 also set every pre-existing logger in the process to ERROR at import. Exhaustive searches for structlog/loguru/JsonFormatter/FileHandler/addHandler/dictConfig, logging plugins, and LOG_LEVEL/LOG_FORMAT hooks across dynamiq/, docs/, tests/, and pyproject.toml found nothing; the callback system (dynamiq/callbacks/) only ships tracing/streaming handlers, where TracingCallbackHandler's JSON run serialization is tracing export, not a logging option. About seven files incidentally use logging.getLogger(__name__), but this is a stray inconsistency rather than a designed per-subsystem logger hierarchy, so the claim stands.

#### [MEDIUM — partial] Callbacks are observation-only — no hook can mutate, block, or short-circuit LLM/tool calls

*Who sets the bar: crewai, langchain, adk, deepagents*

CrewAI's before/after LLM- and tool-call hooks expose a mutable message list, can block execution, and can even request human input; LangChain middleware (wrap_model_call/wrap_tool_call) and ADK's before_* callbacks can replace requests/responses; DeepAgents' PreToolUse/PermissionRequest hooks carry decisions. Dynamiq's run_on_node_* dispatch in nodes/node.py fires handlers in a loop, ignores their return values, and swallows their exceptions, so the callback layer cannot double as a policy/guardrail/redaction layer the way competitors' hook systems do.

> **Verification:** The mechanical claim is accurate: every run_on_node_* dispatch in dynamiq/nodes/node.py (lines 1604-1847) loops over handlers, ignores return values, and catches/logs exceptions, and run_on_node_start even receives a shallow dict copy — so the callback layer itself cannot mutate inputs, replace responses, or block execution. But the broader framing that Dynamiq has no pre-execution policy/blocking/mutation capability is undercut by the per-node ApprovalConfig HITL mechanism (dynamiq/types/feedback.py; node.py get_approved_data_or_origin, called before transform_input/execute): it can block a node (including any tool node inside an Agent) via NodeSkippedException, mutate whitelisted input params via mutable_data_params, and solicit human feedback via console or streaming — a weaker, human-in-the-loop analogue of CrewAI/ADK-style before-hooks, though it is not programmatic, cannot replace responses, and lives outside the callback API.

### Low-severity (unverified)

- **No run-level token/cost aggregation helper** — LangChain ships UsageMetadataCallbackHandler/get_usage_metadata_callback that aggregates usage per model across a run; CrewAI aggregates UsageMetrics onto crew output; DeepAgents renders a per-model /tokens table. Dynamiq computes richer per-call data (including USD cost) but only attaches it to individual trace runs — a user asking 'what did this workflow run cost in total' must walk the trace tree and sum it themselves.
- **No workflow/agent graph visualization API in the package** — LangChain core ships draw_ascii/draw_mermaid/draw_mermaid_png/draw_png for any runnable graph; ADK's dev server renders agent/sub-agent/tool topology via graphviz endpoints. Dynamiq's only aid is an example script (examples/components/core/tracing/draw.py) that converts traces into a Graph pydantic model of nodes/edges without any rendering, and it is not part of the shipped package.

### Dynamiq strengths on this dimension

- Only framework of the six with built-in USD cost tracking in OSS: per-LLM-call prompt/completion/total token costs via litellm cost_per_token plus a shipped model registry (dynamiq/nodes/llms/model_registry.json) with per-token pricing including priority-tier and >512k-context rates, and cache read/creation token accounting (dynamiq/nodes/llms/base.py). ADK, CrewAI, LangChain, LangGraph, and DeepAgents all track tokens at most and delegate dollar-cost analytics to a SaaS.
- Typed tool-input delta streaming events (AgentToolInputStartData/DeltaData, AgentToolInputErrorEventMessageData, AgentToolResultEventMessageData, reasoning deltas in dynamiq/types/streaming.py) give finer-grained live ReAct-loop introspection than competitor callback surfaces — competitors stream tokens and tool start/end but not incremental tool-argument construction.
- Vendor-neutral in-process trace model: TracingCallbackHandler (dynamiq/callbacks/tracing.py) builds the complete Run/ExecutionRun tree locally and hands it to any handler, so custom exporters are straightforward (the Langfuse/AgentOps examples are ~100 lines each); LangChain/LangGraph/DeepAgents hardwire their built-in tracer to LangSmith and CrewAI's TraceListener targets only CrewAI AMP.
- Inner-thoughts extraction (dynamiq/callbacks/inner_thoughts_extractor.py, AgentStreamingParserCallback) parses agent reasoning out of raw token streams — a step-introspection utility no competitor ships as a first-class component.

### At parity

- Lifecycle callback coverage: workflow/flow/node start/end/error/skip/canceled plus inner execute-level hooks (dynamiq/callbacks/base.py) is roughly at par with LangChain's callback hierarchy and ADK's per-agent callbacks in breadth of lifecycle coverage (though smaller than CrewAI's ~190-event taxonomy).
- Trace-UI-behind-commercial-platform model: Dynamiq funnels trace inspection to the paid Dynamiq platform exactly as CrewAI funnels to AMP and LangChain/LangGraph/DeepAgents funnel to LangSmith — only ADK breaks from this pattern with a free local viewer.
- Streaming step-level events for agent runs (StreamingQueueCallbackHandler / sync+async iterator handlers) are functionally comparable to LangChain astream_events and LangGraph's messages/custom stream modes for observing an agent loop live.
- Checkpoint-based resume and time-travel primitives exist (dynamiq/checkpoints/ with APPEND snapshot mode, parent_checkpoint_id chains, resume_from, filesystem/in-memory/postgresql backends) — the same family of replayable run history as LangGraph's checkpointing, though without its get_state_history/update_state forensic-inspection API.


## Evaluation & testing (`evaluation`)

### Gaps

#### [CRITICAL — confirmed] No agent trajectory / tool-use evaluation

*Who sets the bar: adk, crewai, langchain (legacy), deepagents (internal)*

ADK ships 13 registered metrics including multi-turn trajectory, tool-use quality, task success, and hallucination detection; CrewAI has trajectory-aware ToolSelection/ParameterExtraction/ToolInvocation/ReasoningEfficiency judges; even legacy LangChain has TrajectoryEvalChain, and DeepAgents scores full trajectories internally. Dynamiq's evaluators only score final text outputs (question/answer/context strings) — nothing consumes an agent's intermediate steps, tool calls, or multi-turn behavior. For a framework whose core product is agents, output-only RAG metrics miss the thing users most need to evaluate in 2026: whether the agent took the right actions.

> **Verification:** Dynamiq's evaluations package contains only 8 output-text metrics (answer correctness, faithfulness, context precision/recall, factual correctness, BLEU, ROUGE, string match/similarity) whose run() signatures take only questions/answers/contexts/ground_truth_answers string lists, and exhaustive greps across dynamiq/, examples/, tests/, and docs/ for trajectory, tool-selection, tool-argument, intermediate-step, and multi-turn evaluation found no code that scores agent behavior — agents emit intermediate steps via checkpoints/streaming/tracing but nothing consumes them for evaluation, and there are no eval-related pyproject extras. The only caveat is that the generic LLMEvaluator (arbitrary **inputs) and PythonEvaluator (arbitrary dict[str, Any] with user-written evaluate() code) are input-agnostic, so a user could manually serialize a trace into them, but Dynamiq ships no trace schema, trajectory prompt, tool-call parser, or example doing so, so the claimed capability is genuinely absent as shipped.

#### [HIGH — confirmed] No dataset/eval-set management or experiment runner

*Who sets the bar: adk, crewai, deepagents, langchain (legacy load_dataset/run_on_dataset)*

ADK has EvalSet/EvalCase pydantic schemas with local/GCS/in-memory managers plus scenario auto-generation; CrewAI's ExperimentRunner executes a dataset of test cases against a crew and persists ExperimentResults; DeepAgents versions eval datasets in-repo. Dynamiq has no dataset or test-case abstraction at all — callers must hand-assemble parallel Python lists and loop themselves (examples/components/evaluations/workflow_eval.py does exactly this by hand). Without a testset primitive, teams cannot organize, version, or re-run eval suites, which is the entry point to any serious eval workflow.

> **Verification:** Exhaustive search of dynamiq/ (source), examples/, docs/, tests/, and pyproject.toml for EvalSet/EvalCase/TestCase/TestSet/Experiment/Dataset/testset/golden/runner/suite terms found no eval dataset or test-case abstraction, no dataset loader, no CLI eval command, and no runner that executes cases against a workflow or aggregates/persists results. dynamiq/evaluations contains exactly BaseEvaluator (run() raises NotImplementedError), LLMEvaluator (run(**inputs) zips caller-supplied parallel lists), PythonEvaluator, and 8 per-metric evaluators whose run() signatures take parallel lists such as run(ground_truth_answers: list[str], answers: list[str]). The cited example examples/components/evaluations/workflow_eval.py indeed hand-assembles single-element parallel lists and manually builds a metrics dict; the only nuance is that each evaluator batch-scores its input lists internally, which is per-metric scoring, not a test-set/experiment primitive.

#### [HIGH — confirmed] No CI/regression testing harness (pytest integration, baselines, record/replay)

*Who sets the bar: adk, crewai, langchain, deepagents (CI workflows)*

ADK's AgentEvaluator is a pytest entry point with per-file EvalConfig, plus `adk conformance record|test` gives deterministic record/replay regression testing; CrewAI ships assert_experiment_successfully and baseline-drift comparison for CI; LangChain's langchain-tests is a subclassable conformance suite. Dynamiq offers nothing to put agent quality under CI: no assertion helpers, no baseline persistence/comparison, no recorded-session replay. Teams adopting Dynamiq must build their own regression story from raw metric classes, which is a common deal-breaker for production engineering orgs.

> **Verification:** Dynamiq's evaluation surface is limited to raw metric/evaluator classes in dynamiq/evaluations (LLMEvaluator, PythonEvaluator, and metrics like FaithfulnessEvaluator) that return lists of float scores with no assertion helpers, thresholds, or pass/fail semantics. There is no pytest plugin or entry point (pyproject.toml registers only the dynamiq CLI, whose commands cover org/project/service/config — no eval command), no baseline persistence or comparison anywhere (greps for baseline/regression/golden/drift/snapshot found nothing eval-related), and no record/replay machinery — the checkpoints module is durable-execution resume for HITL/iterations, not deterministic test replay. The tests/ directory (unit, integration, integration_with_creds, smoke) tests the framework itself, and examples/components/evaluations contains only ad-hoc scoring scripts, so the claim is accurate as stated since it already concedes the raw metric classes exist.

#### [MEDIUM — confirmed] No eval CLI commands or local eval UI

*Who sets the bar: adk, crewai, deepagents*

ADK has `adk eval` / `adk eval_set` plus a full eval tab in its dev UI; CrewAI has `crewai test` with rich result tables; DeepAgents has a deepagents-evals CLI with trials/aggregation/radar charts. Dynamiq ships a `dynamiq` CLI, but its subcommands (config, context, org, project, resource_profiles, service) are all paid-platform management — there is no way to run an evaluation from the command line or a local UI. This pushes even basic eval workflows to Python scripting or the commercial platform, which weakens the OSS onboarding story.

> **Verification:** The dynamiq CLI (entry point dynamiq.cli:main) registers only org, project, service, resource-profiles, and config command groups, whose subcommands (list/get/set/create/deploy/status/update/delete/show) are all HTTP calls to the paid platform API; a case-insensitive grep for "eval" across dynamiq/cli/ returns zero matches, pyproject.toml declares only the single "dynamiq" console script with no eval-related extras, and no __main__.py exists anywhere in the package. Evaluations exist solely as a Python library (dynamiq/evaluations/ with LLMEvaluator, PythonEvaluator, and 9 metrics) driven by user-written scripts like examples/components/evaluations/workflow_eval.py — exactly the scripting fallback the claim describes. No local dev UI exists: the package ships no HTML/JS assets, all streamlit/fastapi code lives in unrelated examples/ demos, and "Dynamiq UI" references in docs/examples denote the commercial platform.

#### [MEDIUM — confirmed] No user simulation for multi-turn evaluation

*Who sets the bar: adk, deepagents (internal), langgraph (notebook example)*

ADK ships LlmBackedUserSimulator with prebuilt personas, conversation scenarios, simulator-quality metrics, and even TTS-based voice-user simulation; DeepAgents runs tau-bench-style LLM customer simulators; LangGraph at least documents the pattern in notebooks. Dynamiq has no simulated-user utility of any kind, so multi-turn conversational agents cannot be evaluated end-to-end without hand-writing the simulation loop. As multi-turn task success becomes the headline agent metric, this compounds the trajectory-evaluation gap.

> **Verification:** Exhaustive case-insensitive searches of dynamiq/, examples/, docs/, tests/, and pyproject.toml for simulat*, persona, UserSimulator/SimulatedUser, user_proxy, roleplay, tau-bench, synthetic user, self-play, multi-turn, and class names matching User/Simulator/Persona/Scenario found no user-simulator class, prebuilt persona, or conversation-simulation utility. dynamiq/evaluations contains only single-turn text/RAG metrics (answer correctness, faithfulness, context precision/recall, factual correctness, BLEU/ROUGE/string) plus generic LLM/Python evaluators, and examples/components/evaluations mirrors exactly those. All "simulat"/"persona" hits are incidental (crash-simulation tests, iOS-simulator wheels in uv.lock, agent-side persona/style prompt sections), so multi-turn agent evaluation would indeed require hand-writing the simulation loop.

### Low-severity (unverified)

- **No eval-platform integrations for scores** — CrewAI has a Patronus eval tool, ADK delegates two metrics to Vertex Gen AI Eval, and the LangChain ecosystem wires results into LangSmith. Dynamiq's only third-party observability touchpoint is a Langfuse tracing callback that lives in examples/ (not the installed package), and no code pushes evaluation scores anywhere. Mitigated because Dynamiq's callback/tracing system lets platforms ingest traces and score externally, and Dynamiq's own paid platform is the intended destination — but there is no documented OSS path.
- **No eval-driven optimization or training loop** — ADK's `adk optimize` uses its eval metrics to drive GEPA-based prompt/agent optimization, and CrewAI's `crewai train` runs an iterative human-feedback loop whose output is injected into future runs. Dynamiq's metrics are score-and-report only — there is no mechanism that feeds evaluation results back into improving prompts or agents. A nice-to-have differentiator rather than table stakes, but it shows where metric suites are heading.

### Dynamiq strengths on this dimension

- Only framework of the six with a first-class, stable (non-experimental, non-legacy) RAG metric suite in the core installable package: FaithfulnessEvaluator, ContextPrecision/Recall, Answer/FactualCorrectness implemented as real claim-decomposition + NLI-verdict pipelines with typed intermediate models — CrewAI's equivalents are experimental-namespaced, LangChain's are frozen legacy, LangGraph ships none, ADK's built-ins are agent-centric without RAG context precision/recall
- LLMEvaluator is a genuinely clean generic LLM-as-judge API: declarative instructions, typed input/output schemas, validated few-shot examples, batch execution, output sanitization — simpler to adopt for custom judges than ADK's rubric machinery or CrewAI's subclass-based evaluators
- PythonEvaluator lets users define deterministic custom metrics as sandboxed Python code (restricted-Python execution) — no competitor offers code-as-metric with a safety sandbox
- Judge-model agnosticism: every metric takes any BaseLLM node, so judges run on any of Dynamiq's 28 providers with no cloud-project requirement (contrast ADK's safety_v1/response_evaluation_score requiring paid Vertex Gen AI Eval)
- End-to-end runnable example (examples/components/evaluations/workflow_eval.py) scoring a YAML-defined RAG workflow's answers — eval wired to the framework's declarative workflow format

### At parity

- Deterministic string/NLP metrics (BLEU via sacrebleu, ROUGE, exact match, string presence, string similarity with configurable distance measures) — at par with LangChain-classic's string evaluators and ADK's response_match_score
- Final-output LLM-as-judge quality scoring — roughly at par with CrewAI's semantic/goal judges and LangChain-classic's Criteria/ScoreString chains (though below ADK's rubric-based judge system)
- Delegating advanced eval tooling (datasets UI, experiments, regression tracking) to a commercial platform is the same posture as LangChain/LangGraph with LangSmith and CrewAI with AMP — Dynamiq is not alone in this split, though ADK proves the full stack can ship in OSS
- Unit/integration test coverage of the eval metrics themselves (tests/integration/evaluations/, tests/unit/components/evaluators/) — the metrics are tested code, comparable to competitors' maintained eval modules


## Deployment & developer experience (`deployment_devux`)

### Gaps

#### [CRITICAL — confirmed] No local dev UI, playground, or interactive run surface

*Who sets the bar: Google ADK, LangGraph, CrewAI, DeepAgents*

Every competitor ships some interactive local surface: ADK has 'adk web' (Angular dev UI with chat, trace viewer, eval tab, graph view, and a conversational visual builder) plus 'adk run' terminal chat; LangGraph has 'langgraph dev' with hot reload and Studio; CrewAI ships three Rich TUIs plus 'crewai chat'; DeepAgents ships dcode, a full Claude-Code-class TUI. Dynamiq has nothing interactive in OSS — the CLI only registers org/project/service/resource-profiles/config commands (all clients for the paid cloud), and the visual builder exists only on the paid platform. This is the single most visible DX differentiator in 2026: developers evaluate frameworks by running an agent locally and watching it, and Dynamiq offers no first-party way to do that beyond writing Python scripts or copying Streamlit/Chainlit examples.

> **Verification:** The claim is accurate: dynamiq/cli/commands/utils.py registers only org, project, service, resource-profiles, and config groups (plus aliases), and every command is a requests-based HTTP client against the hardcoded paid-platform URL https://api.getdynamiq.ai (dynamiq/cli/config.py); even 'service deploy' just uploads a docker context to the cloud API. Exhaustive greps across dynamiq/ source for streamlit, chainlit, gradio, playground, uvicorn, fastapi, textual, Flask, ASGI, websocket servers, prompt_toolkit, curses, and serve/run/chat/dev/studio commands found no dev server, web UI, TUI, playground, or interactive chat/run command — all hits were false positives (e.g., 'ContextualSplitter') or DB/LLM connection defaults, and pyproject.toml declares a single console script (dynamiq.cli:main) with Streamlit/Chainlit/FastAPI confined to the 'examples' dependency group and examples/ directory, exactly the copy-an-example workaround the claim describes. The only interactive primitive in source is console input() in the HumanFeedbackTool (dynamiq/nodes/tools/human_feedback.py) for human-in-the-loop steps inside user-written workflows, which is not a first-party interactive surface for running/watching agents.

#### [HIGH — confirmed] No project scaffolding or templates (no init/new/create command)

*Who sets the bar: Google ADK, CrewAI, LangGraph, DeepAgents*

ADK has 'adk create' with interactive model/backend prompts, CrewAI ships five project templates (crew, flow, JSONC crew, declarative YAML flow, tool), LangGraph has 'langgraph new' pulling templates from GitHub, and DeepAgents has 'deepagents init'. Dynamiq has no scaffolding command and no template directory anywhere in the repo — a new user's first five minutes are copy-pasting from the 273 examples rather than generating a working project. This is near-table-stakes onboarding UX for CLI-bearing frameworks in 2026.

> **Verification:** The dynamiq CLI (entry point dynamiq.cli:main) exposes only remote-platform management commands — org list/set, project list/set, service list/get/create/deploy/status/update/delete, resource-profiles list, config show — with no init, new, or scaffolding command; the only 'create' subcommand (dynamiq service create) POSTs to the cloud API /v1/services and writes nothing locally. Exhaustive greps for scaffold/cookiecutter/copier/boilerplate/starter/template and finds for template directories or .j2/.tmpl files found only agent prompt templates, not project scaffolds. The quickstart doc confirms onboarding is pip install plus copy-pasting snippets (or git-cloning the repo for its exactly-273 example files), matching the claim.

#### [HIGH — confirmed] No OSS API-server generation or serve command

*Who sets the bar: Google ADK, LangGraph, DeepAgents*

ADK's 'adk api_server' generates a full FastAPI app (run/run_sse/run_live endpoints, session/artifact/memory CRUD, hot agent reload); LangGraph's CLI runs an in-memory API server locally and generates the production stack; DeepAgents auto-generates langgraph.json and manages a langgraph dev subprocess. Dynamiq serves workflows only via hand-written FastAPI shown in examples (SSE/WebSocket servers under examples/components/core/websocket/) — nothing in the installable package imports FastAPI or starts an HTTP server. Users must build and maintain their own serving layer, session handling, and streaming endpoints. CrewAI shares this gap in OSS, but the two best-in-class competitors do not.

> **Verification:** An AST scan of every module in the installable dynamiq/ package found no imports of FastAPI or any other web-server framework (uvicorn, starlette, flask, aiohttp, websockets, sse_starlette), and fastapi appears in pyproject.toml only in the dev-only 'examples' dependency-group; all HTTP/SSE/WebSocket serving code is hand-written under examples/. The claim's one omission is that the package ships a 'dynamiq service create/deploy' CLI, but reading it shows it is only a deployment client for Dynamiq's commercial cloud (tars user source and POSTs to /v1/services/{id}/deploy) — it generates no serving code and starts no local server, and the agent_service example it deploys is a fully user-authored FastAPI app, which actually reinforces that users must build their own serving layer. Nothing comparable to ADK's generated api_server or LangGraph's local dev API server exists; dynamiq/types/streaming.py is transport-less in-process streaming and MCPServer in nodes/tools/mcp.py is an MCP client despite its name.

#### [MEDIUM — confirmed] CLI deploys only to the paid Dynamiq cloud, with no container or deploy-artifact generation

*Who sets the bar: Google ADK, LangGraph*

ADK generates Dockerfiles and deploys to three targets (Cloud Run, Agent Engine, GKE with deployment.yaml); LangGraph generates Dockerfile + docker-compose for BYO infrastructure and runs a full local Dockerized prod stack ('langgraph up'). Dynamiq's 'service deploy' requires a user-supplied Dockerfile and targets only the Dynamiq platform API — there is no Dockerfile/compose generation and no self-hosted or third-party-cloud deployment path. CrewAI and DeepAgents also funnel to paid platforms, so this is not uniquely bad, but Dynamiq additionally lacks the artifact-generation half that softens the lock-in.

> **Verification:** The entire dynamiq/cli package (692 lines) is a thin wrapper over the Dynamiq platform HTTP API (default host https://api.getdynamiq.ai): 'service deploy' tars the user's source and POSTs it with a reference to a user-supplied Dockerfile path ({"docker": {"file": ..., "context": ...}}) to /v1/services/{id}/deploy, where the platform builds it — no code anywhere generates a Dockerfile, docker-compose file, or Kubernetes manifest (exhaustive greps for Dockerfile/compose/kubernetes/helm/manifest/gke/cloud run/fargate/terraform/buildpack/template files across dynamiq/, docs/, examples/, tests/, and pyproject.toml found nothing). The repo's own Dockerfile builds a 'develop' target with tests/examples/pre-commit, and docker-compose.yaml contains only test-runner services plus an ephemeral Neo4j for integration tests — framework development/testing only. The examples/cli/agent_service walkthrough ships a hand-written static Dockerfile, confirming the user-supplied-Dockerfile, platform-API-only deployment model; the only escape hatch is an overridable API host env var, which still requires a Dynamiq-platform-compatible API.

#### [MEDIUM — confirmed] Package is not typed for consumers: no py.typed marker, no strict type checking

*Who sets the bar: LangChain 1.x, CrewAI, Google ADK, DeepAgents, LangGraph*

CrewAI ships py.typed in all five packages with mypy strict=true; LangChain runs mypy strict with reference-grade generics (typed structured-output threading); ADK and DeepAgents both ship py.typed. Dynamiq uses pydantic v2 models pervasively at runtime but ships no py.typed marker and configures no mypy or pyright anywhere, so downstream users' type checkers treat dynamiq as an untyped package and all its annotations are invisible to IDE/CI type checking. This is a cheap fix with outsized perception impact among the senior engineers who evaluate frameworks.

> **Verification:** No py.typed marker exists anywhere in the repo (find across the full tree returned nothing, and the dynamiq/ package dir contains only __init__.py plus subpackages), and the hatchling wheel config has no force-include that could add one at build time. No mypy or pyright configuration exists in any config surface: pyproject.toml has only hatch/bandit/isort/pytest tool sections, setup.cfg only flake8/pycodestyle, there is no mypy.ini/pyrightconfig.json/tox.ini, the pre-commit config runs flake8/darker/bandit but no type checker, the Makefile lint target just runs pre-commit, CI workflows run pre-commit and pytest only, and the dev dependency group contains no mypy/pyright. The only mypy strings are .gitignore boilerplate (.mypy_cache/) and the transitive mypy-extensions dependency of black in uv.lock, so the claim is fully confirmed.

#### [MEDIUM — partial] In-repo docs are thin and the mkdocs build is broken; no llms.txt

*Who sets the bar: CrewAI, Google ADK, DeepAgents, LangChain 1.x*

CrewAI ships version-pinned docs in four languages in-repo; ADK ships in-repo guides plus llms.txt/llms-full.txt for AI-assisted coding and an AGENTS.md. Dynamiq's docs/ holds only three tutorial files (quickstart, rag, agents), mkdocs.yml references a docs_dir ('mkdocs') that does not exist in the repo so the docs site cannot be built from source, and there is no llms.txt for coding-assistant consumption. The 273 examples carry the teaching load, but examples without narrative docs raise time-to-first-success for a surface area this large.

> **Verification:** Two legs are literally true: docs/ holds only three tutorial md files plus one logo image, and no llms.txt/llms-full.txt/AGENTS.md exists anywhere (only .cursor/BUGBOT.md, a Cursor review-rules file). However, the central inference about mkdocs is false: the 'mkdocs' docs_dir is intentionally gitignored (.gitignore line 163) and is generated from source by 'make build-mkdocs', which runs scripts/generate_mkdocs.py to emit mkdocstrings API-reference pages for every module in dynamiq/, copies README.md and docs/tutorials in, and builds; CI (ci.yaml jobs build-mkdocs/publish-updated-mkdocs) builds and gh-deploys this on main. So the docs site CAN be built from source and Dynamiq does ship an auto-generated API reference, though narrative/concept docs remain limited to three tutorials and there is genuinely no llms.txt.

#### [MEDIUM — partial] No workflow/graph visualization in OSS

*Who sets the bar: CrewAI, Google ADK, LangGraph*

CrewAI's 'crewai flow plot' renders an interactive HTML graph, ADK renders agent graphs via graphviz in its dev UI, and LangGraph Studio visualizes graphs live. Dynamiq — despite having a fully declarative YAML DAG format that is ideal for rendering — has no code that draws a Workflow/Flow as an image or HTML page; visualization exists only in the paid platform builder. This undercuts the framework's own strongest asset: portable YAML workflows that nobody can see.

> **Verification:** The installable dynamiq package genuinely has no visualization: Workflow/Flow/GraphOrchestrator expose only to_dict/to_yaml serialization, and the CLI (dynamiq/cli) has only platform commands (org, project, service deploy, config) with no plot/render command; docs and tests contain zero mermaid/graphviz/visualization references. However, the repository does contain examples/components/core/tracing/draw.py, which renders workflow and graph-orchestrator graphs (nodes plus DAG 'depends' edges) to PNG via pygraphviz, with ~11 ready-made wrappers for agents, orchestrators, and YAML workflows. This refutes the categorical 'no code that draws a Workflow/Flow as an image' framing, but only weakly: it is example-only code excluded from the wheel, requires actually executing the workflow (trace-based, cannot render static YAML), depends on pygraphviz which sits in a dev-only 'examples' dependency group rather than a pip extra, and is undocumented — so the practical gap versus CrewAI/ADK/LangGraph visualization tooling largely stands.

### Dynamiq strengths on this dimension

- Full YAML round-trip serialization (WorkflowYAMLLoader + dumper with ConnectionManager injection) makes workflows portable, diffable artifacts — a stronger declarative story than LangChain, LangGraph, or DeepAgents, and unlike CrewAI/ADK it covers dump as well as load, so programmatically-built workflows can be exported
- Durable execution fully in OSS: checkpoints package with InMemory/Filesystem/PostgreSQL backends, APPEND (time-travel) vs REPLACE semantics, and resume_from at flow, node, and tool granularity — LangGraph gates comparable server-side durability behind the proprietary langgraph-api image, and CrewAI/ADK have nothing equivalent in OSS
- Example corpus breadth: 273 example .py files covering nearly every node plus 17 complete use-case apps (gpt_researcher, customer_support, data_analyst...), including working SSE/WebSocket FastAPI servers and Streamlit/Chainlit chat UIs — more end-to-end serving examples than any competitor ships in-repo
- Workflow-generation utilities (utils/workflow_generation/) that emit node YAML and sample inputs from JSON schemas, making workflows machine-generatable — infrastructure for LLM-driven or visual workflow construction that only ADK's builder assistant parallels
- Rich engine-level DX that competitors leave to user code: per-node ErrorHandling (timeout/retry/backoff), jsonpath input/output transformers, Choice/Map/Pass operators, Redis-backed node output caching, and thread/process executor pools configured declaratively per node

### At parity

- Managed-cloud deployment funnel: 'dynamiq service deploy' to the paid platform is structurally the same commercial pattern as CrewAI's 'crewai deploy' to AMP and DeepAgents' 'deepagents deploy' to LangSmith — a working paid path, just without the OSS-side generation extras
- Declarative YAML agent/workflow configuration is at par with ADK's root_agent.yaml and CrewAI's YAML/JSONC tiers as an authoring format (and ahead on round-trip, behind on schema publication and template tooling)
- Runtime typing discipline: pervasive pydantic v2 models with per-node typed input schemas is comparable to ADK's and CrewAI's runtime modeling (the gap is consumer-visible typing via py.typed/strict checking, tracked separately)
- Test-suite scale: 343 test files across unit/integration/smoke tiers is comparable to CrewAI's 364 and in line with the other majors
- Docs hosted on an external site rather than in-repo is the same pattern as LangChain, LangGraph, and DeepAgents (docs.langchain.com) — the gap is the thin in-repo layer and broken build, not the external-site model itself


## Safety & guardrails (`guardrails_safety`)

### Gaps

#### [CRITICAL — confirmed] No agent-level guardrail pipeline or interception middleware

*Who sets the bar: langchain, crewai, adk, langgraph*

LangChain 1.x ships a first-class middleware layer (before/after model, wrap_model_call, wrap_tool_call) where PII, moderation, and HITL guards compose declaratively onto an agent; CrewAI attaches guardrails directly to tasks (including natural-language string guardrails auto-wrapped into an LLM judge, with retry budgets); ADK's plugin/callback hooks and LangGraph's tool-call interceptor can block or rewrite calls in flight. Dynamiq's PII/prompt-injection/LlamaGuard detectors are standalone workflow nodes the user must wire into a flow manually, and its callback system is purely observational — no hook can veto, rewrite, or retry an agent's LLM or tool call. This makes 'add guardrails to my agent' a graph-surgery exercise instead of a one-line config, which is table stakes in 2026.

> **Verification:** Exhaustive search confirms the claim: the Agent class (base.py lines 233-343) exposes no guardrail/middleware/hook parameters; the callback interface (callbacks/base.py) is purely observational — all methods return None, and node.py invokes them inside try/except blocks that log and discard errors, so a callback cannot block, rewrite, or even abort execution by raising; and no code in dynamiq/nodes/agents/ references the detector nodes (the only references to LlamaGuard/PIIDetector/PromptInjectionDetector outside their own package are the NodeGroup.DETECTORS enum and one test — not even examples/ use them). The closest near-misses — jq-style InputTransformer/OutputTransformer for node-boundary data mapping, the voluntary HumanFeedbackTool for HITL, standalone validator nodes, and per-tool safety flags (Exa moderation, Cypher write-block) — none can veto or rewrite an agent's LLM/tool call in flight, so detectors must indeed be wired manually as separate workflow nodes.

#### [HIGH — confirmed] PII handling is detection-only, with no redaction and no offline detectors

*Who sets the bar: langchain, crewai (enterprise)*

LangChain's PIIMiddleware detects and then applies block/redact/mask/hash strategies to agent input and output, including rewriting streamed message deltas in flight, using local regex/Luhn detectors that need no network call; CrewAI offers PII trace redaction on its enterprise platform. Dynamiq's PIIDetector only returns a detection verdict and requires a remote HuggingFace inference API or Lakera Guard call — there is no transformer that removes or masks the PII from message content, no streaming-aware redaction, and no local fallback. For privacy-sensitive enterprise buyers, detection without remediation is half a feature.

> **Verification:** PIIDetector (dynamiq/nodes/detectors/pii_detector.py) only returns {"is_detected", "detected_pii"} verdicts from remote HuggingFace inference API or Lakera Guard calls — it even discards the span offsets HF returns, and no other code in the package consumes its output. Exhaustive greps for redact/mask/anonymize/scrub/pseudonymize/presidio/luhn across dynamiq/, examples/, docs/, tests/ and pyproject extras found no local PII detector, no content-rewriting/masking/hashing node, and no streaming redaction; the only "[REDACTED]" usage is SANITIZED_VALUE_PLACEHOLDER for pgvector connection-string sanitization, exactly as claimed. The sole nuance is that connections exclude secrets from trace serialization (to_dict(for_tracing=True) returns only id/type) and tool/file names are "sanitized" — credential hygiene and protocol fixes, not PII remediation of message or trace content — so the claimed gap is real.

#### [HIGH — confirmed] No programmatic tool-call policy hooks or declarative permission system

*Who sets the bar: langgraph, crewai, adk, deepagents*

Competitors provide enforcement points around individual tool actions: LangGraph's on_tool_call interceptor can inspect/override/deny any tool call, CrewAI's before/after tool hooks can veto or mutate invocations, ADK ships per-tool write-mode policies (BigQuery/Spanner WriteMode BLOCKED by default) and BashToolPolicy with rlimits, and DeepAgents has a glob-based filesystem permission system with allow/deny/interrupt rules that fails closed. Dynamiq has scattered partial equivalents — per-node HITL approval, MCP include/exclude tool lists, a shell blocked_commands substring denylist, path-traversal validation, and a writes_allowed flag on the Cypher tool — but no general hook to programmatically gate tool calls and no rule-based permission model (its SQL executor, for instance, has no read-only mode).

> **Verification:** Exhaustive searches (hook/interceptor/middleware/policy/permission/deny/allowlist/readonly/write_mode/glob/fnmatch across dynamiq/, examples/, docs/, tests/, pyproject.toml) found no before/after tool-call hook, no rule-based permission model, and no SQL read-only mode. The agent tool dispatch (_run_tool in dynamiq/nodes/agents/base.py) has no user-registerable gate; the closest thing, NodeCallbackHandler callbacks, is observation-only — inputs are passed as copies and callback exceptions are explicitly swallowed (node.py run_on_node_start), so callbacks can inspect but cannot deny or modify a tool call. The only gating mechanisms are exactly those the claim enumerates (ApprovalConfig console/stream HITL, MCP include_tools/exclude_tools static lists, shell blocked_commands substring denylist, Cypher writes_allowed regex guard, file-tool path-traversal validation), plus trivia like ToolParams static input overrides and SubAgentTool.max_calls budgets — none constitute a programmatic gate or declarative permission system, and sql_executor.py executes any statement including INSERT/UPDATE/DELETE/DDL with no write-blocking flag.

#### [MEDIUM — confirmed] No SSRF protection on HTTP/web tools

*Who sets the bar: langchain, crewai, deepagents*

LangChain core ships an SSRF policy module (URL and resolved-IP validation blocking private networks), CrewAI's safe_requests blocks private/reserved IPs and strips credentials on cross-origin redirects, and DeepAgents pins DNS across redirects to defeat rebinding. Dynamiq's HttpApiCall node and web-fetch/scraping tools perform no destination validation at all, so a prompt-injected agent can be steered into calling cloud metadata endpoints or internal services. This matters for anyone running agents inside a VPC.

> **Verification:** The claim is fully confirmed. dynamiq/nodes/tools/http_api_call.py takes the URL directly from input/config/connection (_build_request_kwargs, lines 191-201) and passes it straight to requests/httpx (self.client.request, line 267; client.request, line 287) with no hostname parsing, no private/reserved/loopback-IP blocking, no DNS resolution or pinning, and no URL allow/deny policy. The ipaddress module is imported nowhere in the repository (grep for 'import ipaddress'/'from ipaddress' across the whole tree returns zero hits, including tests/docs/examples). Worse, the Http connection's async client (connections/connections.py lines 236-250) explicitly sets follow_redirects=True and trust_env=True, widening the SSRF/rebinding surface with no credential stripping on redirects; the web-fetch/scraping tools (firecrawl, jina, zenrows, scale_serp, tavily) also perform no local destination validation. The only allow/deny-list matches in the codebase are unrelated (streaming tool-input allowlist, python sandbox dunder allow-list, firecrawl blockAds, exa_search SERP domain filters, scale_serp query-vs-url presence check).

#### [MEDIUM — confirmed] No at-rest encryption for checkpoints or persisted state

*Who sets the bar: langgraph*

LangGraph provides an EncryptedSerializer (AES-GCM with pluggable cipher) that encrypts all checkpoint payloads at rest, plus customer-managed encryption handlers for server data, and hardens deserialization with module allowlists. Dynamiq's checkpoint system persists full workflow state — including conversation content and stored HITL approval responses — with no encryption option. For regulated-industry deployments where agent state contains sensitive data, this is a compliance checkbox Dynamiq cannot tick.

> **Verification:** Exhaustive case-insensitive searches for encrypt/cipher/AES/fernet/crypto/kms/pgcrypto/EncryptedSerializer across dynamiq/, examples/, docs/, tests/, and pyproject.toml found no encryption capability — the only matches were incidental words like "aesthetics" and "cryptocurrency" in example prompts. FlowCheckpoint payloads are serialized as plaintext JSON (orjson/json.dumps with encode_reversible, which only base64-encodes bytes) and written unencrypted by all three backends: FileSystem writes plain .json files, PostgreSQL inserts plain JSON into the data column, and InMemory stores objects directly. There is no serializer abstraction, cipher hook, or encryption config field in CheckpointBackend or CheckpointConfig, and no cryptography dependency or extra in pyproject.toml; the only "secure"-related feature (include_secure_params) merely masks connection API keys in config dumps, unrelated to payload encryption at rest.

### Low-severity (unverified)

- **No authentication/authorization abstraction for deployed agents** — LangGraph's SDK includes a resource.action-scoped auth DSL with row-level filters over threads/runs/assistants (enforced by its platform server), and CrewAI documents enterprise RBAC. Dynamiq's OSS framework has no authn/authz layer for multi-user or multi-tenant access to workflows and runs. Most OSS frameworks also lack this and enforcement typically lives platform-side, so impact is limited, but it surfaces in enterprise platform evaluations.

### Dynamiq strengths on this dimension

- Only framework of the six that ships a prompt-injection/jailbreak detector in OSS code (dynamiq/nodes/detectors/prompt_injection_detector.py, HuggingFace classifier or Lakera Guard) — LangChain, LangGraph, CrewAI, ADK, and DeepAgents all have none.
- Broadest OSS content-moderation coverage: LlamaGuardDetector plus Lakera Guard integration; ADK/CrewAI/LangGraph/DeepAgents ship zero moderation hooks and LangChain only covers the OpenAI moderation endpoint.
- Three-tier code-execution containment is unmatched in shape: hardened in-process RestrictedPython with a custom AST transformer (dynamiq/nodes/tools/python.py), the Monty Rust minimal interpreter (python_monty.py), and remote E2B/Daytona sandboxes including a desktop sandbox for computer use (dynamiq/sandboxes/) — ADK matches remote-sandbox breadth but nobody else offers a hardened local Python interpreter tier.
- Plan-approval HITL: PlanApprovalConfig (dynamiq/types/feedback.py) lets a human approve or veto an agent's task plan before execution as a typed streaming-event protocol — no competitor has plan-level (vs tool-level) approval.
- Approval gates are universal and checkpoint-durable: ApprovalConfig exists on every node (dynamiq/nodes/node.py:273), applies to tools invoked inside agent loops, supports mutable_data_params so the human can edit inputs, and approval responses persist in checkpoints (dynamiq/checkpoints/checkpoint.py) for resume after restart.
- Sandbox sharing across delegated sub-agents (SharedSession/SandboxSharingScope) and routing of large tool outputs to sandbox files — operational sandbox ergonomics competitors lack.

### At parity

- Durable human-in-the-loop tool approval: checkpoint-stored approval responses give Dynamiq restart-safe HITL roughly at par with LangChain/LangGraph checkpointed interrupts, though with fewer decision types (no built-in edit/reject/respond schema per tool call).
- MCP tool filtering: include_tools/exclude_tools lists on MCP servers (dynamiq/nodes/tools/mcp.py) match CrewAI's StaticToolFilter; neither has dynamic trust stores like DeepAgents.
- Path-traversal defense on file tools (validate_file_path in dynamiq/nodes/tools/file_tools.py) is at par with CrewAI's safe_path helpers.
- Shell command denylisting (blocked_commands on the sandbox shell tool) is comparable to ADK's BashToolPolicy, minus ADK's POSIX resource limits.
- Secret handling: env-var-resolved connection keys plus pervasive include_secure_params=False serialization (dumped YAML and traces don't leak keys) is at par with LangChain's SecretStr/lc-json approach; weaker only vs ADK's full Secret Manager/credential-service architecture.
- Structural output validation: RegexMatch/ValidChoices/ValidJSON/ValidPython validator nodes are comparable to LangGraph's ValidationNode and CrewAI's callable guardrails for schema-level checks.
- Eval-time faithfulness/hallucination metrics (dynamiq/evaluations/metrics/faithfulness.py) are at par with ADK's eval-time safety metrics and strictly better than CrewAI's OSS HallucinationGuardrail, which is a no-op upsell.
- Runaway protection: max_loops on agents and max_calls on sub-agent tools cover the basic budget case, though without LangChain's thread/run-scoped model- and tool-call limit middleware.


## Interoperability & protocols (`interop_protocols`)

### Gaps

#### [HIGH — confirmed] No A2A protocol support and no way to consume or expose network-addressable remote agents

*Who sets the bar: adk, crewai, langgraph (platform-gated), deepagents (Agent Protocol equivalent)*

ADK is the A2A reference implementation (RemoteA2aAgent to consume, to_a2a to serve, agent cards, interceptors), CrewAI ships one of the deepest A2A stacks (client+server building blocks, auth schemes, polling/streaming/push updates, parallel delegation, A2UI extension), and every LangGraph Platform deployment serves /a2a; even DeepAgents has a practical remote-agent path via Agent Protocol async subagents, and LangGraph has RemoteGraph for cross-deployment composition. Dynamiq has no A2A code at all and no mechanism to embed a remote agent as a node or expose an agent as a network service — agents are strictly in-process. In 2026 enterprise multi-agent RFPs increasingly ask for A2A or an equivalent remote-agent protocol, and Dynamiq has nothing to point at in OSS.

> **Verification:** Case-insensitive grep for 'a2a', 'agent_card', 'AgentCard', and 'agent2agent' returns zero matches across dynamiq/, tests/, examples/, and docs/ (only incidental hex-hash substrings in uv.lock), and pyproject.toml (v0.59.0) declares no a2a-sdk dependency (extras are only 'cua' and 'monty'). The closest capabilities all fall short of the claim's bar: SubAgentTool (dynamiq/nodes/tools/agent_tool.py) wraps only in-process agent instances/factories; the MCP node (dynamiq/nodes/tools/mcp.py) is a client for remote MCP *tools* with no server side and no agent-card/A2A semantics; HttpApiCall is a generic REST tool; and the CLI 'service deploy' (dynamiq/cli/commands/service.py) is generic Docker hosting on the Dynamiq platform, not an agent protocol. There is no node or class that invokes a remote agent in another process/deployment as a sub-agent, and no framework mechanism to expose an agent as a network service.

#### [HIGH — confirmed] No MCP server mode — Dynamiq agents/workflows cannot be exposed as MCP tools

*Who sets the bar: adk, langgraph (platform-gated)*

ADK exposes any agent as a FastMCP server via to_mcp_server, and every LangGraph Server deployment serves /mcp by default, making agents callable from Claude, IDEs, and any MCP host. Dynamiq's MCPServer class is a client-side aggregator that consumes MCP servers; nothing in the package exposes a Dynamiq workflow or agent as an MCP server. This blocks the fastest-growing distribution channel for agents in 2026 (being a tool inside other agents/hosts), though CrewAI and DeepAgents also lack it in OSS.

> **Verification:** Exhaustive searches across dynamiq/, examples/, tests/, docs/, and pyproject.toml confirm every element of the claim: the only mcp.server/FastMCP usage is the two demo servers under examples/components/tools/mcp_server_as_tool/mcp_servers/ (standalone math/weather FastMCP demos used to exercise the client, not wrapping any Dynamiq workflow); dynamiq/nodes/tools/mcp.py imports only ClientSession and its MCPServer class (line 488) is a client-side aggregator that connects to external MCP servers and wraps their tools; dynamiq/connections/connections.py imports only sse_client, streamablehttp_client, stdio_client, and StdioServerParameters. No to_mcp/as_mcp/serve_mcp/stdio_server/sse_app/streamable_http_app/mcp.server.lowlevel usage, no uvicorn/fastapi/starlette in the package, and no code path exposing a Dynamiq workflow or agent over the MCP protocol exists (the dynamiq/cli and dynamiq/clients modules are Dynamiq cloud-platform management/HTTP clients, and the mcp_server variable in dynamiq/nodes/llms/anthropic.py is an unrelated LiteLLM patch pass-through).

#### [HIGH — confirmed] No OpenAPI/Swagger spec import to generate tools

*Who sets the bar: adk, langchain (legacy), crewai (platform-gated)*

ADK turns OpenAPI specs into authenticated toolsets (OpenAPIToolset + RestApiTool with auth schemes) plus enterprise API-catalog integrations; LangChain retains legacy OpenAPI agent toolkits; CrewAI's platform builds typed tools from hosted app action schemas. Dynamiq only offers HttpApiCall, which requires hand-configuring every endpoint (URL, method, params) one at a time. For enterprises with hundreds of internal APIs already described by OpenAPI, 'point at the spec, get tools' is a common evaluation checkbox that Dynamiq fails, despite Pipedream partially covering public SaaS.

> **Verification:** Exhaustive search confirms the claim: case-insensitive grep for 'openapi' and 'swagger' (both ripgrep and raw grep including ignored files) returns zero matches anywhere in the repo, including pyproject.toml and uv.lock, so there is no OpenAPI-parsing dependency or code path. HttpApiCall in dynamiq/nodes/tools/http_api_call.py accepts only manually specified url/method/headers/params/data/files per endpoint with no spec ingestion. The closest capabilities are MCPServer.get_mcp_tools() (auto-generates typed tool nodes from an MCP server's JSON Schema tool list — requires an external OpenAPI-to-MCP bridge to cover this use case) and the Pipedream node (typed tools from Pipedream action schemas, which the claim already concedes); neither ingests OpenAPI/Swagger specs, so 'point at the spec, get tools' is genuinely absent.

#### [MEDIUM — confirmed] No adapters for importing agents or tools from other frameworks

*Who sets the bar: adk, crewai, deepagents*

ADK wraps LangChain tools, CrewAI tools, LlamaIndex retrievers, and compiled LangGraph graphs as first-class citizens; CrewAI has LangGraph and OpenAI-Agents agent adapters plus BaseTool.from_langchain()/to_langchain() and LlamaIndex/Composio/Zapier tool imports; DeepAgents accepts any LangGraph CompiledStateGraph as a subagent. Dynamiq's only generic import path is FunctionTool/@function_tool for plain Python callables — anyone migrating from or coexisting with the LangChain/CrewAI ecosystems must hand-wrap everything. This raises switching costs into Dynamiq and weakens the adoption funnel from the largest ecosystems.

> **Verification:** Exhaustive grep across dynamiq/, examples/, docs/, tests/, and pyproject.toml found zero import statements of langchain, langgraph, crewai, llama_index, or openai-agents in the dynamiq package — only docstring mentions in dynamiq/components/splitters/ (plus one incidental example query string in firecrawl_search.py) — and pyproject.toml declares no dependencies or extras for any of these frameworks. No adapter/converter class exists (no from_langchain/to_langchain, no compat/bridge modules); the only generic wrapping path for foreign Python code is FunctionTool/@function_tool in dynamiq/nodes/tools/function_tool.py, exactly as claimed. Minor mitigations that fall short of refutation: dynamiq/nodes/tools/mcp.py imports tools from any MCP server (protocol-level, not framework-object interop) and Composio integration exists only as a hand-written custom Node in examples/, which itself illustrates the hand-wrapping burden the claim describes.

#### [MEDIUM — partial] No standardized agent-to-UI protocol (AG-UI, A2UI, or a native generative-UI event protocol)

*Who sets the bar: crewai (A2UI over A2A), langgraph (native generative-UI protocol)*

CrewAI implements the a2ui.org generative-UI extension over A2A (validated UI component payloads), and LangGraph ships its own typed UI-message protocol (push_ui_message + useStream React hook) for building agent frontends; literal AG-UI is absent from all six frameworks' repos. Dynamiq offers only raw streaming callbacks with FastAPI/WebSocket/Streamlit/Chainlit examples — no typed UI event contract a frontend team can build against. As embedded agent UIs become a standard deliverable, frameworks with a UI protocol integrate faster with copilot-style frontends.

> **Verification:** The literal greps check out (zero matches for ag-ui/agui/a2ui/ui_message/push_ui repo-wide) and Dynamiq has no UI-component protocol comparable to a2ui or LangGraph's push_ui_message — no renderable-component payload schema, no UI channel, and frontend apps live only under examples/ (FastAPI WebSocket, SSE, Streamlit, Chainlit). However, the sub-claim "no typed UI-event message schema ... only raw streaming events" is overstated: the core package ships Pydantic-typed streaming event contracts frontends build against — StreamingEventMessage envelope plus typed agent reasoning/tool-input-delta/tool-result payloads in types/streaming.py, typed approval events with mutable_data_params in types/feedback.py, and an explicitly UI-directed HFStreamingOutputEventMessage whose is_browser_takeover flag is documented "so a chat UI can render an interactive browser session instead of a plain text prompt". So the generative-UI/component-protocol gap is real, but a limited typed UI-event contract does exist in-package, making the claim partially rather than fully accurate.

### Low-severity (unverified)

- **No chat-channel or telephony integrations in OSS** — DeepAgents ships WhatsApp and Telegram channel adapters (experimental Talon runtime) and ADK has a Slack runner; Dynamiq ships no Slack/Teams/WhatsApp/Telegram/Twilio connectors — only a Chainlit example app. Most competitors are also thin here, so this is a nice-to-have rather than a decisive gap, but 'deploy the agent to Slack' is a frequent first-mile request that currently requires custom glue code with Dynamiq.

### Dynamiq strengths on this dimension

- MCP client engineering quality: the _SchemaModelBuilder in dynamiq/nodes/tools/mcp.py converts arbitrary MCP JSON Schemas (including $defs and recursive references) into typed pydantic input models, with structuredContent-aware result handling — more robust schema fidelity than typical ad-hoc conversions, and agents auto-expand MCPServer instances into tools transparently.
- Pipedream integration (dynamiq/nodes/tools/pipedream.py + PipedreamOAuth2 connection) gives one-node OAuth access to thousands of SaaS app actions — a breadth-of-integration play no competitor matches in open source (ADK's equivalent catalogs are gated on Google Cloud services; CrewAI's on its commercial platform).
- Everything Dynamiq offers on this dimension is fully in the OSS package with no platform gating — unlike LangGraph, whose MCP/A2A server exposure lives in a closed-source server image, and unlike LangChain, whose MCP support requires a separate package and whose protocol servers are commercial platform features.
- Portable declarative workflow format: full YAML load/dump round-trip (dynamiq/serializers/) with secret masking makes entire workflows (not just single agents) exchangeable artifacts consumable by code and the Dynamiq platform UI — LangGraph has no agent-definition export format at all.

### At parity

- MCP client support is roughly at par with the best (ADK, CrewAI, DeepAgents): three transports (stdio/SSE/streamable-HTTP), include/exclude tool filtering, and agent-level MCP server configs — and ahead of the LangChain/LangGraph monorepos, which contain no MCP client code at all (it lives in the separate langchain-mcp-adapters package).
- Wrapping arbitrary Python callables as typed tools (FunctionTool/@function_tool) matches the @tool decorators and from_function converters offered by every competitor.
- Generic authenticated REST calling via HttpApiCall is equivalent to the hand-built HTTP tools competitors rely on when no spec importer applies.
- Declarative config/serialization is at par with ADK's YAML agent configs, CrewAI's JSONC/flow-YAML definitions, and LangChain's lc-json format — each framework has a portable definition format of comparable utility.
- Literal AG-UI protocol support is absent from all six frameworks' repos, so on that specific protocol Dynamiq is at par (the gap is the lack of any UI protocol alternative, tracked separately).
- CrewAI and DeepAgents likewise ship no MCP server mode in OSS, so among the pure-OSS libraries Dynamiq's client-only MCP posture matches two of five competitors.
