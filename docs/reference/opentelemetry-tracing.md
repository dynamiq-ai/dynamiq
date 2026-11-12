# OpenTelemetry tracing

Dynamiq emits OpenTelemetry spans for every workflow, flow and node run. The spans follow the OpenTelemetry
Generative AI semantic conventions where a convention exists and use the `dynamiq.*` namespace for everything else.
This page is the reference for that output: a backend that ingests Dynamiq traces and a user who reads them in a
third-party backend can both rely on it.

| | |
|---|---|
| Emitted shape version | `1` |
| Pinned convention | OpenTelemetry semantic conventions **v1.41.1** (`gen_ai.*`, `error.type`, `session.id`, `user.id`) |
| Schema URL on the tracer | `https://opentelemetry.io/schemas/1.41.1` |
| Instrumentation scope | name `dynamiq`, version = the installed `dynamiq` version, attribute `dynamiq.trace.shape_version = 1` |
| Signals | Spans, span events and span links. No metrics and no log records. |

The GenAI conventions are in Development status and have no release after v1.41.1; their repository takes breaking
changes without a version. The emitted shape is therefore pinned to v1.41.1 and versioned by Dynamiq itself. See
[Versioning](#versioning) and [Upcoming upstream changes](#upcoming-upstream-changes).

The run-based `TracingCallbackHandler` and `DynamiqTracingCallbackHandler` are separate from this output and keep
working unchanged. Both can be active for the same run.

## Enabling

Dynamiq calls the OpenTelemetry API on every run. Nothing is recorded until the application installs an
OpenTelemetry SDK `TracerProvider`; until then the calls are no-ops and no node is serialized for tracing.

Dynamiq never sets the global provider and ships no exporter. `pip install "dynamiq[otel]"` adds
`opentelemetry-sdk` and the OTLP/HTTP exporter; the core install depends only on `opentelemetry-api`.

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # reads OTEL_EXPORTER_OTLP_* variables
trace.set_tracer_provider(provider)

workflow.run(input_data={"input": "..."})  # traced
```

The provider is resolved per run, in this order:

1. `RunnableConfig.tracing.tracer_provider`, when set.
2. The global provider returned by `opentelemetry.trace.get_tracer_provider()`.

### Per-run settings

`RunnableConfig.tracing` takes a `TracingConfig` (`dynamiq.types.telemetry`). Every field is optional.

| Field | Type | Default | Effect |
|---|---|---|---|
| `enabled` | `bool` | `True` | `False` emits no Dynamiq spans for this run. |
| `tracer_provider` | `TracerProvider` | `None` | Provider for this run instead of the global one. Not serialized, not saved in checkpoints and not sent to worker processes. |
| `session_id` | `str` | `None` | `gen_ai.conversation.id` and `session.id` on every span of the run. |
| `user_id` | `str` | `None` | `user.id` on every span of the run. |
| `tags` | `list[str]` | `[]` | `dynamiq.tags` on every span of the run. |
| `metadata` | `dict[str, str \| int \| float \| bool]` | `{}` | `dynamiq.metadata.<key>` on every span of the run. |
| `source_id` | `str` | `None` | `dynamiq.source.id` on every span of the run. Names the platform entity the trace belongs to. |
| `content_capture` | `ContentCaptureMode` | `None` | Overrides the content capture environment variables for this run. |
| `binary_content_enabled` | `bool` | `None` | Overrides `DYNAMIQ_TRACING_BINARY_CONTENT_ENABLED`. |
| `node_config_enabled` | `bool` | `None` | Overrides `DYNAMIQ_TRACING_NODE_CONFIG_ENABLED`. |
| `content_max_length` | `int` | `None` | Overrides `DYNAMIQ_TRACING_CONTENT_MAX_LENGTH`. |
| `prompt_compaction_enabled` | `bool` | `None` | Overrides `DYNAMIQ_TRACING_PROMPT_COMPACTION_ENABLED`. |

`RunnableConfig.tracing` is excluded from checkpoints. Pass it again when resuming a run.

### Environment variables

Standard OpenTelemetry variables (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_SERVICE_NAME`,
`OTEL_RESOURCE_ATTRIBUTES`, `OTEL_TRACES_SAMPLER`, `OTEL_SDK_DISABLED`, ...) are read by the OpenTelemetry SDK and
apply as usual. Dynamiq reads these:

| Variable | Values | Default | Effect |
|---|---|---|---|
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` | `NO_CONTENT`, `SPAN_ONLY`, `EVENT_ONLY`, `SPAN_AND_EVENT` | `SPAN_ONLY` | Content capture mode. See [Content capture](#content-capture). |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | the same four values, or `true` / `false` | unset | Older name of the same switch, read when the variable above is unset. |
| `DYNAMIQ_TRACING_ENABLED` | `true`, `false` | `true` | `false` emits no Dynamiq spans in this process. |
| `DYNAMIQ_TRACING_BINARY_CONTENT_ENABLED` | `true`, `false` | `false` | Inline binary message content (base64 images, audio, files) is recorded in full. |
| `DYNAMIQ_TRACING_NODE_CONFIG_ENABLED` | `true`, `false` | `false` | Adds the serialized node configuration as `dynamiq.node.config`. |
| `DYNAMIQ_TRACING_CONTENT_MAX_LENGTH` | positive integer | unset | Truncates each string inside recorded content to this many characters. The JSON structure stays valid. |
| `DYNAMIQ_TRACING_PROMPT_COMPACTION_ENABLED` | `true`, `false` | `false` | A `chat` span that continues the prompt of an earlier `chat` span records only its new messages and a pointer to that span. Only for a backend that follows the pointer. See [Prompt compaction](#prompt-compaction). |

## Trace structure

A run joins the trace that is active when it starts. If the calling code is inside a span, for example an HTTP
server span, the Dynamiq spans are descendants of it. Otherwise the top-level run starts a new trace. A workflow
span is therefore not always the trace root, and one trace can hold several workflow runs. The span of every
top-level run carries `dynamiq.run.is_root`, wherever it is in the trace. See [Root runs](#root-runs).

Parents are resolved in this order:

1. The span of the run named by `parent_run_id`, when that run has a span in this process.
2. The span that is current in the OpenTelemetry context.

While a node executes, its span is the current span, so spans created by other instrumentation inside the node
(HTTP clients, database drivers, MCP) are its children. Workflow and flow spans are not made current: code that
runs between nodes, such as a checkpoint backend, is attached to whatever span was current before the run.

```text
invoke_workflow support-flow            (Workflow)
├── start                               (Input node)
├── invoke_agent support-agent          (Agent node)
│   ├── chat gpt-4o                     (LLM node, loop 1)
│   ├── execute_tool web-search         (tool node)
│   └── chat gpt-4o                     (LLM node, loop 2)
└── end                                 (Output node)
```

### What gets a span

| Run | Span | Notes |
|---|---|---|
| `Workflow` | one span | |
| `Flow` whose parent run is a `Workflow` | none | The workflow span represents it and carries `dynamiq.flow.id` and `dynamiq.flow.run_id`. Its nodes are children of the workflow span. |
| `Flow` run on its own, or by a node (sub-workflow) | one span | Same shape as a workflow span, with `dynamiq.run.type = flow`. |
| Node | one span per run | Includes skipped, canceled, mocked and cached runs. A retried node is one span. |
| Agent loop iteration | none | The LLM and tool runs of all iterations are children of the agent span, in start-time order. |

## Span mapping

`gen_ai.operation.name` is set only on the span types the convention defines. A node whose group has no GenAI
operation gets a plain internal span that carries `dynamiq.*` attributes only.

| Dynamiq entity | Span name | Kind | `gen_ai.operation.name` |
|---|---|---|---|
| `Workflow`; standalone `Flow` | `invoke_workflow {name}` | `INTERNAL` | `invoke_workflow` |
| Node group `agents` (`Agent`, orchestrators, `SubAgentTool`) | `invoke_agent {node.name}` | `INTERNAL` | `invoke_agent` |
| Node group `llms` | `chat {model}` | `CLIENT` | `chat` |
| Node group `embedders` | `embeddings {model}` | `CLIENT` | `embeddings` |
| Node group `retrievers` (vector store retriever nodes) | `retrieval {data source}`, or `retrieval` | `CLIENT` | `retrieval` |
| Node group `tools` (includes MCP tools, `VectorStoreRetriever`, knowledge base search, Python and HTTP tools) | `execute_tool {node.name}` | `INTERNAL` | `execute_tool` |
| Every other group: `operators` (`Map`, `Choice`, `Pass`, sub-workflow), `utils` (`Input`, `Output`), `rankers`, `converters`, `splitters`, `writers`, `validators`, `transformers`, `extractors`, `filters`, `parsers`, `detectors`, `audio`, `images` | `{node.name}` | `INTERNAL` | not set |

`{model}` is the model name without the provider prefix Dynamiq adds (`openai/gpt-4o` is recorded as `gpt-4o`).

Nesting follows the runs. A `VectorStoreRetriever` tool span contains an `embeddings` span and a `retrieval` span;
a `Map` span contains one child per item; an agent used as a tool appears as an `invoke_agent` span under the
calling agent.

## Attributes

Requirement levels: **Required** is always present; **Conditional** is present when the stated condition holds;
**Recommended** is present when the value is available; **Content** is present only when
[content capture](#content-capture) is on.

### Every Dynamiq span

| Attribute | Type | Level | Source |
|---|---|---|---|
| `dynamiq.run.id` | string | Required | Run id (UUID) of the workflow, flow or node run. The `run_id` callbacks receive. |
| `dynamiq.run.type` | string | Required | `workflow`, `flow` or `node`. |
| `dynamiq.run.is_root` | boolean | Conditional: the run has no Dynamiq parent run | `true`. Set when the span starts, so a sampler sees it. Absent on every other span, never `false`. See [Root runs](#root-runs). |
| `dynamiq.run.status` | string | Required at end | `succeeded`, `failed`, `skipped`, `canceled` or `paused`. See [Status](#status-and-errors). |
| `gen_ai.conversation.id` | string | Conditional: a session id is known | `TracingConfig.session_id`, else the `session_id` input of the nearest enclosing agent. |
| `session.id` | string | Conditional: same | Same value as `gen_ai.conversation.id`. |
| `user.id` | string | Conditional: a user id is known | `TracingConfig.user_id`, else the `user_id` input of the nearest enclosing agent. |
| `dynamiq.source.id` | string | Conditional: set | `TracingConfig.source_id`. |
| `dynamiq.tags` | string[] | Conditional: not empty | `TracingConfig.tags`. |
| `dynamiq.metadata.<key>` | string, int, double or boolean | Conditional: set | `TracingConfig.metadata`. |
| `error.type` | string | Conditional: the run failed or was canceled | Class name of the exception, for example `RateLimitError`; `CanceledException` for a canceled run. |

Run-level values (session, user, source, tags, metadata) are repeated on every span of the run, because some
backends store spans without joining them to a trace-level record.

#### Root runs

A run is a root when no Dynamiq run started it: it was given no `parent_run_id`. A workflow run, a flow run on its
own and a node run on its own each have exactly one span with `dynamiq.run.is_root = true`, and no span below them
has it. The nodes of a flow, the flow of a sub-workflow, the runs below an agent, `Map` items and a node that a
`ProcessExecutor` worker runs all have a parent run and are never roots.

The attribute describes the run, not the place of its span in the trace:

- A root whose caller is inside a span, for example an HTTP server span, has a parent span and is still a root.
- A trace holds one root for every top-level run in it.
- Code inside a node that runs another node without passing `parent_run_id` starts a root. Its span is a child of
  the calling node's span, because that span is current, and callbacks also see a run without a parent.
- The fallback LLM of an LLM that runs on its own is a root of its own, as the failed primary run is.

A backend receives the spans of a trace children first, so a span whose parent has not arrived yet looks like a
top-most one. `dynamiq.run.is_root` is what tells a top-most Dynamiq run from a run whose parent span is late.

### Workflow span (`invoke_workflow`)

| Attribute | Type | Level | Source |
|---|---|---|---|
| `gen_ai.operation.name` | string | Required | `invoke_workflow` |
| `gen_ai.workflow.name` | string | Required | `Workflow.name`; `Flow.name` for a standalone flow. |
| `dynamiq.workflow.id` | string | Conditional: workflow span | `Workflow.id` |
| `dynamiq.workflow.version` | string | Conditional: set | `Workflow.version` |
| `dynamiq.flow.id` | string | Required | `Flow.id` |
| `dynamiq.flow.run_id` | string | Conditional: workflow span | Run id of the workflow's flow run, which has no span of its own. |
| `dynamiq.config.run_id` | string | Conditional: set | `RunnableConfig.run_id`, the `wf_run_id` of streaming events and checkpoints. |
| `dynamiq.input` | string (JSON) | Content | Input of the run. |
| `dynamiq.output` | string (JSON) | Content | Output of the flow's terminal nodes (nodes no other node depends on): the output itself when there is one such node, otherwise an object keyed by node id. |
| `dynamiq.failed_nodes` | string (JSON) | Conditional: the run failed because nodes failed | `[{"id", "name", "error_message"}]` |
| `dynamiq.checkpoint.id` | string | Conditional: the run paused | Id of the checkpoint to resume from. |
| `dynamiq.resume.checkpoint_id` | string | Conditional: the run resumed another | Id of the checkpoint the run started from. |
| `dynamiq.resume.original_run_id` | string | Conditional: the run resumed another | Flow run id of the run that wrote that checkpoint. |

### Node spans (all groups)

| Attribute | Type | Level | Source |
|---|---|---|---|
| `dynamiq.node.id` | string | Required | `Node.id` |
| `dynamiq.node.name` | string | Required | `Node.name` |
| `dynamiq.node.type` | string | Required | `Node.type`, for example `dynamiq.nodes.llms.OpenAI` |
| `dynamiq.node.group` | string | Required | `Node.group`, for example `llms` |
| `dynamiq.node.model` | string | Conditional: the node has a `model` | The `model` field as configured, with its provider prefix. |
| `dynamiq.node.depends` | string[] | Conditional: not empty | Ids of the nodes this node depends on in the flow definition. |
| `dynamiq.run.depends` | string[] | Conditional: not empty | Ids of the nodes whose runs preceded this run. Inside an agent this orders LLM and tool runs. |
| `dynamiq.node.is_mocked` | boolean | Conditional: `true` | The output came from a mock. |
| `dynamiq.node.is_output_from_cache` | boolean | Conditional: `true` | The output came from the cache. |
| `dynamiq.node.config` | string (JSON) | Conditional: node config capture and content capture are on | `Node.to_dict(for_tracing=True)` without `prompt.messages`. |
| `dynamiq.input` | string (JSON) | Content; not on `execute_tool` spans | Validated input of the run. |
| `dynamiq.output` | string (JSON) | Content; not on `execute_tool` spans | Output of the run after the output transformer. |
| `dynamiq.skip` | string (JSON) | Conditional: the run was skipped | `{"failed_dependency": {...}}`. With content capture on it is the full skip data. With it off the dependency keeps only its identifiers: node `id`, `name` and `type`, and the option `id` or `name` when there is one. Conditions and other payload are omitted. |
| `dynamiq.human_feedback` | string | Content; a human rejected the run | Feedback text. |
| `dynamiq.tool.data` | string (JSON) | Conditional: reported by the node | For example `{"tool_session_id", "tool_session_host"}` of a sandbox tool. |
| `dynamiq.usage.*` | | Conditional | See [Usage and cost](#usage-and-cost). |

### Inference span (`chat`)

| Attribute | Type | Level | Source |
|---|---|---|---|
| `gen_ai.operation.name` | string | Required | `chat` |
| `gen_ai.provider.name` | string | Required | From the LLM node class. Convention values where one exists (`openai`, `anthropic`, `aws.bedrock`, `azure.ai.openai`, `gcp.gemini`, `gcp.vertex_ai`, `cohere`, `groq`, `mistral_ai`, `deepseek`, `perplexity`, `x_ai`, `ibm.watsonx.ai`); otherwise the provider prefix of the model (`openrouter`, `together_ai`, `ollama`, ...). |
| `gen_ai.system` | string | Required | Same value as `gen_ai.provider.name`. See [Aliases](#aliases). |
| `gen_ai.request.model` | string | Required | Model name without the provider prefix. |
| `gen_ai.request.temperature`, `.max_tokens`, `.top_p`, `.frequency_penalty`, `.presence_penalty` | double / int | Recommended | LLM node fields that are set. |
| `gen_ai.request.stop_sequences` | string[] | Recommended | `stop` |
| `gen_ai.request.seed` | int | Conditional: set | `seed` |
| `gen_ai.request.stream` | boolean | Conditional: streaming | `true` when the node streams. |
| `gen_ai.output.type` | string | Conditional: a response format is requested | `json` |
| `gen_ai.response.id` | string | Recommended | Provider response id. |
| `gen_ai.response.model` | string | Recommended | Model named in the response. |
| `gen_ai.response.finish_reasons` | string[] | Recommended | One per choice: `stop`, `length`, `content_filter`, `tool_call`, or the provider's value. |
| `gen_ai.usage.input_tokens` | int | Recommended | Prompt tokens, cached tokens included. |
| `gen_ai.usage.output_tokens` | int | Recommended | Completion tokens. |
| `gen_ai.usage.cache_read.input_tokens` | int | Conditional: reported | Tokens read from the provider cache. |
| `gen_ai.usage.cache_creation.input_tokens` | int | Conditional: reported | Tokens written to the provider cache. |
| `gen_ai.usage.cache_write.input_tokens` | int | Conditional: reported | Same value. See [Aliases](#aliases). |
| `gen_ai.input.messages` | string (JSON) | Content | Every message of the request. See [Messages](#messages). With [prompt compaction](#prompt-compaction) on, the messages after the ones an earlier span already holds. |
| `gen_ai.output.messages` | string (JSON) | Content | The response message. |
| `gen_ai.tool.definitions` | string (JSON) | Content; tools were passed | `[{"type": "function", "name", "description", "parameters"}]` |
| `dynamiq.llm.is_fallback` | boolean | Conditional: `true` | The run is the fallback LLM of a failed primary run. The failed primary run is a sibling span. |
| `dynamiq.prompt.messages_base.span_id` | string | Conditional: prompt compaction is on and the prompt continues the one of an earlier span | Span id of that span, as 16 lowercase hex digits. See [Prompt compaction](#prompt-compaction). |
| `dynamiq.prompt.messages_base.count` | int | Conditional: same | Number of messages at the start of the prompt that are left out because that span holds them. |

`dynamiq.input` on an inference span holds the node input (prompt template parameters), not the messages.
System messages are part of `gen_ai.input.messages`; `gen_ai.system_instructions` is not emitted.

### Embeddings span (`embeddings`)

| Attribute | Type | Level | Source |
|---|---|---|---|
| `gen_ai.operation.name` | string | Required | `embeddings` |
| `gen_ai.provider.name`, `gen_ai.system` | string | Required | As on inference spans. |
| `gen_ai.request.model` | string | Required | Model name without the provider prefix. |
| `gen_ai.response.model` | string | Conditional: the provider reports it | `meta.model` of the embedder output. |
| `gen_ai.usage.input_tokens` | int | Conditional: the provider reports it | `meta.usage.prompt_tokens` of the embedder output. |
| `gen_ai.embeddings.dimension.count` | int | Recommended | The node's `dimensions`, else the length of the returned vector. |

### Retrieval span (`retrieval`)

| Attribute | Type | Level | Source |
|---|---|---|---|
| `gen_ai.operation.name` | string | Required | `retrieval` |
| `gen_ai.data_source.id` | string | Conditional: known | The index, collection or table name of the vector store node. |
| `gen_ai.request.top_k` | double | Recommended | `top_k` of the input, else of the node. |
| `gen_ai.retrieval.query.text` | string | Content; the input has a text query | `query` |
| `gen_ai.retrieval.documents` | string (JSON) | Content | `[{"id", "score"}]` for the returned documents. Their content is in `dynamiq.output`. |

Each entry of `gen_ai.retrieval.documents` has a string `id`; a document without an id is given its position in the
list, as a string. `score` is present when the store returned one and omitted otherwise. A score is not invented,
which differs from the convention; see [Differences from the convention](#differences-from-the-convention).

### Tool span (`execute_tool`)

| Attribute | Type | Level | Source |
|---|---|---|---|
| `gen_ai.operation.name` | string | Required | `execute_tool` |
| `gen_ai.tool.name` | string | Required | `Node.name` |
| `gen_ai.tool.type` | string | Required | `function` |
| `gen_ai.tool.call.arguments` | string (JSON) | Content | Validated input of the run. Holds what `dynamiq.input` holds on other spans. |
| `gen_ai.tool.call.result` | string (JSON) | Content | Output of the run. Holds what `dynamiq.output` holds on other spans. |

### Agent span (`invoke_agent`)

| Attribute | Type | Level | Source |
|---|---|---|---|
| `gen_ai.operation.name` | string | Required | `invoke_agent` |
| `gen_ai.provider.name` | string | Conditional: the agent has an LLM | Provider of the agent's LLM. |
| `gen_ai.request.model` | string | Conditional: the agent has an LLM | Model of the agent's LLM, without the provider prefix. |
| `gen_ai.agent.name` | string | Required | `Node.name` |
| `gen_ai.agent.id` | string | Required | `Node.id` |
| `gen_ai.agent.description` | string | Conditional: set | The agent's `description`. |
| `gen_ai.input.messages` | string (JSON) | Content | One user message holding the agent's `input` text. |
| `gen_ai.output.messages` | string (JSON) | Content; the run succeeded | One assistant message holding the final answer. |

Token usage is not recorded on agent or workflow spans. Sum the `chat` and `embeddings` spans below them.

### Usage and cost

| Attribute | Type | Emitted on | Source |
|---|---|---|---|
| `dynamiq.usage.total_tokens` | int | Spans that report usage | `total_tokens` |
| `dynamiq.usage.input_cost_usd` | double | Spans that report a cost | Cost of the prompt tokens in USD. |
| `dynamiq.usage.output_cost_usd` | double | Spans that report a cost | Cost of the completion tokens in USD. |
| `dynamiq.usage.total_cost_usd` | double | Spans that report a cost | Total cost in USD. |
| `dynamiq.usage.input_tokens`, `dynamiq.usage.output_tokens` | int | Spans other than `chat` and `embeddings` that report usage | A tool or detector that calls a model without running an LLM node. `gen_ai.usage.*` stays reserved for inference and embeddings spans, so that a backend never counts the same tokens twice. |

The convention has no cost attribute. An upstream proposal adds `gen_ai.usage.cost.*`; Dynamiq will emit it once
it is part of a released convention.

## Span events and links

| Event name | Emitted on | Attributes |
|---|---|---|
| `exception` | A failed run | `exception.type`, and with content capture on `exception.message` and `exception.stacktrace`. |
| `dynamiq.node.execution` | A node run with a failed attempt, once per attempt | `dynamiq.execution.id` (string), `dynamiq.execution.attempt` (int, from 1), `dynamiq.execution.status` (`succeeded` or `failed`), `dynamiq.execution.start_time_unix_nano` (int), and for a failed attempt `error.type` and, with content capture on, `exception.message`. The event time is the end of the attempt. |
| `dynamiq.agent.hook` | An agent span, once per hook event | `dynamiq.hook.name`, `dynamiq.hook.type`, `dynamiq.hook.point`, `dynamiq.hook.decision`, and with content capture on `dynamiq.hook.details` (JSON of the remaining event fields). |

A node run that succeeds on its first attempt has no `dynamiq.node.execution` event: the span itself is the attempt.
A mocked run and a cache hit have no attempt.

| Link | From | To | Attributes |
|---|---|---|---|
| Resume | The workflow or flow span of a resumed run | The workflow or flow span of the run that paused | `dynamiq.link.type = resumed_from` |

## Status and errors

| Outcome | `dynamiq.run.status` | Span status | Other |
|---|---|---|---|
| Succeeded | `succeeded` | `UNSET` | |
| Failed | `failed` | `ERROR`, description = the error message when content capture is on | `error.type`, `exception` event |
| Canceled | `canceled` | `ERROR` | `error.type = CanceledException` |
| Skipped (failed dependency, unmet condition, rejected approval) | `skipped` | `UNSET` | `dynamiq.skip`, `dynamiq.human_feedback` |
| Paused (waiting for input or for a time, with a checkpoint) | `paused` | `UNSET` | `dynamiq.checkpoint.id` on the workflow span |

A retried node is one span. Its status is the status of the last attempt, and the failed attempts are
`dynamiq.node.execution` events.

A tool that fails inside an agent has status `failed` even when the agent recovers and continues. The agent span
reports the agent's own outcome.

A span never stays open. If a run ends without reporting an outcome, for example because the task was canceled by
the event loop, the span is ended as `canceled` or `failed` according to the exception that ended it.

## Content capture

Content is everything that can hold user data or model output: messages, tool arguments and results, tool
definitions, retrieval queries, node inputs and outputs, node configuration, human feedback, hook details, error
messages and stack traces.

Content capture is **on by default** (`SPAN_ONLY`). This differs from the convention, whose default is off.

| Mode | Effect |
|---|---|
| `SPAN_ONLY`, `SPAN_AND_EVENT`, `true` | Content is recorded on span attributes. |
| `NO_CONTENT`, `EVENT_ONLY`, `false` | No content is recorded, except the identifiers in `dynamiq.skip`. Dynamiq emits no log-record events, so `EVENT_ONLY` records nothing. |

Rules that hold in every mode:

- `dynamiq.skip` is the one content attribute that is not removed by `NO_CONTENT`: it keeps the identifiers of the
  failed dependency (node `id`, `name` and `type`, and the option `id` or `name` when there is one) and omits
  conditions and other payload. With content capture on it is the full skip data.
- Content attributes are JSON strings. Structured attribute values are not used, because several backends accept
  only the string form.
- Values pass through the same redaction as run-based traces: `format_value(for_tracing=True)`,
  `to_dict(for_tracing=True)` and the `TRACING_REDACTED_KEYS` list. A value redacted there is `"***"` here. Secrets
  of connections and sandboxes never reach a span.
- Lists longer than 50 items are cut to 50, and embedding vectors to 20 values, as in run-based traces.
- Inline binary content is replaced by a placeholder unless binary capture is on. See [Messages](#messages).
- `DYNAMIQ_TRACING_CONTENT_MAX_LENGTH` truncates string values inside the JSON; it never cuts the JSON itself.
- The OpenTelemetry SDK limit `OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT`, when set, cuts the JSON string and makes
  it invalid. Prefer the Dynamiq limit.

### Messages

`gen_ai.input.messages` and `gen_ai.output.messages` follow the convention's message schema:
`[{"role", "parts": [...], "name"?}]`, with `finish_reason` on each output message.

| Dynamiq message content | Part |
|---|---|
| Text (`content` string, or `{"type": "text", "text"}`) | `{"type": "text", "content"}` |
| Tool call of an assistant message (`tool_calls[]`) | `{"type": "tool_call", "id", "name", "arguments"}`; `arguments` is the parsed object, or the raw string when it is not valid JSON |
| Tool message (`role: tool`, `tool_call_id`) | `{"type": "tool_call_response", "id", "response"}` |
| Image by URL (`{"type": "image_url", "image_url": {"url": "https://..."}}`) | `{"type": "uri", "modality": "image", "uri"}` |
| Inline image (`data:` URL) | `{"type": "blob", "modality": "image", "mime_type", "content"}`; `content` is the base64 data when binary capture is on, otherwise an empty string with `"size"` set to the length of the omitted data |
| File by id | `{"type": "file", "modality": "document", "file_id"}` |
| Any other part | The part unchanged, with its own `type` |

Message fields that are not part of the request (`static`, `metadata`) are dropped. The original message list can
be rebuilt from the parts by inverting the table.

### Prompt compaction

An agent loop sends its whole conversation on every model call, so each `chat` span carries every message of its
request, and the size of a trace grows with the square of the number of loops. Prompt compaction records every
message once. It is **off by default**; `TracingConfig.prompt_compaction_enabled` or
`DYNAMIQ_TRACING_PROMPT_COMPACTION_ENABLED=true` turns it on for a run tree.

With it on, a `chat` span is compared with the previous `chat` span of the same LLM node (`dynamiq.node.id`) under
the same parent run. When its prompt starts with messages of that span's prompt, it records in
`gen_ai.input.messages` only the messages after the shared start, and names the span that holds the others:

| Attribute | Type | Value |
|---|---|---|
| `dynamiq.prompt.messages_base.span_id` | string | Span id of the earlier `chat` span, as 16 lowercase hex digits. That span is in the same trace and has the same parent span. |
| `dynamiq.prompt.messages_base.count` | int | Number of messages the two prompts share at their start. Never 0. |

The whole prompt of a span is the first `count` messages of the whole prompt of the span named by `span_id`,
followed by the span's own messages. The named span can be compacted too, so the rule is applied again to it. A
`chat` span without the two attributes holds its whole prompt.

```text
invoke_agent support-agent
├── chat  span a1  gen_ai.input.messages = [system, user]
├── execute_tool web-search
├── chat  span a2  gen_ai.input.messages = [assistant, tool]   messages_base = {span_id: a1, count: 2}
├── execute_tool web-search
└── chat  span a3  gen_ai.input.messages = [assistant, tool]   messages_base = {span_id: a2, count: 4}
```

The rules are those of `TracingCallbackHandler(compact_llm_prompts_enabled=True)`:

- The previous span is the last `chat` span of that LLM node under that parent run whose prompt was recorded
  before this one reported its own. Two runs never name each other, also when runs of one parent overlap.
- Messages are compared as they are sent, before redaction and truncation. The shared start ends at the first
  message that differs, so a prompt whose history was rewritten keeps only what still matches, and `count` can be
  smaller than the number of messages of the named span. A prompt equal to the previous one records `[]`.
- These spans hold their whole prompt and have neither attribute: a run without a parent run, such as an LLM node
  run on its own; the first run of an LLM node under a parent run; a run whose prompt shares no first message with
  the previous one.
- A retried run keeps what its first attempt recorded.

Three more rules follow from spans being exported one by one:

- A run whose parent run has no span in the same process is not compacted: the previous prompt is kept with the
  span of the parent run and goes with it. This is the node a `ProcessExecutor` worker runs; the runs below that
  node are compacted as usual.
- A span that is not sampled, or whose prompt could not be recorded, is never named by another span.
- With content capture off no messages are recorded, and neither attribute is emitted.

System messages are part of `gen_ai.input.messages`, so they are recorded once per chain like every other message.
`gen_ai.tool.definitions` is not compacted: every `chat` span that passed tools carries all of them.

A backend has to follow the pointer to show a prompt. No third-party backend does: it shows a compacted span with
its new messages only, and anything that reads `gen_ai.input.messages` as the whole request, such as prompt
replay, evaluation or token counting, gets a part of it. That is why the mode is off by default. A collector or a
backend that drops single spans of a trace also breaks the chains that pass through them. Turn the mode on only
when the receiving backend expands `dynamiq.prompt.messages_base.*` and keeps traces whole.

## Ids and correlation

- Trace and span ids are generated by the OpenTelemetry SDK. Dynamiq does not choose them. Code that needs the
  trace id of a run before it starts creates the surrounding span itself and reads the id from it.
- `dynamiq.run.id` is the run id that callbacks and streaming events use. `dynamiq.config.run_id` on the workflow
  span is `RunnableConfig.run_id`.
- `dynamiq.node.id` identifies the node in the workflow definition. One node has many runs: per `Map` item, per
  agent loop, per retry of the workflow.
- `gen_ai.tool.call.id` is not emitted: the id the model gives a tool call does not reach the tool run. The id is
  present in the `tool_call` and `tool_call_response` parts of the surrounding `chat` spans.

## Pause and resume

A run that pauses ends its spans with `dynamiq.run.status = paused`. The checkpoint stores the W3C `traceparent`
of the workflow span under `metadata["traceparent"]`.

A resumed run is a new run with new spans. It joins the trace that is current when it starts, or starts a new one.
Its workflow span carries `dynamiq.resume.checkpoint_id` and `dynamiq.resume.original_run_id`, and a
`resumed_from` link to the workflow span of the paused run when the checkpoint holds its `traceparent`. Nodes that
completed before the pause do not run again and have no span in the resumed run.

## Threads, async code and processes

- Thread pools and asyncio tasks used by flows, agents and `Map` carry the OpenTelemetry context, and Dynamiq
  resolves parents by run id, so spans nest correctly under any of them.
- A node that times out keeps running in its thread. Spans it starts after the timeout are children of its span,
  which has already ended with status `failed`.
- `ProcessExecutor` passes the parent run's trace context to the worker. A worker records spans only if a
  `TracerProvider` is configured in that process; `TracingConfig.tracer_provider` is not sent to workers. The node
  a worker runs names the flow run that submitted it, so it is not a [root](#root-runs).
- MCP tools send the trace context to the server in the request `_meta` (`traceparent`, `tracestate`, `baggage`).

### Other instrumentation

Dynamiq owns the `chat` and `embeddings` spans. Instrumentation of the model client underneath (LiteLLM's `otel`
callback, OpenAI or Anthropic instrumentors) adds a second span for the same call, or writes onto the Dynamiq span.
Dynamiq logs a warning when LiteLLM's OpenTelemetry callback is enabled. Leave it off, or accept duplicate
inference spans.

## Rebuilding a run

A backend that needs the run-based shape of `TracingCallbackHandler` can derive it from the spans of shape
version 1. `dynamiq.telemetry.rebuild.rebuild_runs(spans)` applies these rules to finished spans and returns
records in the shape of `Run.to_dict()`; `spans_from_otlp_json()` reads spans stored as OTLP/JSON for it. The
golden fixtures in `tests/integration/telemetry/fixtures/` hold traces with the runs a handler recorded for them,
for an implementation in another language to be tested against.

A Dynamiq span is a span with `dynamiq.run.id`; every other span is ignored and yields no run.

| Run field | Source |
|---|---|
| `id` | `dynamiq.run.id` |
| `type` | `dynamiq.run.type` |
| `name` | `dynamiq.node.name` for nodes; `Workflow` or `Flow` for the others |
| `trace_id` | Trace id of the span, as a lowercase UUID string (`0af76519-16cd-43dd-8448-eb211c80319c`). The handler records its own `trace_id` instead. |
| `parent_run_id` | None when `dynamiq.run.is_root` is `true`. Otherwise `dynamiq.run.id` of the nearest Dynamiq span among the span's ancestors, skipping spans of other instrumentation; none when no ancestor in the trace is a Dynamiq span. |
| `start_time`, `end_time` | Span start and end |
| `status` | `dynamiq.run.status`; `paused` is recorded as `failed` |
| `input`, `output` | `dynamiq.input`, `dynamiq.output`; on `execute_tool` spans `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result`. The `output` of a workflow or flow run is the exception: see [Workflow and flow output](#workflow-and-flow-output). |
| `error.message`, `error.traceback` | `exception.message` and `exception.stacktrace` of the `exception` event. None without the event. |
| `executions` | One per `dynamiq.node.execution` event: `id` = `dynamiq.execution.id`, `start_time` = `dynamiq.execution.start_time_unix_nano`, `end_time` = the event time, `status` = `dynamiq.execution.status`, `error` = `{message: exception.message, traceback: none}` for a failed attempt. With no event: for a succeeded run that is neither mocked nor cached, one succeeded execution covering the span, without an id; for a paused run, one execution with the span's start time and no end time and no status, which is how the handler records an attempt that paused; otherwise none. |
| `tags`, `session_id`, `source_id` | `dynamiq.tags`, `gen_ai.conversation.id`, `dynamiq.source.id` |
| `metadata.node` | `dynamiq.node.config` when the span has it; otherwise `{id, name, type, group, model, depends}` from `dynamiq.node.*`, without the keys whose attribute is absent (`depends` is always present) |
| `metadata.node.prompt.messages` | `gen_ai.input.messages` of a `chat` span, converted back with the [Messages](#messages) table |
| `metadata.node.prompt.messages_base` | `{run_id, count}`, where `run_id` is the `dynamiq.run.id` of the span named by `dynamiq.prompt.messages_base.span_id` and `count` is `dynamiq.prompt.messages_base.count`. See below for a span with neither attribute. |
| `metadata.node.depends`, `metadata.run_depends` | `dynamiq.node.depends`, `dynamiq.run.depends` (node ids only) |
| `metadata.usage` | `prompt_tokens` = `gen_ai.usage.input_tokens`, `completion_tokens` = `gen_ai.usage.output_tokens`, `total_tokens` = `dynamiq.usage.total_tokens`, cache tokens from `gen_ai.usage.cache_*`, costs from `dynamiq.usage.*_cost_usd`; on other spans `dynamiq.usage.*`. Only the keys whose attribute is present. |
| `metadata.tool_data`, `metadata.skip`, `metadata.human_feedback`, `metadata.failed_nodes` | `dynamiq.tool.data`, `dynamiq.skip`, `dynamiq.human_feedback`, `dynamiq.failed_nodes`. With content capture off `metadata.skip` holds only the identifiers of the failed dependency. |
| `metadata.hooks` | One per `dynamiq.agent.hook` event: `{hook, type, point, decision}` from `dynamiq.hook.*`, merged with the fields of `dynamiq.hook.details` |
| `metadata.is_mocked`, `metadata.is_output_from_cache` | The attributes of the same name, `false` when absent. As the handler records them: both on a succeeded run, `is_mocked` only on a failed or paused run, neither on a skipped or canceled run. |
| `metadata.workflow` | On a workflow run: `{id: dynamiq.workflow.id, version: dynamiq.workflow.version}` |
| `metadata.flow` | On a flow run: `{id: dynamiq.flow.id}` |
| `metadata.<key>` | `dynamiq.metadata.<key>`, for the run-level metadata a handler adds to every run |
| Host packages | Instrumentation scope name and version. Not part of a run record. |

The flow run of a workflow has no span and is not rebuilt: the nodes of that flow name the workflow run as their
parent, where the handler names the flow run. Dependency options and conditions (`Choice` branches) are present
only in `dynamiq.node.config`.

A `chat` span with neither `dynamiq.prompt.messages_base.*` attribute holds its whole prompt. A handler with
`compact_llm_prompts_enabled` records `messages_base` as none for such a run when it has a parent run, and leaves
the key out for a run without one; a handler without the setting never writes the key. The spans do not say which
handler is being reproduced, so the rebuild is told (`rebuild_runs(spans, prompt_compaction_enabled=True)`). With
prompt compaction on, `metadata.node.prompt.messages` and `messages_base` are what a compacting handler records: the
same messages are left out and the same run is named.

Some of what the handler records is on no span, and a rebuild leaves it out rather than make it up:

- `error` of a paused run (the handler records the `RunPausedException`) and of a canceled run (the handler records
  a message of its own); the `traceback` of a failed attempt in `executions`.
- The text of `error.traceback`: the handler formats the exception being handled when it is called, the span the
  exception's own traceback.
- The `id` of an execution that is the span itself.
- Message fields that are not part of the request (`static`, `metadata`), the `detail` of an image, and binary
  content that was not captured. Tool call `arguments` are rebuilt as JSON text of the recorded object, which can
  differ in spacing from the text the model sent.
- `metadata.node` beyond its identifiers when node configuration capture is off, and dependency options and
  conditions then.
- Usage fields the provider did not report (a cache read of 0, a cache write of none).
- Everything that is content when content capture is off: `input`, `output`, error messages, prompts,
  `tool_data`, `human_feedback`, hook details, the `error_message` of `failed_nodes`, and all of `dynamiq.skip` but
  the identifiers.

### Workflow and flow output

`dynamiq.output` of a workflow span, or of the span of a flow run on its own, holds the output of the terminal
nodes. The handler's workflow and flow runs hold a map of every node the flow ran: `{node id: {"status", "input",
"output"}}`, plus `"error"` for a node that failed. Rebuild it from the node spans whose parent run is that workflow
or flow run, not from `dynamiq.output`:

| Entry field | Source |
|---|---|
| key | `dynamiq.node.id` |
| `input`, `output` | The `input` and `output` of the node's rebuilt run |
| `status` | `dynamiq.run.status`, mapped as below |
| `error.type`, `error.message` | `exception.type` and `exception.message` of the `exception` event, when there is one |

The flow reports the statuses of `RunnableStatus`, which differ from the values of `dynamiq.run.status`:

| `dynamiq.run.status` | Status in the map |
|---|---|
| `succeeded` | `success` |
| `failed` | `failure` |
| `paused` | `failure` |
| `skipped` | `skip` |
| `canceled` | `canceled` |

The handler sets the `output` of a workflow or flow run only when it succeeded; for any other outcome it is empty.
Three parts of the map are on no span, and a rebuild omits them: a node that never ran, which the handler lists with
status `undefined` and no input or output; `error.recoverable` and `error.failed_nodes`; and the `error` of a
skipped node (for example `NodeConditionSkippedException`), which has no `exception` event.

## Differences from the convention

| Topic | Convention v1.41.1 | Dynamiq | Reason |
|---|---|---|---|
| Content capture default | Off | On | Traces without prompts and outputs are of little use for debugging agents. One variable turns it off. |
| Exceptions | `gen_ai.client.operation.exception` log record | `exception` span event | Only spans are emitted in shape version 1. |
| `gen_ai.system_instructions` | Separate attribute for instructions passed outside the chat history | Not emitted | Dynamiq sends system messages inside the chat history, where the convention says to record them. |
| `gen_ai.input.messages` with prompt compaction on | The chat history of the request | The messages an earlier span does not hold, with `dynamiq.prompt.messages_base.*` | Opt-in, for a backend that follows the pointer. Off by default, and then the attribute is the whole history. |
| `gen_ai.tool.description` | Recommended on tool spans | Not emitted | Dynamiq tool descriptions are prompt-sized. They are in `gen_ai.tool.definitions` and `dynamiq.node.config`. |
| `gen_ai.tool.call.id` | Recommended on tool spans | Not emitted | Not available to the tool run. |
| Usage on agent spans | Recommended | Not emitted | A backend that sums usage over all spans would count tokens twice. |
| `gen_ai.retrieval.documents[].score` | Required on each document | Omitted when the store returned no score | A score is not invented. |
| `server.address`, `server.port` | Recommended on client spans | Not emitted | The endpoint is chosen inside LiteLLM. |

### Aliases

Three attributes are emitted under a second name because common backends read only that name. No vendor-prefixed
attribute (`langfuse.*`, `openinference.*`, `traceloop.*`) is emitted.

| Alias | Canonical | Reason |
|---|---|---|
| `gen_ai.system` | `gen_ai.provider.name` | Deprecated in v1.37 but still the only provider key several backends read. |
| `gen_ai.usage.cache_write.input_tokens` | `gen_ai.usage.cache_creation.input_tokens` | The upstream rename. Backends are split between the two names. |
| `session.id` | `gen_ai.conversation.id` | Session views in several backends key on `session.id`. |

### Backend notes

- Backends that classify spans by the presence of `gen_ai.request.model` may show an agent span as a model call.
  Backends that read `gen_ai.operation.name` are not affected.
- `invoke_workflow` and `retrieval` are not mapped by every backend and may render as generic spans.
- Backends with small attribute limits truncate content. Use `DYNAMIQ_TRACING_CONTENT_MAX_LENGTH` to keep the JSON
  valid.
- No third-party backend follows `dynamiq.prompt.messages_base.*`. Leave
  [prompt compaction](#prompt-compaction) off when exporting to one.

## Upcoming upstream changes

The unreleased convention (`open-telemetry/semantic-conventions-genai`, commit `e6658f1`, 2026-11-06) changes the
following. None of it is emitted in shape version 1.

| Change upstream | Effect on a later shape version |
|---|---|
| JSON-string values are no longer allowed for message attributes on spans; structured values are required | Content attributes become structured values. |
| The `invoke_workflow` span is merged into the internal `invoke_agent` span; `gen_ai.workflow.name` is replaced by `gen_ai.agent.name` (the operation name `invoke_workflow` stays) | The workflow span is renamed and re-keyed. |
| `gen_ai.provider.name`, `gen_ai.agent.id` and cache token attributes are removed from the internal `invoke_agent` span | Dropped from agent spans. |
| `gen_ai.usage.cache_creation.input_tokens` is renamed `gen_ai.usage.cache_write.input_tokens` | The old name is dropped. |
| The client retrieval span is deprecated in favour of an internal `retrieval` span; `gen_ai.request.top_k` becomes `gen_ai.retrieval.top_k` on it | Retrieval spans change kind, name and one attribute. |
| `execute_tool` spans gain `gen_ai.agent.name` and `gen_ai.conversation.id` | Added. |
| `finish_reason` on output messages is deprecated in favour of `gen_ai.response.finish_reasons` | Dropped from messages. |
| The provider value `mistral_ai` is renamed `mistral` | Value changes. |
| Exceptions move to per-operation log-record events | Needs the logs signal. |
| Proposals not merged: cost (`gen_ai.usage.cost.*`), `user.id` on GenAI spans, pause and resume events, guardrails, tasks | Replace the matching `dynamiq.*` attributes when released. |

## Versioning

- `dynamiq.trace.shape_version` on the instrumentation scope identifies this document's shape. Version `1` is
  pinned to convention v1.41.1.
- Within a shape version, attributes are only added. A rename, a removal, a type change or a change of span name or
  kind needs a new shape version.
- A mode that changes what an attribute holds, as prompt compaction does for `gen_ai.input.messages`, is off unless
  the application turns it on, and marks every span it changes.
- A new shape version is introduced as an option first and becomes the default in a later release. The release
  notes name the shape version a release emits by default.
- `dynamiq.*` attributes that gain a released convention equivalent are emitted under both names for one shape
  version before the `dynamiq.*` name is removed.
