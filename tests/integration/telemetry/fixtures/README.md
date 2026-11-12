# Golden fixtures: spans and the runs a handler recorded for them

Each directory holds one run of the Dynamiq SDK, traced two ways at once: as OpenTelemetry spans and by a
`TracingCallbackHandler`. An implementation of the "Rebuilding a run" rules of
[`docs/reference/opentelemetry-tracing.md`](../../../../docs/reference/opentelemetry-tracing.md) passes when it
rebuilds the spans into the recorded runs, under the equivalence below. The Python implementation is
`dynamiq.telemetry.rebuild.rebuild_runs`; `test_golden_fixtures.py` runs it on these files.

| | |
|---|---|
| Trace shape version | `1` (`dynamiq.trace.shape_version` on the instrumentation scope) |
| Pinned convention | OpenTelemetry semantic conventions v1.41.1 |

## Files

| File | Content |
|---|---|
| `<scenario>/otlp.json` | The trace as an OTLP/JSON `ExportTraceServiceRequest`, encoded by the SDK's OTLP encoder (`opentelemetry.exporter.otlp.proto.common.trace_encoder.encode_spans`) and written with the OTLP/JSON rules: lowerCamelCase keys, `traceId`, `spanId` and `parentSpanId` as lowercase hex, 64-bit integers (`startTimeUnixNano`, `intValue`, ...) as strings, enums (`kind`, `status.code`) as integers, default values left out. |
| `<scenario>/otlp.pb` | The same request as binary protobuf. It decodes to the same request as `otlp.json`. |
| `<scenario>/runs.json` | What the handler recorded: `[run.to_dict() for run in handler.runs.values()]`, encoded as `DynamiqTracingClient` encodes runs (orjson with `orjson_encode`), indented. |
| `manifest.json` | One entry per scenario: `description`, and the settings a consumer needs: `prompt_compaction_enabled` (spans and handler both compacted prompts; pass it to the rebuild), `content_capture_enabled`, `node_config_enabled`. |

The resource of every trace is `{"service.name": "dynamiq-golden-fixtures"}`. The scope version is the `dynamiq`
version that wrote the files. Run ids, span ids and times are those of the run that wrote them.

## Scenarios

| Directory | Run |
|---|---|
| `llm_workflow` | Input -> LLM -> Output in a workflow, node configuration capture on |
| `agent_function_calling` | Agent with a tool in a workflow, `FUNCTION_CALLING`; a call-limit hook blocks a call (hook events) |
| `agent_xml` | Agent with a tool run on its own (a root node run), `XML` |
| `agent_tool_failure` | Agent whose tool fails; the agent recovers |
| `retry` | A node that fails its first attempt and succeeds on the second (`dynamiq.node.execution` events) |
| `choice_skipped` | A `Choice` whose second branch is skipped, node configuration capture on |
| `map` | A `Map` over three items |
| `sub_workflow` | A sub-workflow node: a flow span below a node span, with a flow output map |
| `paused` | A workflow that pauses on a checkpoint (`dynamiq.run.status = paused`) |
| `foreign_spans` | The workflow runs inside an application span; a node starts an HTTP client span and runs a node without a parent run inside it, which makes that run a root |
| `prompt_compaction` | An agent with four model calls, prompt compaction on for the spans and the handler |
| `failed_workflow` | A node fails, the node after it is skipped, the workflow fails (`dynamiq.failed_nodes`) |
| `content_capture_off` | Content capture off: hook events, a failing tool, a node that fails and returns, a skipped branch |

No scenario covers a canceled run, a cached or mocked run, embeddings, retrieval or a `ProcessExecutor` worker.

## Regenerating

```bash
DYNAMIQ_GOLDEN_FIXTURES_WRITE_ENABLED=1 uv run --frozen python -m tests.integration.telemetry.generate_fixtures [scenario ...]
```

The scenarios are in `golden_scenarios.py`. Model calls are scripted, so no credentials are needed. The generator
refuses to write a scenario whose rebuilt runs do not match its handler's runs, and rewrites `manifest.json`. Without
the environment variable it writes nothing. `test_golden_freshness.py` reruns every scenario on each test run, so a
change to the emitter that breaks the mapping fails even while these files are unchanged.

## Equivalence

Compare the rebuilt runs with `runs.json` after JSON encoding (times are RFC 3339 strings, ids strings). Every field
must be equal except as listed here. The Python form of these rules is `equivalence.py`.

Runs and the tree

1. The handler records a `flow` run for the flow of every workflow; it has no span. Leave out every recorded run of
   type `flow` whose parent run is a `workflow` run. All other runs match one to one by `id`.
2. `parent_run_id`: a recorded run whose parent is such a flow run is compared with the workflow run's id.
3. `trace_id` is not compared: the handler records an id of its own. The rebuilt `trace_id` is the OTLP `traceId` as
   a dashed UUID string, the same for every run.
4. `start_time` and `end_time` of runs and executions agree within 0.5 seconds; a null time equals only a null time.
   (The handler and the span take their own clock readings; a skipped run's recorded `end_time` is its
   `start_time`.)

Errors

5. A run whose span has `dynamiq.run.status = paused` has `status: failed` on both sides, a recorded `error`, and
   rebuilt `error: null`.
6. Otherwise `error` is null on both sides or on neither. With content capture on, `error.message` is equal and
   `error.traceback` is a non-empty string on both sides; its text is not compared. With content capture off the
   rebuilt `error` is `{"message": null, "traceback": null}`.

Executions

7. The same number of executions with the same `status` in order.
8. A rebuilt execution with `id: null` is allowed only as the single execution of a run whose recorded execution has
   status `succeeded` or `null` (a first attempt that succeeded or paused: the span is the attempt). Every other
   `id` is equal.
9. Execution `error`: null when the recorded one is null; otherwise `{"message": <recorded message>, "traceback":
   null}`, with `message` null when content capture is off. `input`, `output` and `metadata` are equal.

Input and output

10. With content capture on, `input` is equal, and `output` of a node run is equal. With it off both are null in the
    rebuilt run.
11. `output` of a `workflow` or `flow` run: the recorded map without entries whose `status` is `undefined`; in each
    entry, `error` is reduced to `{"type", "message"}` and removed from entries with status `skip`. With content
    capture off, every entry's `input`, `output` and `error.message` are null. A run that did not succeed has
    `output: null` on both sides.

Metadata

12. The key sets of `metadata` are equal, except that with content capture off the rebuilt run has no `tool_data`
    and no `human_feedback`.
13. `metadata.node`, when the run's span has `dynamiq.node.config`: equal to the recorded `node`, with
    `prompt.messages` compared by rule 15. Without that attribute (node configuration capture off or content capture
    off): the rebuilt `node` without `prompt` equals `{id, name, type, group, depends}` of the recorded node plus
    `model` when the recorded `model` is not null, where `depends` is the list of `depends[].node.id`.
14. `metadata.run_depends` equals the list of `node.id` of the recorded `run_depends`.
15. `metadata.node.prompt`, with content capture on and recorded `messages`: `messages` equal after both sides drop
    message keys whose value is null, `static` and `metadata`, and tool call `function.arguments` strings are parsed
    as JSON (when they parse). `messages_base` is equal, and present on both sides or on neither. With content
    capture off the rebuilt node has no prompt messages.
16. `metadata.usage`: every rebuilt key equals the recorded value; a recorded key missing from the rebuilt usage has
    the value `0` or null; the rebuilt usage has no key the recorded one lacks.
17. With content capture off: `metadata.hooks` equals the recorded hooks reduced to `{hook, type, point,
    decision}`; `metadata.skip` equals `{"failed_dependency": {"node": {id, name, type}, "option"}}` of the recorded
    skip; `metadata.failed_nodes` equals the recorded entries without `error_message`.
18. All other fields (`id`, `type`, `name`, `status`, `tags`, `source_id`, `session_id`, `metadata.workflow`,
    `metadata.flow`, `metadata.is_mocked`, `metadata.is_output_from_cache`, `metadata.hooks`, `metadata.skip`,
    `metadata.failed_nodes`, `metadata.tool_data`, `metadata.human_feedback`) are equal.
