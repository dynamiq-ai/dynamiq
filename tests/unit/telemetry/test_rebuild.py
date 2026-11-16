"""``rebuild_runs`` on spans built by hand, and the inverse of the message mapping.

Nothing here imports the OpenTelemetry SDK: the rebuild runs in the default install.
"""

import json
import subprocess
import sys
import textwrap

import pytest

from dynamiq.telemetry import content
from dynamiq.telemetry.rebuild import (
    OtlpEvent,
    OtlpSpan,
    OtlpSpanContext,
    messages_from_parts,
    rebuild_runs,
    spans_from_otlp_json,
)
from dynamiq.telemetry.types import CaptureSettings

TRACE_ID = 0x0AF7651916CD43DD8448EB211C80319C
START = 1_700_000_000_000_000_000


def span(
    span_id: int,
    parent: int | None = None,
    *,
    events: list[OtlpEvent] | None = None,
    start: int = 0,
    **attributes,
) -> OtlpSpan:
    return OtlpSpan(
        name=f"span {span_id}",
        context=OtlpSpanContext(trace_id=TRACE_ID, span_id=span_id),
        parent=OtlpSpanContext(trace_id=TRACE_ID, span_id=parent) if parent is not None else None,
        start_time=START + start,
        end_time=START + start + 1_000_000,
        attributes={key.replace("__", "."): value for key, value in attributes.items()},
        events=events or [],
    )


def node(span_id: int, parent: int | None = None, status: str = "succeeded", **attributes) -> OtlpSpan:
    defaults = {
        "dynamiq__run__id": f"run-{span_id}",
        "dynamiq__run__type": "node",
        "dynamiq__run__status": status,
        "dynamiq__node__id": f"node-{span_id}",
        "dynamiq__node__name": f"name-{span_id}",
        "dynamiq__node__type": "dynamiq.nodes.utils.Input",
        "dynamiq__node__group": "utils",
    }
    return span(span_id, parent, start=span_id, **(defaults | attributes))


def workflow(span_id: int, status: str = "succeeded", **attributes) -> OtlpSpan:
    defaults = {
        "dynamiq__run__id": f"run-{span_id}",
        "dynamiq__run__type": "workflow",
        "dynamiq__run__status": status,
        "dynamiq__run__is_root": True,
        "dynamiq__workflow__id": "wf",
        "dynamiq__flow__id": "flow",
    }
    return span(span_id, start=0, **(defaults | attributes))


def by_name(runs: list[dict]) -> dict[str, dict]:
    return {run["name"]: run for run in runs}


# Parents


def test_spans_without_a_run_id_are_left_out_and_runs_below_them_name_the_nearest_run():
    spans = [
        node(1, dynamiq__run__is_root=True),
        span(2, 1),  # an HTTP client span, say
        node(3, 2),
        span(4),  # a request span of the application
        node(5, 4, dynamiq__run__is_root=True),
        node(6, 4),
    ]

    runs = by_name(rebuild_runs(spans))

    assert set(runs) == {"name-1", "name-3", "name-5", "name-6"}
    assert runs["name-3"]["parent_run_id"] == "run-1"
    assert runs["name-5"]["parent_run_id"] is None
    # no run above it in the spans given: the run it names as parent is not known
    assert runs["name-6"]["parent_run_id"] is None


def test_a_root_has_no_parent_run_even_below_the_span_of_another_run():
    runs = by_name(rebuild_runs([node(1, dynamiq__run__is_root=True), node(2, 1, dynamiq__run__is_root=True)]))

    assert runs["name-2"]["parent_run_id"] is None


def test_a_parent_span_missing_from_the_batch_gives_no_parent_run():
    (run,) = rebuild_runs([node(2, 1)])

    assert run["parent_run_id"] is None


def test_run_fields_come_from_the_span():
    spans = [node(1, dynamiq__run__is_root=True, dynamiq__tags=("a", "b"), dynamiq__source__id="src")]
    spans[0].attributes |= {"gen_ai.conversation.id": "session", "dynamiq.metadata.team": "core"}

    (run,) = rebuild_runs(spans)

    assert run["trace_id"] == "0af76519-16cd-43dd-8448-eb211c80319c"
    assert (run["tags"], run["source_id"], run["session_id"]) == (["a", "b"], "src", "session")
    assert run["metadata"]["team"] == "core"
    assert (run["end_time"] - run["start_time"]).total_seconds() == 0.001


# Status, error and executions


def test_a_paused_run_is_failed_without_an_error_and_its_attempt_never_ended():
    (run,) = rebuild_runs([node(1, status="paused")])

    assert run["status"] == "failed"
    assert run["error"] is None
    assert [(e["status"], e["end_time"]) for e in run["executions"]] == [(None, None)]
    assert run["metadata"]["is_mocked"] is False
    assert "is_output_from_cache" not in run["metadata"]


def test_a_successful_first_attempt_is_the_span_and_has_no_id():
    (run,) = rebuild_runs([node(1)])

    (execution,) = run["executions"]
    assert execution["id"] is None
    assert (execution["start_time"], execution["end_time"]) == (run["start_time"], run["end_time"])
    assert execution["status"] == "succeeded"


@pytest.mark.parametrize(
    "status, attributes",
    [
        ("succeeded", {"dynamiq__node__is_mocked": True}),
        ("succeeded", {"dynamiq__node__is_output_from_cache": True}),
        ("skipped", {}),
        ("failed", {}),  # failed before an attempt started
        ("canceled", {}),
    ],
)
def test_runs_without_an_attempt_have_no_execution(status, attributes):
    (run,) = rebuild_runs([node(1, status=status, **attributes)])

    assert run["executions"] == []


def test_attempts_and_the_error_come_from_events():
    events = [
        OtlpEvent(
            name="dynamiq.node.execution",
            timestamp=START + 500_000,
            attributes={
                "dynamiq.execution.id": "attempt-1",
                "dynamiq.execution.attempt": 1,
                "dynamiq.execution.status": "failed",
                "dynamiq.execution.start_time_unix_nano": START + 100_000,
                "error.type": "ValueError",
                "exception.message": "boom",
            },
        ),
        OtlpEvent(
            name="exception",
            timestamp=START + 600_000,
            attributes={"exception.type": "ValueError", "exception.message": "boom", "exception.stacktrace": "tb"},
        ),
    ]

    (run,) = rebuild_runs([node(1, status="failed", events=events)])

    assert run["error"] == {"message": "boom", "traceback": "tb"}
    (execution,) = run["executions"]
    assert (execution["id"], execution["status"]) == ("attempt-1", "failed")
    assert execution["error"] == {"message": "boom", "traceback": None}
    assert execution["end_time"] > execution["start_time"]
    assert run["metadata"] == {
        "node": {
            "id": "node-1",
            "name": "name-1",
            "type": "dynamiq.nodes.utils.Input",
            "group": "utils",
            "depends": [],
        },
        "run_depends": [],
        "is_mocked": False,
    }


def test_hooks_merge_their_details():
    event = OtlpEvent(
        name="dynamiq.agent.hook",
        attributes={
            "dynamiq.hook.name": "call_limit",
            "dynamiq.hook.point": "before_tool",
            "dynamiq.hook.decision": "block",
            "dynamiq.hook.details": json.dumps({"tool": "search"}),
        },
    )

    (run,) = rebuild_runs([node(1, events=[event])])

    assert run["metadata"]["hooks"] == [
        {"hook": "call_limit", "type": None, "point": "before_tool", "decision": "block", "tool": "search"}
    ]


def test_usage_of_a_chat_span_and_of_another_span():
    chat = node(
        1,
        gen_ai__operation__name="chat",
        gen_ai__usage__input_tokens=11,
        gen_ai__usage__output_tokens=7,
        dynamiq__usage__total_tokens=18,
        dynamiq__usage__total_cost_usd=0.5,
    )
    tool = node(2, dynamiq__usage__input_tokens=3, dynamiq__usage__output_tokens=4)

    runs = by_name(rebuild_runs([chat, tool]))

    assert runs["name-1"]["metadata"]["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "total_tokens_cost_usd": 0.5,
    }
    assert runs["name-2"]["metadata"]["usage"] == {"prompt_tokens": 3, "completion_tokens": 4}


# Workflow and flow output


def test_workflow_output_is_the_map_of_its_node_runs():
    spans = [
        workflow(1, dynamiq__output=json.dumps("terminal output only")),
        node(2, 1, dynamiq__input='{"a": 1}', dynamiq__output='{"b": 2}'),
        node(3, 1, status="skipped"),
        node(
            4,
            1,
            status="failed",
            events=[OtlpEvent(name="exception", attributes={"exception.type": "E", "exception.message": "m"})],
        ),
        node(5, 2),  # below a node: not in the map
    ]

    runs = by_name(rebuild_runs(spans))

    assert runs["Workflow"]["output"] == {
        "node-2": {"status": "success", "input": {"a": 1}, "output": {"b": 2}},
        "node-3": {"status": "skip", "input": None, "output": None},
        "node-4": {"status": "failure", "input": None, "output": None, "error": {"type": "E", "message": "m"}},
    }
    assert runs["Workflow"]["metadata"] == {"workflow": {"id": "wf", "version": None}}


def test_a_flow_run_has_the_map_too_and_no_output_when_it_did_not_succeed():
    flow = workflow(1, dynamiq__run__type="flow")
    failed = workflow(2, status="failed")

    runs = rebuild_runs([flow, node(3, 1), failed, node(4, 2)])

    assert runs[0]["name"] == "Flow"
    assert runs[0]["output"] == {"node-3": {"status": "success", "input": None, "output": None}}
    assert runs[0]["metadata"] == {"flow": {"id": "flow"}}
    assert runs[1]["output"] is None


# Prompts


def chat_span(span_id: int, parent: int | None, messages: list[dict], **attributes) -> OtlpSpan:
    settings = CaptureSettings(binary_content_enabled=True)
    return node(
        span_id,
        parent,
        gen_ai__operation__name="chat",
        dynamiq__node__group="llms",
        gen_ai__input__messages=content.to_json(content.input_messages(messages, settings), settings),
        **attributes,
    )


def test_a_compacted_prompt_names_the_run_of_its_base_span():
    agent = node(1, dynamiq__run__is_root=True)
    first = chat_span(2, 1, [{"role": "user", "content": "hi"}])
    second = chat_span(
        3,
        1,
        [{"role": "assistant", "content": "hello"}],
        dynamiq__prompt__messages_base__span_id=format(2, "016x"),
        dynamiq__prompt__messages_base__count=1,
    )

    runs = rebuild_runs([agent, first, second], prompt_compaction_enabled=True)

    assert runs[1]["metadata"]["node"]["prompt"] == {
        "messages": [{"content": "hi", "role": "user"}],
        "messages_base": None,
    }
    assert runs[2]["metadata"]["node"]["prompt"]["messages_base"] == {"run_id": "run-2", "count": 1}


def test_without_compaction_a_whole_prompt_has_no_messages_base_key():
    runs = rebuild_runs([node(1), chat_span(2, 1, [{"role": "user", "content": "hi"}])])

    assert runs[1]["metadata"]["node"]["prompt"] == {"messages": [{"content": "hi", "role": "user"}]}


ROUND_TRIP_MESSAGES = [
    {"content": "Answer briefly.", "role": "system"},
    {"content": "", "role": "user"},
    {
        "content": "",
        "role": "assistant",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"query": "q"}'}},
            {"id": "call_2", "type": "function", "function": {"name": "broken", "arguments": "{not json"}},
        ],
    },
    {"content": "results", "role": "tool", "tool_call_id": "call_1", "name": "search"},
    {
        "content": [
            {"type": "text", "text": "Describe these."},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "file", "file": {"file_id": "file-1", "format": "application/pdf"}},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0="}},
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
            {"type": "refusal", "refusal": "no"},
        ],
        "role": "user",
    },
]


def test_messages_from_parts_inverts_the_message_mapping():
    settings = CaptureSettings(binary_content_enabled=True)
    parts = json.loads(content.to_json(content.input_messages(ROUND_TRIP_MESSAGES, settings), settings))

    assert messages_from_parts(parts) == ROUND_TRIP_MESSAGES


def test_inline_binary_left_out_of_the_span_rebuilds_without_its_data():
    message = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
    parts = json.loads(content.to_json(content.input_messages([message], CaptureSettings()), CaptureSettings()))

    assert messages_from_parts(parts) == [
        {"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}], "role": "user"}
    ]


# OTLP/JSON


def test_spans_from_otlp_json_reads_ids_values_and_events():
    request = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "scope": {"name": "dynamiq", "version": "1.0"},
                        "spans": [
                            {
                                "traceId": "0af7651916cd43dd8448eb211c80319c",
                                "spanId": "00f067aa0ba902b7",
                                "parentSpanId": "00000000000000aa",
                                "name": "end",
                                "startTimeUnixNano": "1700000000000000000",
                                "endTimeUnixNano": "1700000000000000500",
                                "attributes": [
                                    {"key": "s", "value": {"stringValue": "x"}},
                                    {"key": "i", "value": {"intValue": "9007199254740993"}},
                                    {"key": "d", "value": {"doubleValue": 0.5}},
                                    {"key": "b", "value": {"boolValue": True}},
                                    {"key": "a", "value": {"arrayValue": {"values": [{"stringValue": "p"}]}}},
                                ],
                                "events": [{"name": "e", "timeUnixNano": "1700000000000000100"}],
                            }
                        ],
                    }
                ]
            }
        ]
    }

    (parsed,) = spans_from_otlp_json(json.dumps(request))

    assert parsed.context == OtlpSpanContext(trace_id=TRACE_ID, span_id=0x00F067AA0BA902B7)
    assert parsed.parent.span_id == 0xAA
    assert (parsed.start_time, parsed.end_time) == (1_700_000_000_000_000_000, 1_700_000_000_000_000_500)
    assert parsed.attributes == {"s": "x", "i": 9007199254740993, "d": 0.5, "b": True, "a": ("p",)}
    assert [(event.name, event.timestamp) for event in parsed.events] == [("e", 1_700_000_000_000_000_100)]
    assert (parsed.scope_name, parsed.scope_version) == ("dynamiq", "1.0")


def test_the_rebuild_imports_without_the_sdk():
    script = textwrap.dedent(
        """
        import json
        import sys

        import dynamiq.telemetry.rebuild

        print(json.dumps(sorted(name for name in sys.modules if name.startswith("opentelemetry.sdk"))))
        """
    )

    output = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout

    assert json.loads(output.strip().splitlines()[-1]) == []
