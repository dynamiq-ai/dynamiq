"""The equivalence under which runs rebuilt from spans equal the runs a ``TracingCallbackHandler`` recorded.

Every exception to plain equality is a rule here, and each rule is listed in fixtures/README.md. Both sides are
compared in their JSON form, as the handler's client sends runs.
"""

import json
from datetime import datetime, timedelta
from typing import Any

import orjson
from pydantic import BaseModel

from dynamiq.utils.utils import orjson_encode

TIME_TOLERANCE = timedelta(seconds=0.5)
WORKFLOW, FLOW, NODE = "workflow", "flow", "node"
PAUSED = "paused"


class ScenarioOptions(BaseModel):
    """How a scenario ran: what a consumer passes to its rebuild and what the comparison allows for."""

    description: str = ""
    prompt_compaction_enabled: bool = False
    content_capture_enabled: bool = True
    node_config_enabled: bool = False


def to_json(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs as the handler's client encodes them, read back."""
    return json.loads(orjson.dumps(runs, default=orjson_encode, option=orjson.OPT_NON_STR_KEYS))


def paused_run_ids(spans: list) -> set[str]:
    return {
        span.attributes["dynamiq.run.id"]
        for span in spans
        if "dynamiq.run.id" in span.attributes and span.attributes.get("dynamiq.run.status") == PAUSED
    }


def configured_run_ids(spans: list) -> set[str]:
    return {
        span.attributes["dynamiq.run.id"]
        for span in spans
        if "dynamiq.run.id" in span.attributes and "dynamiq.node.config" in span.attributes
    }


class Mismatches:
    def __init__(self):
        self.found: list[str] = []

    def equal(self, where: str, rebuilt: Any, recorded: Any) -> None:
        if rebuilt != recorded:
            self.found.append(f"{where}: rebuilt {_short(rebuilt)} != recorded {_short(recorded)}")

    def check(self, where: str, condition: bool, detail: str = "") -> None:
        if not condition:
            self.found.append(f"{where}: {detail}")


def _short(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) < 400 else text[:400] + "..."


def _time(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _close(found: Mismatches, where: str, rebuilt: str | None, recorded: str | None) -> None:
    if rebuilt is None or recorded is None:
        found.equal(where, rebuilt, recorded)
        return
    found.check(where, abs(_time(rebuilt) - _time(recorded)) <= TIME_TOLERANCE, f"{rebuilt} vs {recorded}")


def _ids(depends: list | None) -> list[str]:
    return [item["node"]["id"] if isinstance(item, dict) else item for item in depends or []]


def canonical_messages(messages: list[dict]) -> list[dict]:
    """Prompt messages without null fields and without the fields a request does not carry, with tool call
    arguments parsed: a JSON string and the object it encodes are the same arguments."""
    out = []
    for message in json.loads(json.dumps(messages, default=str)):
        message = {k: v for k, v in message.items() if v is not None and k not in ("static", "metadata")}
        for call in message.get("tool_calls") or []:
            arguments = call.get("function", {}).get("arguments")
            if isinstance(arguments, str):
                try:
                    call["function"]["arguments"] = json.loads(arguments)
                except ValueError:
                    pass
        out.append(message)
    return out


def _expected_flow_output(output: dict | None, content: bool) -> dict | None:
    """The handler's output map of a workflow or flow run without what no span holds."""
    if output is None:
        return None
    expected = {}
    for node_id, entry in output.items():
        if entry.get("status") == "undefined":
            continue
        item = {"status": entry.get("status")}
        item["input"] = entry.get("input") if content else None
        item["output"] = entry.get("output") if content else None
        error = entry.get("error")
        if error is not None and entry.get("status") != "skip":
            item["error"] = {"type": error.get("type"), "message": error.get("message") if content else None}
        expected[node_id] = item
    return expected


def _skip_identifiers(skip: dict | None) -> dict | None:
    if skip is None:
        return None
    dependency = skip.get("failed_dependency") or {}
    node = dependency.get("node") or {}
    return {
        "failed_dependency": {
            "node": {key: node.get(key) for key in ("id", "name", "type")},
            "option": dependency.get("option"),
        }
    }


def assert_equivalent(
    rebuilt: list[dict[str, Any]], recorded: list[dict[str, Any]], spans: list, options: ScenarioOptions
) -> None:
    """Assert that ``rebuilt`` (from ``rebuild_runs``) and ``recorded`` (a handler's runs) agree, in JSON form."""
    rebuilt, recorded = to_json(rebuilt), to_json(recorded)
    content = options.content_capture_enabled
    paused = paused_run_ids(spans)
    configured = configured_run_ids(spans)
    found = Mismatches()

    by_id = {run["id"]: run for run in recorded}
    # The flow run of a workflow has no span: it is not rebuilt, and its nodes hang below the workflow run.
    workflow_flows = {
        run["id"]: run["parent_run_id"]
        for run in recorded
        if run["type"] == FLOW and by_id.get(run["parent_run_id"], {}).get("type") == WORKFLOW
    }
    expected = {run_id: run for run_id, run in by_id.items() if run_id not in workflow_flows}
    rebuilt_by_id = {run["id"]: run for run in rebuilt}
    found.equal("run ids", sorted(rebuilt_by_id), sorted(expected))
    found.check("trace ids", len({run["trace_id"] for run in rebuilt}) <= 1, "one trace")

    for run_id in sorted(set(rebuilt_by_id) & set(expected)):
        _compare_run(
            found,
            rebuilt_by_id[run_id],
            expected[run_id],
            workflow_flows,
            run_id in paused,
            run_id in configured,
            content,
        )

    assert not found.found, "rebuilt runs differ from the handler's:\n" + "\n".join(found.found)


def _compare_run(
    found: Mismatches,
    run: dict,
    handler: dict,
    workflow_flows: dict[str, str],
    paused: bool,
    configured: bool,
    content: bool,
) -> None:
    where = f"{handler['type']} {handler['name']} {handler['id']}"
    for key in ("type", "name", "status", "tags", "source_id", "session_id"):
        found.equal(f"{where} {key}", run[key], handler[key])
    parent = handler["parent_run_id"]
    found.equal(f"{where} parent_run_id", run["parent_run_id"], workflow_flows.get(parent, parent))
    _close(found, f"{where} start_time", run["start_time"], handler["start_time"])
    _close(found, f"{where} end_time", run["end_time"], handler["end_time"])

    if content:
        found.equal(f"{where} input", run["input"], handler["input"])
    else:
        found.equal(f"{where} input", run["input"], None)
    if handler["type"] in (WORKFLOW, FLOW):
        found.equal(f"{where} output", run["output"], _expected_flow_output(handler["output"], content))
    else:
        found.equal(f"{where} output", run["output"], handler["output"] if content else None)

    _compare_error(found, where, run, handler, paused, content)
    _compare_executions(found, where, run["executions"], handler["executions"], content)
    _compare_metadata(found, where, run["metadata"], handler["metadata"], configured, content)


def _compare_error(found: Mismatches, where: str, run: dict, handler: dict, paused: bool, content: bool) -> None:
    error, handler_error = run["error"], handler["error"]
    if paused:
        # A paused run reports an error to the handler; its span records no exception.
        found.check(f"{where} error", error is None and handler_error is not None, f"{error} / {handler_error}")
        return
    found.equal(f"{where} error is set", error is not None, handler_error is not None)
    if error is None or handler_error is None:
        return
    if content:
        found.equal(f"{where} error.message", error["message"], handler_error["message"])
        found.check(f"{where} error.traceback", bool(error["traceback"]) and bool(handler_error["traceback"]))
    else:
        found.equal(f"{where} error", error, {"message": None, "traceback": None})


def _compare_executions(found: Mismatches, where: str, executions: list, handler_executions: list, content: bool):
    found.equal(
        f"{where} execution statuses",
        [e["status"] for e in executions],
        [e["status"] for e in handler_executions],
    )
    for index, (execution, handler_execution) in enumerate(zip(executions, handler_executions)):
        at = f"{where} execution {index}"
        if execution["id"] is None:
            # The span is the attempt of a run that succeeded or paused on its first attempt: no event names it.
            found.check(
                f"{at} id",
                len(handler_executions) == 1 and handler_execution["status"] in ("succeeded", None),
                "only a lone attempt that succeeded or paused has no id",
            )
        else:
            found.equal(f"{at} id", execution["id"], handler_execution["id"])
        _close(found, f"{at} start_time", execution["start_time"], handler_execution["start_time"])
        _close(found, f"{at} end_time", execution["end_time"], handler_execution["end_time"])
        for key in ("input", "output", "metadata"):
            found.equal(f"{at} {key}", execution[key], handler_execution[key])
        handler_error = handler_execution["error"]
        expected_error = None
        if handler_error is not None:
            expected_error = {"message": handler_error["message"] if content else None, "traceback": None}
        found.equal(f"{at} error", execution["error"], expected_error)


def _compare_metadata(
    found: Mismatches, where: str, metadata: dict, handler: dict, configured: bool, content: bool
) -> None:
    content_keys = {"tool_data", "human_feedback"}
    expected_keys = set(handler) if content else set(handler) - content_keys
    found.equal(f"{where} metadata keys", sorted(metadata), sorted(expected_keys))

    for key in set(metadata) & set(handler):
        value, handler_value = metadata[key], handler[key]
        at = f"{where} metadata.{key}"
        if key == "node":
            _compare_node(found, at, value, handler_value, configured, content)
        elif key == "run_depends":
            found.equal(at, value, _ids(handler_value))
        elif key == "usage":
            for field, handler_field in handler_value.items():
                if field in value:
                    found.equal(f"{at}.{field}", value[field], handler_field)
                else:
                    found.check(f"{at}.{field}", handler_field in (None, 0), f"missing, recorded {handler_field}")
            found.check(at, set(value) <= set(handler_value), f"extra keys {set(value) - set(handler_value)}")
        elif key == "hooks" and not content:
            found.equal(
                at, value, [{k: hook.get(k) for k in ("hook", "type", "point", "decision")} for hook in handler_value]
            )
        elif key == "skip" and not content:
            found.equal(at, value, _skip_identifiers(handler_value))
        elif key == "failed_nodes" and not content:
            found.equal(at, value, [{k: v for k, v in node.items() if k != "error_message"} for node in handler_value])
        else:
            found.equal(at, value, handler_value)


def _compare_node(found: Mismatches, where: str, node: dict, handler: dict, configured: bool, content: bool) -> None:
    prompt = node.get("prompt") if isinstance(node.get("prompt"), dict) else None
    handler_prompt = handler.get("prompt") if isinstance(handler.get("prompt"), dict) else None
    handler_messages = (handler_prompt or {}).get("messages")
    if content and handler_messages:
        found.check(f"{where}.prompt", prompt is not None, "no prompt rebuilt")
        if prompt is not None:
            found.equal(
                f"{where}.prompt.messages",
                canonical_messages(prompt.get("messages") or []),
                canonical_messages(handler_messages),
            )
            found.equal(
                f"{where}.prompt has messages_base", "messages_base" in prompt, "messages_base" in handler_prompt
            )
            found.equal(
                f"{where}.prompt.messages_base", prompt.get("messages_base"), handler_prompt.get("messages_base")
            )
    else:
        found.check(f"{where}.prompt", prompt is None or not prompt.get("messages"), "messages without content")

    if configured:
        rest = {k: v for k, v in node.items() if k != "prompt"}
        handler_rest = {k: v for k, v in handler.items() if k != "prompt"}
        found.equal(f"{where} (configuration)", rest, handler_rest)
        if prompt is not None and handler_prompt is not None:
            found.equal(
                f"{where}.prompt (configuration)",
                {k: v for k, v in prompt.items() if k not in ("messages", "messages_base")},
                {k: v for k, v in handler_prompt.items() if k not in ("messages", "messages_base")},
            )
        return
    expected = {key: handler.get(key) for key in ("id", "name", "type", "group")}
    if handler.get("model") is not None:
        expected["model"] = handler["model"]
    expected["depends"] = _ids(handler.get("depends"))
    found.equal(where, {k: v for k, v in node.items() if k != "prompt"}, expected)
