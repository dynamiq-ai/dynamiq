"""A trace keeps every Rules finding whole, however many rules a node carries.

Workflow traces cut every list longer than `TRUNCATE_LIST_LIMIT` (50) when `format_value` runs `for_tracing`.
The platform UI reads a node's traced `output.findings` to show each rule's coverage, so without an opt-out a
node with more than 50 rules would have every rule past the 50th read as "never fired". These tests pin the fix
at the level the UI actually reads (the node run inside a `TracingCallbackHandler`), check the opt-out is narrow
(an ordinary list beside it still gets cut), and check the marker changes nothing about the node's own,
non-traced output.
"""

import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunType
from dynamiq.flows import Flow
from dynamiq.nodes.operators import Rules
from dynamiq.nodes.types import NamedField, Rule
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.utils import TRUNCATE_LIST_LIMIT, UntruncatedList, format_value

RULE_COUNT = 60
assert RULE_COUNT > TRUNCATE_LIST_LIMIT, "the node must carry more rules than a trace would otherwise keep"


def _node_with_many_rules() -> Rules:
    """A node whose rules split cleanly on `order.total = 30`: the first 30 pass, the rest fail."""
    rules = [Rule(id=f"r{index}", name=f"rule {index}", check=f"order.total > {index}") for index in range(RULE_COUNT)]
    return Rules(id="checks", name="checks", input_fields=[NamedField(name="order")], rules=rules)


def test_a_traced_node_run_keeps_every_finding_past_the_truncate_limit():
    """The platform UI reads coverage from the node run inside the trace, not the workflow's own result."""
    node = _node_with_many_rules()
    tracing = TracingCallbackHandler()
    workflow = Workflow(flow=Flow(nodes=[node]))

    result = workflow.run(input_data={"order": {"total": 30}}, config=RunnableConfig(callbacks=[tracing]))

    assert result.status == RunnableStatus.SUCCESS
    node_run = next(run for run in tracing.runs.values() if run.type == RunType.NODE)
    assert len(node_run.output["findings"]) == RULE_COUNT
    assert sum(node_run.output["summary"].values()) == RULE_COUNT
    assert node_run.output["summary"] == {
        "pass": 30,
        "fail": 30,
        "warn": 0,
        "info": 0,
        "not_applicable": 0,
        "not_evaluated": 0,
    }


def test_a_plain_list_alongside_it_still_gets_cut_to_the_limit():
    """The opt-out is narrow: an ordinary list at the same depth keeps today's truncation."""
    payload = {
        "findings": UntruncatedList([{"n": i} for i in range(RULE_COUNT)]),
        "other_list": list(range(RULE_COUNT)),
    }

    traced = format_value(payload, for_tracing=True)

    assert len(traced["findings"]) == RULE_COUNT
    assert len(traced["other_list"]) == TRUNCATE_LIST_LIMIT


def test_format_value_keeps_the_marker_type_and_every_item_when_tracing():
    marker = UntruncatedList([{"n": i} for i in range(RULE_COUNT)])

    formatted = format_value(marker, for_tracing=True)

    assert type(formatted) is UntruncatedList
    assert len(formatted) == RULE_COUNT
    assert formatted == list(marker)


def test_the_nodes_direct_output_is_unchanged_and_round_trips_through_json():
    """Only the traced copy is exempt from truncation; the node's own returned findings are a list like any
    other - unaffected in shape, length, equality or JSON round trip by carrying the marker type."""
    node = _node_with_many_rules()

    result = node.run(input_data={"order": {"total": 30}}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    findings = result.output["findings"]
    assert isinstance(findings, list)
    assert len(findings) == RULE_COUNT
    assert [finding["rule_id"] for finding in findings] == [f"r{index}" for index in range(RULE_COUNT)]
    assert [finding["status"] for finding in findings] == ["pass"] * 30 + ["fail"] * 30

    # Equal to a plain list built independently of the marker type, and identical once serialized.
    plain_list_findings = [dict(finding) for finding in findings]
    assert findings == plain_list_findings
    assert json.dumps(findings) == json.dumps(plain_list_findings)
    assert json.loads(json.dumps(findings)) == findings
