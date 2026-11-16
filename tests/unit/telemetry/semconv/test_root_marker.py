from uuid import UUID, uuid4

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.utils import Input
from dynamiq.telemetry import semconv
from dynamiq.telemetry.types import RunKind

from .helpers import assert_otel_valid

IS_ROOT = "dynamiq.run.is_root"
RUN_ID = "0b0e6a53-6a0f-4a55-9f0c-6f2f3f0f6f11"


def node_attributes(run_kwargs: dict) -> dict:
    spec = semconv.node_span_spec(Input(name="start"), run_id=RUN_ID, run_kwargs=run_kwargs)
    assert_otel_valid(spec.attributes)
    return spec.attributes


def flow_attributes(run_kwargs: dict, workflow: bool = False) -> dict:
    flow = Flow(nodes=[Input()])
    spec = semconv.workflow_span_spec(
        Workflow(flow=flow) if workflow else None,
        flow,
        run_id=RUN_ID,
        run_kind=RunKind.WORKFLOW if workflow else RunKind.FLOW,
        run_kwargs=run_kwargs,
    )
    assert_otel_valid(spec.attributes)
    return spec.attributes


@pytest.mark.parametrize(
    "run_kwargs",
    [{}, {"parent_run_id": None}, {"parent_run_id": RUN_ID}, {"parent_run_id": UUID(RUN_ID)}],
    ids=["no-parent", "parent-none", "parent-is-the-run", "parent-is-the-run-as-uuid"],
)
@pytest.mark.parametrize("attributes_of", [node_attributes, flow_attributes], ids=["node", "flow"])
def test_run_without_a_parent_run_is_marked_as_a_root(attributes_of, run_kwargs):
    # Flow.run_async names the flow run itself as its parent run when it has none.
    assert attributes_of(run_kwargs)[IS_ROOT] is True


@pytest.mark.parametrize("parent_run_id", [uuid4(), str(uuid4())], ids=["uuid", "string"])
@pytest.mark.parametrize("attributes_of", [node_attributes, flow_attributes], ids=["node", "flow"])
def test_run_with_a_parent_run_has_no_marker(attributes_of, parent_run_id):
    # Absent rather than false: a backend counts the spans that have the attribute.
    assert IS_ROOT not in attributes_of({"parent_run_id": parent_run_id})


def test_workflow_run_is_a_root_unless_a_run_started_it():
    assert flow_attributes({"wf_run_id": "user-run"}, workflow=True)[IS_ROOT] is True
    assert IS_ROOT not in flow_attributes({"parent_run_id": uuid4()}, workflow=True)
