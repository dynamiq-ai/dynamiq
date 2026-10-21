import asyncio
from collections import Counter, defaultdict
from unittest.mock import patch

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import BaseCallbackHandler, TracingCallbackHandler
from dynamiq.callbacks.base import READS_SERIALIZATION_ONLY, reads_serialization_only
from dynamiq.callbacks.streaming import AgentStreamingParserCallback, StreamingQueueCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.nodes.tools.judgement import _EvidenceRecorder
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableConfig, RunnableStatus

NODE_HOOKS = ("on_node_end", "on_node_execute_start", "on_node_execute_end", "on_node_execute_run")


class Reader(BaseCallbackHandler):
    """Records the serialization each marked hook receives."""

    def __init__(self):
        self.seen = defaultdict(list)

    def _record(self, hook, serialized):
        self.seen[(hook, serialized["id"])].append(serialized)

    @reads_serialization_only
    def on_workflow_start(self, serialized, input_data, **kwargs):
        self._record("on_workflow_start", serialized)

    @reads_serialization_only
    def on_workflow_end(self, serialized, output_data, **kwargs):
        self._record("on_workflow_end", serialized)

    @reads_serialization_only
    def on_flow_start(self, serialized, input_data, **kwargs):
        self._record("on_flow_start", serialized)

    @reads_serialization_only
    def on_flow_end(self, serialized, output_data, **kwargs):
        self._record("on_flow_end", serialized)

    @reads_serialization_only
    def on_node_end(self, serialized, output_data, **kwargs):
        self._record("on_node_end", serialized)

    @reads_serialization_only
    def on_node_execute_start(self, serialized, input_data, **kwargs):
        self._record("on_node_execute_start", serialized)

    @reads_serialization_only
    def on_node_execute_end(self, serialized, output_data, **kwargs):
        self._record("on_node_execute_end", serialized)


class Editor(BaseCallbackHandler):
    """Edits every serialization its unmarked hooks receive."""

    def __init__(self):
        self.seen = []

    def on_node_start(self, serialized, input_data, **kwargs):
        self.seen.append(serialized)
        serialized["name"] = "edited"

    def on_node_end(self, serialized, output_data, **kwargs):
        self.seen.append(serialized)
        serialized["name"] = "edited"


def make_workflow() -> Workflow:
    first = Input()
    return Workflow(flow=Flow(nodes=[first, Output(depends=[NodeDependency(first)])]))


@pytest.fixture
def to_dict_calls():
    calls = Counter()
    originals = {cls: cls.to_dict for cls in (Node, Flow, Workflow)}

    def counting(cls):
        def to_dict(self, *args, **kwargs):
            calls[cls.__name__] += 1
            return originals[cls](self, *args, **kwargs)

        return to_dict

    with (
        patch.object(Node, "to_dict", counting(Node)),
        patch.object(Flow, "to_dict", counting(Flow)),
        patch.object(Workflow, "to_dict", counting(Workflow)),
    ):
        yield calls


def run(workflow: Workflow, *callbacks):
    result = workflow.run_sync(input_data={"x": 1}, config=RunnableConfig(callbacks=list(callbacks)))
    assert result.status == RunnableStatus.SUCCESS, result.error
    return result


def test_hooks_a_callback_does_not_override_are_neither_called_nor_serialized_for(to_dict_calls):
    run(make_workflow(), BaseCallbackHandler())

    assert to_dict_calls == Counter()


def test_marked_hooks_share_one_serialization_of_each_node_flow_and_workflow_per_run():
    workflow = make_workflow()
    reader = Reader()

    run(workflow, reader)

    for entity in (workflow, workflow.flow, *workflow.flow.nodes):
        received = [s for (_, entity_id), seen in reader.seen.items() if entity_id == entity.id for s in seen]
        assert len(received) >= 2
        assert all(s is received[0] for s in received)
        assert received[0] == entity.to_dict()


def test_a_run_serializes_each_node_once_for_marked_hooks(to_dict_calls):
    run(make_workflow(), Reader())

    # Serializing the output node serializes its dependency, the input node, again: three node serializations each for
    # the workflow's hooks, the flow's hooks and the nodes' own. The workflow's serialization holds the flow's.
    assert to_dict_calls == Counter({"Workflow": 1, "Flow": 2, "Node": 9})


def test_unmarked_hooks_get_a_new_serialization_and_their_edits_reach_no_one_else():
    workflow = make_workflow()
    reader, editor = Reader(), Editor()

    run(workflow, editor, reader)

    assert len({id(s) for s in editor.seen}) == len(editor.seen) == 2 * len(workflow.flow.nodes)
    names = {s["name"] for (hook, _), seen in reader.seen.items() if hook in NODE_HOOKS for s in seen}
    assert names == {node.name for node in workflow.flow.nodes}


def test_an_unmarked_override_of_a_marked_hook_gets_a_new_serialization():
    class Override(Reader):
        def on_node_end(self, serialized, output_data, **kwargs):
            self._record("on_node_end", serialized)

    workflow = make_workflow()
    override = Override()

    run(workflow, override)

    node = workflow.flow.nodes[0]
    shared = override.seen[("on_node_execute_start", node.id)][0]
    assert override.seen[("on_node_end", node.id)][0] is not shared


def test_each_run_serializes_anew():
    workflow = make_workflow()
    reader = Reader()

    run(workflow, reader)
    run(workflow, reader)

    first, second = reader.seen[("on_workflow_start", workflow.id)]
    assert first is not second


def test_outside_a_run_marked_hooks_get_a_new_serialization():
    node = Input()
    reader = Reader()

    node.run_on_node_end([reader], {})
    node.run_on_node_end([reader], {})

    first, second = reader.seen[("on_node_end", node.id)]
    assert first is not second


@pytest.mark.asyncio
async def test_concurrent_runs_do_not_share_serializations():
    workflow = make_workflow()
    readers = [Reader(), Reader()]

    await asyncio.gather(
        *(workflow.run_async(input_data={"x": 1}, config=RunnableConfig(callbacks=[r])) for r in readers)
    )

    first, second = (r.seen[("on_workflow_start", workflow.id)][0] for r in readers)
    assert first is not second
    assert readers[0].seen[("on_workflow_end", workflow.id)][0] is first


def test_tracing_keeps_its_own_copy_of_each_node():
    workflow = make_workflow()
    tracing, reader = TracingCallbackHandler(), Reader()

    run(workflow, tracing, reader)

    node = workflow.flow.nodes[0]
    shared = reader.seen[("on_node_end", node.id)][0]
    (traced,) = (r.metadata["node"] for r in tracing.runs.values() if r.metadata.get("node", {}).get("id") == node.id)
    assert traced is not shared


@pytest.mark.parametrize(
    "handler, hook, marked",
    [
        *((TracingCallbackHandler, hook, True) for hook in NODE_HOOKS + ("on_node_execute_error",)),
        *(
            (TracingCallbackHandler, f"on_{entity}_{event}", True)
            for entity in ("workflow", "flow")
            for event in ("start", "end", "error", "canceled")
        ),
        # These keep the serialization in the trace and edit it later.
        *((TracingCallbackHandler, hook, False) for hook in ("on_node_start", "on_node_error", "on_node_skip")),
        (TracingCallbackHandler, "on_node_canceled", False),
        *(
            (StreamingQueueCallbackHandler, f"on_workflow_{event}", True)
            for event in ("start", "end", "error", "canceled")
        ),
        (AgentStreamingParserCallback, "on_node_execute_end", True),
        (_EvidenceRecorder, "on_node_end", True),
    ],
)
def test_dynamiq_handlers_mark_only_the_hooks_that_neither_edit_nor_keep_the_serialization(handler, hook, marked):
    assert getattr(getattr(handler, hook), READS_SERIALIZATION_ONLY, False) is marked
