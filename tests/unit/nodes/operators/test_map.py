import threading
import time
from queue import Queue
from threading import Event

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.operators import Map, Pass, SubWorkflow
from dynamiq.nodes.tools.human_feedback import (
    HFStreamingInputEventMessage,
    HFStreamingInputEventMessageData,
    HFStreamingOutputEventMessage,
    HumanFeedbackTool,
)
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.runnables.base import NodeRunnableConfig
from dynamiq.types.feedback import (
    ApprovalConfig,
    ApprovalInputData,
    ApprovalStreamingInputEventMessage,
    ApprovalStreamingOutputEventMessage,
    FeedbackMethod,
)
from dynamiq.types.streaming import StreamingConfig


class ReplyByAskingId(BaseCallbackHandler):
    """Answers each feedback request through the queue registered for the id it was asked under.

    That is what a client and the server routing its reply do: the reply carries the asking id, so a
    reply to an id nobody registered is lost. ``hold`` keeps each request open for a while, so requests
    made at the same time overlap and are counted in ``most_open``.
    """

    def __init__(self, queues: dict[str, Queue], hold: float = 0.0):
        self.queues = queues
        self.hold = hold
        self.asked_ids: list[str] = []
        self.most_open = 0
        self._open = 0
        self._lock = threading.Lock()

    def on_node_execute_stream(self, serialized, chunk=None, **kwargs):
        event = kwargs.get("event")
        if not isinstance(event, HFStreamingOutputEventMessage):
            return
        with self._lock:
            self.asked_ids.append(event.entity_id)
            self._open += 1
            self.most_open = max(self.most_open, self._open)
        time.sleep(self.hold)
        if queue := self.queues.get(event.entity_id):
            reply = HFStreamingInputEventMessage(
                entity_id=event.entity_id,
                data=HFStreamingInputEventMessageData(content=f"answer to {event.data.prompt}"),
                event=event.event,
            )
            queue.put(reply.model_dump_json())
        with self._lock:
            self._open -= 1


class Approve(BaseCallbackHandler):
    """Approves each approval request through ``queue``, or, if ``late``, just after the request has timed out."""

    def __init__(self, queue: Queue, late: bool = False):
        self.queue = queue
        self.late = late
        self.asked: list[dict] = []
        self._unanswered: list[str] = []

    def on_node_execute_stream(self, serialized, chunk=None, **kwargs):
        event = kwargs.get("event")
        if not isinstance(event, ApprovalStreamingOutputEventMessage):
            return
        self.asked.append(event.data.data)
        reply = ApprovalStreamingInputEventMessage(
            entity_id=event.entity_id, data=ApprovalInputData(feedback="", is_approved=True), event=event.event
        ).model_dump_json()
        if self.late:
            self._unanswered.append(reply)
        else:
            self.queue.put(reply)

    def on_node_error(self, serialized, error, **kwargs):
        while self._unanswered:
            self.queue.put(self._unanswered.pop())


def input_override(queue: Queue, timeout: float = 1) -> NodeRunnableConfig:
    """The override a server registers for a node so that it reads its reply from ``queue``."""
    return NodeRunnableConfig(
        streaming=StreamingConfig(enabled=True, input_queue=queue, input_queue_done_event=Event(), timeout=timeout)
    )


@pytest.mark.parametrize("in_sub_workflow", [False, True], ids=["direct", "in_sub_workflow"])
def test_a_feedback_tool_in_a_map_asks_under_its_registered_id_so_each_reply_reaches_it(in_sub_workflow):
    ask = HumanFeedbackTool(id="ask", input_method=FeedbackMethod.STREAM)
    node = SubWorkflow(id="sub", flow=Flow(id="inner", nodes=[ask])) if in_sub_workflow else ask
    queue = Queue()
    client = ReplyByAskingId({"ask": queue})
    config = RunnableConfig(callbacks=[client], nodes_override={"ask": input_override(queue)})

    result = Workflow(flow=Flow(nodes=[Map(id="map", node=node)])).run(
        input_data={"input": [{"input": "first?"}, {"input": "second?"}]}, config=config
    )

    # Each item's copy asked under the id its queue was registered for, so both replies arrived. A copy
    # that renamed the tool asked under a fresh id, and its reply was lost.
    assert client.asked_ids == ["ask", "ask"]
    assert result.status == RunnableStatus.SUCCESS
    first, second = (str(item) for item in result.output["map"]["output"]["output"])
    assert "answer to first?" in first and "answer to second?" in second


def test_the_items_of_a_parallel_map_ask_one_at_a_time_and_each_reads_its_own_reply():
    queue = Queue()
    client = ReplyByAskingId({"ask": queue}, hold=0.05)
    config = RunnableConfig(callbacks=[client], nodes_override={"ask": input_override(queue)})
    map_node = Map(id="map", node=HumanFeedbackTool(id="ask", input_method=FeedbackMethod.STREAM), max_workers=3)

    result = Workflow(flow=Flow(nodes=[map_node])).run(
        input_data={"input": [{"input": f"item {i}?"} for i in range(3)]}, config=config
    )

    # A reply names only the node it answers, so copies asking at the same time could each read another's
    # reply: each item asks once the request before it has its reply.
    assert client.most_open == 1
    outputs = [str(output) for output in result.output["map"]["output"]["output"]]
    assert all(f"answer to item {i}?" in output for i, output in enumerate(outputs))


def test_an_item_whose_timeout_passes_while_it_waits_for_its_turn_never_asks():
    queue = Queue()
    client = ReplyByAskingId({"ask": queue}, hold=1.5)
    config = RunnableConfig(callbacks=[client], nodes_override={"ask": input_override(queue, timeout=0.9)})
    ask = HumanFeedbackTool(
        id="ask", input_method=FeedbackMethod.STREAM, error_handling=ErrorHandling(timeout_seconds=1)
    )

    Workflow(flow=Flow(nodes=[Map(id="map", node=ask, max_workers=3)])).run(
        input_data={"input": [{"input": f"item {i}?"} for i in range(3)]}, config=config
    )
    time.sleep(1)

    # The first request holds the turn past every item's 1 s timeout. The other items time out waiting for it,
    # and their threads, which outlive the timeout, give up instead of asking for items already failed.
    assert client.asked_ids == ["ask"]


def test_the_items_of_a_map_that_never_ask_run_in_parallel():
    all_started = threading.Barrier(3, timeout=5)

    class Worker(Node):
        """Holds a feedback tool it never calls, as an agent with nothing to ask."""

        group: NodeGroup = NodeGroup.UTILS
        tool: HumanFeedbackTool

        def execute(self, input_data, config=None, **kwargs):
            all_started.wait()
            return {"done": input_data["item"]}

    # An Agent has a 3600 s timeout by default, so the worker has one too.
    worker = Worker(
        tool=HumanFeedbackTool(id="ask", input_method=FeedbackMethod.STREAM),
        error_handling=ErrorHandling(timeout_seconds=3600),
    )
    result = Workflow(flow=Flow(nodes=[Map(id="map", node=worker, max_workers=3)])).run(
        input_data={"input": [{"item": i} for i in range(3)]}
    )

    # Each item waits until all three have started, which only items running at the same time can do.
    assert result.output["map"]["output"]["output"] == [{"done": 0}, {"done": 1}, {"done": 2}]


@pytest.mark.parametrize("max_workers", [1, 2])
def test_an_approval_given_too_late_is_not_taken_for_another_items_request(max_workers):
    queue = Queue()
    refund = Pass(id="refund", approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.STREAM))
    workflow = Workflow(flow=Flow(nodes=[Map(id="map", node=refund, max_workers=max_workers)]))

    def run(client: Approve) -> list:
        config = RunnableConfig(callbacks=[client], nodes_override={"refund": input_override(queue, timeout=0.2)})
        result = workflow.run(input_data={"input": [{"refund": "A"}, {"refund": "B"}]}, config=config)
        return result.output["map"]["output"]["output"]

    late = Approve(queue, late=True)
    outputs = run(late)

    # The first request timed out, so no other item asked: the approval given late for it stays unread,
    # where the next request would have taken it for its own.
    assert len(late.asked) == 1
    assert outputs == [None, None]
    assert queue.qsize() == 1

    # A missed reply stops the asking only in the Map run that missed it.
    queue.get_nowait()
    assert run(Approve(queue)) == [{"refund": "A"}, {"refund": "B"}]


class ApproveByAskingId(BaseCallbackHandler):
    """Approves each approval request through the queue registered for the id it was asked under, echoing its
    request id, as a client answering an agent hook's approval does."""

    def __init__(self, queues: dict[str, Queue]):
        self.queues = queues
        self.asked_ids: list[str] = []

    def on_node_execute_stream(self, serialized, chunk=None, **kwargs):
        event = kwargs.get("event")
        if not isinstance(event, ApprovalStreamingOutputEventMessage):
            return
        self.asked_ids.append(event.entity_id)
        if queue := self.queues.get(event.entity_id):
            answer = ApprovalInputData(feedback="", is_approved=True, request_id=event.data.request_id)
            reply = ApprovalStreamingInputEventMessage(entity_id=event.entity_id, data=answer, event=event.event)
            queue.put(reply.model_dump_json())


def test_an_agent_whose_hook_asks_for_approvals_in_a_map_asks_under_its_registered_id():
    from dynamiq.nodes.agents.hooks import ToolPolicyHook
    from dynamiq.nodes.types import InferenceMode
    from tests.helpers.agent_hooks import TOOL_CALLS, build_agent, final_step, scripted_stateless, tool_step

    TOOL_CALLS.clear()
    queue = Queue()
    client = ApproveByAskingId({"agent": queue})
    config = RunnableConfig(callbacks=[client], nodes_override={"agent": input_override(queue)})
    agent = build_agent([ToolPolicyHook(tools=["search"], approval=True, feedback_method=FeedbackMethod.STREAM)])
    steps = [tool_step("search", query="q"), final_step("done")]

    with scripted_stateless(InferenceMode.STRUCTURED_OUTPUT, steps, agents=[agent]):
        result = Workflow(flow=Flow(nodes=[Map(id="map", node=agent, max_workers=2)])).run(
            input_data={"input": [{"input": "first"}, {"input": "second"}]}, config=config
        )

    # Each item's copy asked under the id its queue was registered for, so both approvals reached it.
    assert client.asked_ids == ["agent", "agent"]
    assert [output["content"] for output in result.output["map"]["output"]["output"]] == ["done", "done"]
    assert [name for name, _ in TOOL_CALLS] == ["search", "search"]
