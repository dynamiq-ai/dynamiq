import threading
import time
from queue import Queue
from threading import Event

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.operators import Map, SubWorkflow
from dynamiq.nodes.tools.human_feedback import (
    HFStreamingInputEventMessage,
    HFStreamingInputEventMessageData,
    HFStreamingOutputEventMessage,
    HumanFeedbackTool,
)
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.runnables.base import NodeRunnableConfig
from dynamiq.types.feedback import FeedbackMethod
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


def input_override(queue: Queue) -> NodeRunnableConfig:
    """The override a server registers for a node so that it reads its reply from ``queue``."""
    return NodeRunnableConfig(
        streaming=StreamingConfig(enabled=True, input_queue=queue, input_queue_done_event=Event(), timeout=1)
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
    output = str(result.output)
    assert "answer to first?" in output and "answer to second?" in output


def test_a_map_holding_a_feedback_tool_runs_its_items_one_at_a_time():
    queue = Queue()
    client = ReplyByAskingId({"ask": queue}, hold=0.05)
    config = RunnableConfig(callbacks=[client], nodes_override={"ask": input_override(queue)})
    map_node = Map(id="map", node=HumanFeedbackTool(id="ask", input_method=FeedbackMethod.STREAM), max_workers=3)

    Workflow(flow=Flow(nodes=[map_node])).run(
        input_data={"input": [{"input": f"item {i}?"} for i in range(3)]}, config=config
    )

    # Copies asking under one id at the same time could each read the other's reply, so a Map holding a
    # node that waits for a reply asks one item at a time whatever its max_workers.
    assert client.asked_ids == ["ask"] * 3
    assert client.most_open == 1
