from collections import defaultdict

import pytest

from dynamiq import Workflow, flows
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.streaming import STREAMING_EVENT, StreamingConfig


@pytest.fixture()
def streaming_custom_event():
    return "streaming_custom_event"


@pytest.fixture()
def node_with_streaming(openai_node, streaming_custom_event):
    openai_node.streaming = StreamingConfig(
        enabled=True,
        event=streaming_custom_event,
    )
    return openai_node


def test_node_streaming(
    node_with_streaming,
    streaming_custom_event,
    mock_llm_response_text,
    mock_llm_executor,
):
    streaming = StreamingIteratorCallbackHandler()
    input_data = {"a": 1, "b": 2}
    wf = Workflow(flow=flows.Flow(nodes=[node_with_streaming]))
    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[streaming]),
    )
    events_output = defaultdict(list)
    final_wf_event_output = None
    for e in streaming:
        if e.entity_id != wf.id:
            events_output[e.entity_id].append((e.event, e.data["choices"][0]["delta"]["content"]))
        else:
            final_wf_event_output = e.event, e.data

    expected_output = {
        node_with_streaming.id: RunnableResult(
            status=RunnableStatus.SUCCESS,
            input=input_data,
            output={"content": mock_llm_response_text},
        ).to_dict()
    }
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS, input=input_data, output=expected_output
    )
    assert final_wf_event_output[0] == STREAMING_EVENT
    assert final_wf_event_output[1] == expected_output
    assert mock_llm_executor.call_count == 1
    node_output = events_output[node_with_streaming.id]
    assert (
        "".join([content for event, content in node_output]) == mock_llm_response_text
    )
    assert all(event == streaming_custom_event for event, content in node_output)
