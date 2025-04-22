from collections import defaultdict

import pytest

from dynamiq import Workflow, flows
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.streaming import STREAMING_EVENT, StreamingConfig, StreamingEntitySource


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
            events_output[e.entity_id].append((e.event, e.data["choices"][0]["delta"]["content"], e.source))
        else:
            final_wf_event_output = e.event, e.data, e.source

    expected_output = {
        node_with_streaming.id: RunnableResult(
            status=RunnableStatus.SUCCESS,
            input=input_data,
            output={"content": mock_llm_response_text},
        ).to_dict()
    }

    expected_final_output_source = StreamingEntitySource(name="Workflow", group=None, type="dynamiq.workflows.Workflow")

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS, input=input_data, output=expected_output
    )
    assert final_wf_event_output[0] == STREAMING_EVENT
    assert final_wf_event_output[1] == expected_output
    assert final_wf_event_output[2] == expected_final_output_source

    assert mock_llm_executor.call_count == 1
    node_output = events_output[node_with_streaming.id]
    assert "".join([content for _, content, _ in node_output]) == mock_llm_response_text
    assert all(event == streaming_custom_event for event, _, _ in node_output)
    expected_streaming_node_source = StreamingEntitySource(
        name="OpenAI", group="llms", type="dynamiq.nodes.llms.OpenAI"
    )
    assert all(source == expected_streaming_node_source for _, _, source in node_output)
