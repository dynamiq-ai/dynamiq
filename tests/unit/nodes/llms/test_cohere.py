from unittest.mock import patch

from dynamiq.connections import Cohere as CohereConnection
from dynamiq.nodes.llms import Cohere
from dynamiq.nodes.llms.base import BaseLLM


def test_messages_drop_the_fields_cohere_rejects():
    tool_call = {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
    messages = [
        {"role": "user", "content": "Find it."},
        {"role": "assistant", "content": "Calling: search", "tool_calls": [tool_call]},
        {"role": "tool", "content": "found", "tool_call_id": "call_1", "name": "search"},
        {"role": "assistant", "content": "Here it is."},
    ]
    llm = Cohere(connection=CohereConnection(api_key="test"), model="cohere/command-a-03-2025")

    with patch.object(BaseLLM, "get_messages", return_value=messages):
        sent = llm.get_messages(prompt=None, input_data={})

    assert sent == [
        {"role": "user", "content": "Find it."},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "tool", "content": "found", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "Here it is."},
    ]


def test_streamed_tool_calls_are_read_from_cohere_v2_events():
    from litellm.llms.cohere.common_utils import CohereV2ModelResponseIterator

    from dynamiq.nodes.llms.cohere import _patch_litellm_cohere_streamed_tool_calls

    assert _patch_litellm_cohere_streamed_tool_calls()
    parser = CohereV2ModelResponseIterator(streaming_response=iter([]), sync_stream=True)
    start = {
        "type": "tool-call-start",
        "index": 1,
        "delta": {
            "message": {
                "tool_calls": {"id": "lookup_1", "type": "function", "function": {"name": "lookup", "arguments": ""}}
            }
        },
    }
    delta = {
        "type": "tool-call-delta",
        "index": 1,
        "delta": {"message": {"tool_calls": {"function": {"arguments": '{"q": "A"}'}}}},
    }

    assert parser.chunk_parser(start)["tool_use"] == {
        "id": "lookup_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": ""},
        "index": 1,
    }
    assert parser.chunk_parser(delta)["tool_use"]["function"]["arguments"] == '{"q": "A"}'
    assert (
        parser.chunk_parser({"type": "content-delta", "delta": {"message": {"content": {"text": "hi"}}}})["text"]
        == "hi"
    )
