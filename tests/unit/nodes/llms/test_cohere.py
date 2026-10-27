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
