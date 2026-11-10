from unittest.mock import patch

import litellm
from litellm.types.llms.openai import ResponsesAPIResponse

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.llms.openai import ReasoningEffort
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableStatus

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look a policy up.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]

REPLY = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1760000000,
    "status": "completed",
    "model": "gpt-5.4",
    "output": [
        {"id": "rs_1", "type": "reasoning", "summary": []},
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Looking it up.", "annotations": []}],
        },
        {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"query": "travel policy"}',
            "status": "completed",
        },
    ],
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


def test_a_bridged_reply_with_text_keeps_its_tool_call():
    """LiteLLM serves GPT-5.4+ calls with function tools and a reasoning effort through the Responses API.

    Up to 1.103.2 its conversion put such a reply's tool calls in a second choice, which the node never reads.
    """
    llm = OpenAI(
        model="gpt-5.4", connection=OpenAIConnection(api_key="sk-test"), reasoning_effort=ReasoningEffort.MEDIUM
    )
    prompt = Prompt(messages=[Message(role="user", content="Find the travel policy.")])

    with patch.object(litellm, "responses", return_value=ResponsesAPIResponse(**REPLY)) as responses:
        result = llm.run(input_data={}, prompt=prompt, tools=TOOLS)

    assert responses.called
    assert result.status == RunnableStatus.SUCCESS, result.error
    assert [call["function"]["name"] for call in result.output["tool_calls"]] == ["lookup"]
    assert result.output["tool_calls"][0]["function"]["arguments"] == {"query": "travel policy"}
