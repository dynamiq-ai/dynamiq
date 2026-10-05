from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.llms.base import LLMContentFilteredError
from dynamiq.nodes.llms.bedrock import Bedrock
from dynamiq.runnables import RunnableConfig


def _mock_response(content, finish_reason, tool_calls=None):
    """Minimal litellm ModelResponse stand-in for _handle_completion_response."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    response.model_extra = {}
    usage = MagicMock()
    usage.prompt_tokens = 6000
    usage.completion_tokens = 5
    usage.total_tokens = 6005
    usage.prompt_tokens_details = None
    usage.cache_read_input_tokens = usage.cache_creation_input_tokens = None
    response.usage = usage
    return response


@pytest.fixture
def llm():
    return Bedrock(
        name="bedrock",
        model="bedrock/us.anthropic.claude-opus-5-5",
        connection=AWSConnection(access_key_id="x", secret_access_key="x", region="us-east-1"),
    )


def test_empty_content_filtered_response_raises(llm):
    with pytest.raises(LLMContentFilteredError, match="content filter"):
        llm._handle_completion_response(_mock_response("", "content_filter"), config=RunnableConfig(callbacks=[]))


def test_none_content_filtered_response_raises(llm):
    with pytest.raises(LLMContentFilteredError):
        llm._handle_completion_response(_mock_response(None, "content_filter"), config=RunnableConfig(callbacks=[]))


def test_filtered_response_still_reports_usage(llm):
    with patch.object(Bedrock, "run_on_node_execute_run") as report_usage, pytest.raises(LLMContentFilteredError):
        llm._handle_completion_response(_mock_response("", "content_filter"), config=RunnableConfig(callbacks=[]))

    report_usage.assert_called_once()
    assert report_usage.call_args.kwargs["usage_data"]["completion_tokens"] == 5


def test_partially_filtered_response_returns_content(llm):
    result = llm._handle_completion_response(
        _mock_response("Hello! How can", "content_filter"), config=RunnableConfig(callbacks=[])
    )

    assert result["content"] == "Hello! How can"


def test_filtered_response_with_tool_calls_returns_tool_calls(llm):
    tool_call = MagicMock()
    tool_call.model_dump.return_value = {"id": "1", "function": {"name": "provide_final_answer", "arguments": "{}"}}

    result = llm._handle_completion_response(
        _mock_response("", "content_filter", tool_calls=[tool_call]), config=RunnableConfig(callbacks=[])
    )

    assert result["tool_calls"][0]["function"]["name"] == "provide_final_answer"


def test_empty_response_with_normal_stop_is_returned(llm):
    result = llm._handle_completion_response(_mock_response("", "stop"), config=RunnableConfig(callbacks=[]))

    assert result["content"] == ""
