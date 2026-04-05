import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig


def make_mock_response(content="test response"):
    """Create a mock litellm ModelResponse."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    response = MagicMock()
    response.choices = [choice]
    return response


class TestBaseLLMAsync:
    def test_base_llm_has_native_async(self):
        """BaseLLM should report has_native_async=True after we add execute_async."""
        with patch("litellm.completion"), \
             patch("litellm.stream_chunk_builder"):
            node = OpenAI(
                model="gpt-4o-mini",
                connection=OpenAIConnection(api_key="test-key"),
            )
            assert node.has_native_async is True

    @pytest.mark.asyncio
    async def test_execute_async_calls_acompletion(self):
        """execute_async should call litellm.acompletion, not completion."""
        mock_response = make_mock_response("async response")

        with patch("litellm.completion"), \
             patch("litellm.stream_chunk_builder"):
            node = OpenAI(
                model="gpt-4o-mini",
                connection=OpenAIConnection(api_key="test-key"),
                prompt=Prompt(messages=[{"role": "user", "content": "Hello"}]),
            )
            node._acompletion = AsyncMock(return_value=mock_response)

            result = await node.execute_async(
                input_data=MagicMock(messages=None, files=None),
                config=RunnableConfig(callbacks=[]),
            )

            node._acompletion.assert_called_once()
            assert result["content"] == "async response"

    @pytest.mark.asyncio
    async def test_execute_async_streaming(self):
        """execute_async should handle streaming via async for."""
        chunk1 = MagicMock()
        chunk1.model_dump.return_value = {"choices": [{"delta": {"content": "hel"}}]}
        chunk2 = MagicMock()
        chunk2.model_dump.return_value = {"choices": [{"delta": {"content": "lo"}}]}

        async def async_chunk_iter():
            for chunk in [chunk1, chunk2]:
                yield chunk

        full_response = make_mock_response("hello")

        from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
        from dynamiq.types.streaming import StreamingConfig

        with patch("litellm.completion"), \
             patch("litellm.stream_chunk_builder"):
            node = OpenAI(
                model="gpt-4o-mini",
                connection=OpenAIConnection(api_key="test-key"),
                prompt=Prompt(messages=[{"role": "user", "content": "Hello"}]),
                streaming=StreamingConfig(enabled=True),
            )
            node._acompletion = AsyncMock(return_value=async_chunk_iter())
            node._stream_chunk_builder = MagicMock(return_value=full_response)

            streaming_handler = StreamingIteratorCallbackHandler()
            result = await node.execute_async(
                input_data=MagicMock(messages=None, files=None),
                config=RunnableConfig(callbacks=[streaming_handler]),
            )

            assert result["content"] == "hello"
            node._stream_chunk_builder.assert_called_once()
