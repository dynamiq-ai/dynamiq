"""The context estimate must count what the request carries, not just the messages.

`tools` and `response_format` are sent as request parameters, so a message-only count
misses them entirely — and compaction, which keys off that count, never fires.
"""

import pytest

from dynamiq import connections
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message, MessageRole


def _agent(mode, *, with_tools=True, **kwargs):
    tools = [TavilyTool(connection=connections.Tavily(api_key="not-used"))] if with_tools else []
    agent = Agent(
        name="agent",
        llm=llms.OpenAI(
            model="gpt-4o-mini",
            connection=connections.OpenAI(api_key="not-used"),
            is_postponed_component_init=True,
        ),
        tools=tools,
        inference_mode=mode,
        **kwargs,
    )
    agent._prompt.messages = [Message(role=MessageRole.USER, content="find something", static=True)]
    return agent


class TestRequestParametersAreCounted:
    def test_function_calling_counts_tool_schemas(self):
        """The schemas ride on `tools=`, never in the messages."""
        tokens = _agent(InferenceMode.FUNCTION_CALLING).count_context_tokens()

        assert tokens["tools"] > 0
        assert tokens["total"] == tokens["messages"] + tokens["tools"] + tokens["response_format"]
        assert tokens["total"] > tokens["messages"], "a message-only count misses the schemas"

    def test_structured_output_counts_the_response_schema(self):
        """The per-step schema rides on `response_format=`."""
        tokens = _agent(InferenceMode.STRUCTURED_OUTPUT).count_context_tokens()

        assert tokens["response_format"] > 0
        assert tokens["total"] > tokens["messages"]

    @pytest.mark.parametrize("mode", [InferenceMode.XML, InferenceMode.DEFAULT])
    def test_prompt_based_modes_are_unchanged(self, mode):
        """These render tool descriptions into the system prompt, so they were already
        counted — adding request parameters must not double-count them."""
        tokens = _agent(mode).count_context_tokens()

        assert tokens["tools"] == 0
        assert tokens["response_format"] == 0
        assert tokens["total"] == tokens["messages"]

    def test_the_injected_final_answer_schema_costs_tokens_too(self):
        """FUNCTION_CALLING always ships `provide_final_answer`, so even a tool-less agent
        sends a schema — and adding a real tool costs more on top."""
        bare = _agent(InferenceMode.FUNCTION_CALLING, with_tools=False).count_context_tokens()
        with_tool = _agent(InferenceMode.FUNCTION_CALLING).count_context_tokens()

        assert bare["tools"] > 0
        assert with_tool["tools"] > bare["tools"]


class TestTokenLimitUsesTheFullRequest:
    def test_schemas_can_be_what_pushes_it_over(self):
        """The case the old count missed: messages fit, the request does not."""
        agent = _agent(InferenceMode.FUNCTION_CALLING)
        tokens = agent.count_context_tokens()
        assert tokens["tools"] > 0

        # a ceiling between the two: messages alone are under it, the whole request is over
        ceiling = (tokens["messages"] + tokens["total"]) // 2
        agent.summarization_config = SummarizationConfig(enabled=True, max_token_context_length=ceiling)

        assert tokens["messages"] < ceiling < tokens["total"]
        assert agent.is_token_limit_exceeded(), "compaction must fire on the real request size"

    def test_headroom_is_still_headroom(self):
        agent = _agent(InferenceMode.FUNCTION_CALLING)
        total = agent.count_context_tokens()["total"]
        agent.summarization_config = SummarizationConfig(enabled=True, max_token_context_length=total * 10)

        assert not agent.is_token_limit_exceeded()
