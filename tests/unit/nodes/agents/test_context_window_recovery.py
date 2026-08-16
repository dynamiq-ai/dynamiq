"""A prompt the provider rejects as too long should be compacted and retried, not fatal.

The proactive size check runs on a local token estimate, which can disagree with the
provider — tool schemas and images are easy to under-count. When it does, the request is
rejected and every completed loop of the run used to be lost.

Recovery is deliberately narrow: it only engages when summarization is already enabled,
it retries once, and it gives up immediately if compaction did not actually shorten the
history.
"""

import json
from typing import ClassVar
from unittest.mock import patch

import pytest
from litellm import ModelResponse
from litellm.exceptions import ContextWindowExceededError, RateLimitError
from pydantic import BaseModel, Field

from dynamiq import connections
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.exceptions import LLMContextWindowExceededError
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableStatus

OVERFLOW_MESSAGE = (
    "litellm.ContextWindowExceededError: OpenAIException - This model's maximum context "
    "length is 128000 tokens. However, your messages resulted in 191204 tokens."
)


def _final_answer_response():
    return ModelResponse(
        choices=[
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_final",
                            "type": "function",
                            "function": {
                                "name": "provide_final_answer",
                                "arguments": json.dumps({"thought": "done", "answer": "recovered"}),
                            },
                        }
                    ],
                }
            }
        ]
    )


def _overflow_error():
    return ContextWindowExceededError(message=OVERFLOW_MESSAGE, model="gpt-4o-mini", llm_provider="openai")


class _EchoInput(BaseModel):
    text: str = Field(default="")


class EchoTool(Node):
    group: NodeGroup = NodeGroup.TOOLS
    name: str = "echo"
    description: str = "returns its input"
    input_schema: ClassVar[type[BaseModel]] = _EchoInput

    def execute(self, input_data, config=None, **kwargs):
        return {"content": f"echo: {input_data.text}"}


def _build_agent(summarization_enabled: bool, with_tool: bool = False) -> Agent:
    return Agent(
        id="agent",
        name="agent",
        llm=llms.OpenAI(
            id="llm",
            model="gpt-4o-mini",
            connection=connections.OpenAI(api_key="not-used"),
            is_postponed_component_init=True,
        ),
        tools=[EchoTool()] if with_tool else [],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        # A small preserve budget makes compaction actually reclaim something from a
        # short test history, the way a real budget does from a long one.
        summarization_config=SummarizationConfig(enabled=summarization_enabled, max_preserved_tokens=10),
    )


def _echo_call(i: int):
    return ModelResponse(
        choices=[
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": json.dumps({"thought": f"t{i}", "text": "x" * 200}),
                            },
                        }
                    ],
                }
            }
        ]
    )


class _Script:
    """Raise the overflow error `fail_times` times, then answer normally."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, stream=False, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _overflow_error()
        return _final_answer_response()


class _HistoryThenOverflow:
    """Build real history with tool calls, then overflow, then answer.

    Mirrors production: the prompt only becomes too long after a few loops, which is
    also the only situation in which compaction has anything to reclaim.
    """

    def __init__(self, warmup: int = 3, fail_times: int = 1):
        self.warmup = warmup
        self.fail_times = fail_times
        self.calls = 0
        self.failures = 0
        self._summary_due = False

    def __call__(self, stream=False, *args, **kwargs):
        self.calls += 1
        if self._summary_due:
            # The compaction the overflow triggered asks for prose, not a tool call.
            self._summary_due = False
            return ModelResponse(choices=[{"message": {"content": "Summary of earlier steps."}}])
        if self.calls <= self.warmup:
            return _echo_call(self.calls)
        if self.failures < self.fail_times:
            self.failures += 1
            self._summary_due = True
            raise _overflow_error()
        return _final_answer_response()


class TestRecoveryWhenSummarizationEnabled:
    def test_run_survives_a_context_overflow(self):
        agent = _build_agent(summarization_enabled=True, with_tool=True)
        script = _HistoryThenOverflow(warmup=3, fail_times=1)

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=script):
            result = agent.run(input_data={"input": "go"})

        assert result.status == RunnableStatus.SUCCESS
        assert result.output["content"] == "recovered"
        assert script.failures == 1, "the overflow must actually have been hit"

    def test_the_request_is_retried_after_compaction(self):
        agent = _build_agent(summarization_enabled=True, with_tool=True)
        script = _HistoryThenOverflow(warmup=3, fail_times=1)

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=script):
            agent.run(input_data={"input": "go"})

        # 3 warmup turns, the rejected request, a summarization call, then the retry
        assert script.calls > script.warmup + 1, "the failed request must be re-issued"

    def test_overflow_does_not_consume_a_loop(self):
        """The rejected step produced nothing, so it must not count against max_loops.

        max_loops is set to exactly the number of productive steps the script needs
        (3 tool calls + the final answer), so if the rejected attempt consumed one the
        run would hit the limit instead of answering.
        """
        agent = _build_agent(summarization_enabled=True, with_tool=True)
        agent.max_loops = 4
        script = _HistoryThenOverflow(warmup=3, fail_times=1)

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=script):
            result = agent.run(input_data={"input": "go"})

        assert result.status == RunnableStatus.SUCCESS
        assert result.output["content"] == "recovered"
        assert script.failures == 1

    def test_history_is_compacted_before_the_retry(self):
        agent = _build_agent(summarization_enabled=True)
        script = _Script(fail_times=1)

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=script):
            with patch.object(Agent, "_try_summarize_history", autospec=True, return_value=True) as compact:
                agent.run(input_data={"input": "go"})

        forced = [c for c in compact.call_args_list if c.kwargs.get("force")]
        assert forced, "recovery must force compaction, ignoring the local estimate"

    def test_retry_happens_only_once(self):
        """A prompt that stays too long must fail rather than loop.

        The call count is the assertion that matters: a 'retry until it works' loop
        would still end in FAILURE while burning provider calls the whole way.
        """
        agent = _build_agent(summarization_enabled=True)
        script = _Script(fail_times=99)

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=script):
            with patch.object(Agent, "_try_summarize_history", autospec=True, return_value=True):
                result = agent.run(input_data={"input": "go"})

        assert result.status == RunnableStatus.FAILURE
        assert script.calls == 2, f"expected the original request plus one retry, got {script.calls}"

    def test_compaction_failure_reports_the_original_overflow(self):
        """The summarizer failing must not bury the error that actually stopped the run."""
        agent = _build_agent(summarization_enabled=True)
        script = _Script(fail_times=99)

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=script):
            with patch.object(
                Agent, "_try_summarize_history", autospec=True, side_effect=RuntimeError("summarizer exploded")
            ):
                result = agent.run(input_data={"input": "go"})

        assert result.status == RunnableStatus.FAILURE
        assert result.error.type is LLMContextWindowExceededError
        assert "summarizer exploded" not in result.error.message

    def test_force_summarizes_even_when_the_local_estimate_says_it_fits(self):
        """The whole point of recovery is that the local estimate was wrong. If the split
        still trusted it, forced compaction would reclaim nothing exactly when needed."""
        from dynamiq.prompts import Message, MessageRole

        agent = _build_agent(summarization_enabled=True)
        # Default preserve budget, so a short history is fully preserved and the
        # unforced split has nothing to hand over.
        agent.summarization_config = SummarizationConfig(enabled=True)
        agent._prompt = Prompt(
            messages=[
                Message(role=MessageRole.SYSTEM, content="s"),
                Message(role=MessageRole.USER, content="hello"),
                Message(role=MessageRole.ASSISTANT, content="short reply"),
            ]
        )
        agent._history_offset = 1

        # Estimate says the history fits — which is the condition under which the
        # unforced split preserves everything and summarizes nothing.
        assert agent.is_token_limit_exceeded() is False

        unforced, _ = agent._split_history()
        forced, _ = agent._split_history(force=True)

        assert unforced == [], "sanity: without force the estimate keeps everything"
        assert forced, "force must offer the history up for summarization"

    def test_effectiveness_is_measured_by_size_not_message_count(self):
        """Compaction can replace several huge messages with one long summary, leaving
        the message count flat or higher while the prompt shrinks dramatically. Judging
        by count would refuse to retry after a large genuine reduction."""
        from dynamiq.prompts import Message, MessageRole

        agent = _build_agent(summarization_enabled=True)
        agent._prompt = Prompt(
            messages=[
                Message(role=MessageRole.SYSTEM, content="s"),
                Message(role=MessageRole.USER, content="x" * 50_000),
                Message(role=MessageRole.ASSISTANT, content="y" * 50_000),
            ]
        )
        before = agent._history_size()

        # Same message count, far less content.
        agent._prompt.messages = [
            Message(role=MessageRole.SYSTEM, content="s"),
            Message(role=MessageRole.USER, content="summary"),
            Message(role=MessageRole.ASSISTANT, content="tail"),
        ]

        assert len(agent._prompt.messages) == 3, "message count deliberately unchanged"
        assert agent._history_size() < before

    def test_no_retry_when_compaction_cannot_shrink_history(self):
        """Re-sending an unchanged prompt would just burn another request."""
        agent = _build_agent(summarization_enabled=True)
        script = _Script(fail_times=99)

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=script):
            with patch.object(Agent, "_try_summarize_history", autospec=True, return_value=False):
                result = agent.run(input_data={"input": "go"})

        assert result.status == RunnableStatus.FAILURE
        assert script.calls == 1, "no retry may be issued when nothing was reclaimed"


class TestNoRecoveryWhenSummarizationDisabled:
    def test_overflow_fails_the_run(self):
        agent = _build_agent(summarization_enabled=False)
        script = _Script(fail_times=99)

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=script):
            result = agent.run(input_data={"input": "go"})

        assert result.status == RunnableStatus.FAILURE

    def test_error_is_typed(self):
        agent = _build_agent(summarization_enabled=False)
        script = _Script(fail_times=99)

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=script):
            result = agent.run(input_data={"input": "go"})

        assert result.error.type is LLMContextWindowExceededError

    def test_typed_error_stays_backwards_compatible(self):
        """Callers already catching ValueError must keep working."""
        assert issubclass(LLMContextWindowExceededError, ValueError)


class TestOverflowClassification:
    @pytest.fixture
    def llm(self):
        return llms.OpenAI(
            id="llm",
            model="gpt-4o-mini",
            connection=connections.OpenAI(api_key="not-used"),
            is_postponed_component_init=True,
        )

    def test_typed_litellm_error_is_recognised(self, llm):
        assert llm._is_context_window_error(ContextWindowExceededError, "") is True

    @pytest.mark.parametrize(
        "message",
        [
            "this model's maximum context length is 8192 tokens",
            "error code: 400 - context_length_exceeded",
            "prompt is too long: 210000 tokens > 200000 maximum",
            "please reduce the length of the messages",
            "input length and `max_tokens` exceed context limit",
        ],
    )
    def test_untyped_providers_are_matched_on_message(self, llm, message):
        assert llm._is_context_window_error(ValueError, message) is True

    @pytest.mark.parametrize(
        "exc,message",
        [
            (RateLimitError, "rate limit reached for gpt-4o-mini"),
            (ValueError, "connection reset by peer"),
            (ValueError, "invalid api key provided"),
            (ValueError, ""),
        ],
    )
    def test_unrelated_failures_are_not_misclassified(self, llm, exc, message):
        assert llm._is_context_window_error(exc, message) is False

    @pytest.mark.parametrize(
        "message",
        [
            "rate limit reached for gpt-4o-mini",
            "429 too many requests: you have sent too many tokens this minute, "
            "please reduce the length of your requests",
            "rate_limit_exceeded: request too large, limit 30000 tokens per min",
            "you have exceeded your token quota; the input length is being throttled",
        ],
    )
    def test_a_typed_rate_limit_outranks_its_message(self, llm, message):
        """Provider rate-limit text talks about tokens and length too.

        Matching on those words classified a throttle as an over-long prompt, which
        compacts history that was never the problem and retries into the same limit.
        The type is unambiguous, so it decides before the message is read.
        """
        assert llm._is_context_window_error(RateLimitError, message) is False

    def test_rate_limit_does_not_trigger_compaction(self):
        """Waiting fixes a rate limit; shrinking the prompt does not."""
        agent = _build_agent(summarization_enabled=True)

        def rate_limited(stream=False, *args, **kwargs):
            raise RateLimitError(message="rate limit reached", model="gpt-4o-mini", llm_provider="openai")

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=rate_limited):
            with patch.object(Agent, "_try_summarize_history", autospec=True, return_value=True) as compact:
                agent.run(input_data={"input": "go"})

        assert not [c for c in compact.call_args_list if c.kwargs.get("force")]
