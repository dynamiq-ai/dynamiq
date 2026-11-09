"""Context compaction inside a running agent."""

from unittest.mock import MagicMock, patch

import pytest

from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableResult, RunnableStatus
from tests.helpers.agent_hooks import SearchTool, build_agent, encode, final_step, parallel_step, tool_step


def run_function_calling(steps, max_preserved_tokens: int):
    """Run a natively built FUNCTION_CALLING agent; return the result and every agent request's messages."""
    agent = build_agent(
        tools=[SearchTool()],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        summarization_config=SummarizationConfig(enabled=True, max_preserved_tokens=max_preserved_tokens),
    )
    requests: list[list] = []

    def run(**kwargs):
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        if not kwargs.get("tools"):  # the summarizer, which is sent no tool schemas
            result.output = {"content": "SUMMARY: searched for x."}
            return result
        requests.append(list(kwargs["prompt"].messages))
        index = len(requests) - 1
        result.output = encode(InferenceMode.FUNCTION_CALLING, steps[min(index, len(steps) - 1)], index)
        return result

    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": "hello"})
    return result, requests


def orphan_tool_replies(messages) -> list[str]:
    """Tool replies whose call no earlier assistant message carries: the provider would reject them."""
    announced: set[str] = set()
    orphans: list[str] = []
    for message in messages:
        if message.role == MessageRole.ASSISTANT:
            announced.update(call["id"] for call in message.tool_calls or [])
        elif message.role == MessageRole.TOOL and message.tool_call_id not in announced:
            orphans.append(message.tool_call_id)
    return orphans


@pytest.mark.parametrize(
    "compaction_step",
    [
        tool_step("context-manager", notes="order 42"),
        parallel_step(("search", {"query": "y"}), ("context-manager", {"notes": "order 42"})),
    ],
    ids=["alone", "in-a-batch"],
)
def test_a_model_called_compaction_keeps_the_call_its_result_answers(compaction_step):
    # With nothing preserved, the message carrying the call itself was summarized away and its replies,
    # among them the notice that the batched search did not run, were dropped before the next request.
    steps = [tool_step("search", query="x"), compaction_step, final_step("done")]

    result, requests = run_function_calling(steps, max_preserved_tokens=0)

    assert result.status == RunnableStatus.SUCCESS
    after_compaction = requests[2]
    assert orphan_tool_replies(after_compaction) == []
    replies = [message.content for message in after_compaction if message.role == MessageRole.TOOL]
    assert replies and all("summarized" in reply for reply in replies)


def run_with_summarizer(mode, steps, summary: str, **summarization):
    """Run with compaction enabled; return the result, the agent's requests and how often the summarizer ran."""
    agent = build_agent(
        tools=[SearchTool(result="order line item shipped " * 300)],
        inference_mode=mode,
        summarization_config=SummarizationConfig(enabled=True, **summarization),
    )
    requests: list[list] = []
    summaries = {"calls": 0}

    def run(**kwargs):
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        last = str(getattr(kwargs["prompt"].messages[-1], "content", ""))
        if "summary of the conversation history" in last or "Merge them into a single" in last:
            summaries["calls"] += 1
            result.output = {"content": summary}
            return result
        requests.append(list(kwargs["prompt"].messages))
        index = len(requests) - 1
        result.output = encode(mode, steps[min(index, len(steps) - 1)], index)
        return result

    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": "Find the order total."})
    return result, requests, summaries["calls"]


@pytest.mark.parametrize("mode", [InferenceMode.FUNCTION_CALLING, InferenceMode.XML])
def test_a_summarizer_that_returns_nothing_falls_back_to_dropping_old_messages(mode):
    # Reasoning models and safety filters can return an empty summary; that must not end the run.
    steps = [tool_step("search", query="a"), tool_step("search", query="b"), final_step("done")]

    result, requests, summary_calls = run_with_summarizer(
        mode, steps, summary="", max_token_context_length=3000, max_preserved_tokens=0
    )

    assert result.status == RunnableStatus.SUCCESS
    assert summary_calls >= 3
    after = "\n".join(str(message.content) for message in requests[-1])
    assert "no summary could be generated" in after
    assert "Find the order total." in after


def test_old_history_is_still_summarized_when_the_system_prompt_alone_exceeds_the_limit():
    # No summary can get under a limit the fixed prompt already exceeds, but the history must stay bounded:
    # what exceeds max_preserved_tokens is summarized, and the whole history is not summarized every loop.
    steps = [
        tool_step("search", query="a"),
        tool_step("search", query="b"),
        tool_step("search", query="c"),
        final_step("done"),
    ]

    result, requests, summary_calls = run_with_summarizer(
        InferenceMode.FUNCTION_CALLING,
        steps,
        summary="SUMMARY",
        max_token_context_length=50,
        max_preserved_tokens=2000,
    )

    assert result.status == RunnableStatus.SUCCESS
    assert 1 <= summary_calls < len(requests) - 1
    assert "SUMMARY" in "\n".join(str(message.content) for message in requests[-1])
    kept_results = [message for message in requests[-1] if "order line item shipped" in str(message.content)]
    assert len(kept_results) < 3


def test_a_compaction_batched_with_a_final_answer_keeps_the_call_its_replies_answer():
    steps = [
        tool_step("search", query="x"),
        parallel_step(("context-manager", {"notes": "order 42"}), ("provide_final_answer", {"answer": "early"})),
        final_step("done"),
    ]

    result, requests = run_function_calling(steps, max_preserved_tokens=0)

    assert result.status == RunnableStatus.SUCCESS
    assert orphan_tool_replies(requests[2]) == []
