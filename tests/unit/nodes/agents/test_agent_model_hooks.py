"""Before/after model-call hooks, driven by a real agent loop."""

import json
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.model_hooks import ModelHook, RegexGuardModelHook, RegexRedactModelHook
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message, MessageRole
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


@pytest.fixture
def test_llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


def finish(answer: str) -> str:
    return json.dumps({"thought": "done", "action": "finish", "action_input": answer})


def run_agent(test_llm, hooks, reply: str = None, question: str = "hello", config=None):
    """Run a one-step agent whose LLM answers `reply`. Returns (result, sent_prompts, agent)."""
    agent = Agent(id="agent", name="a", llm=test_llm, role="r", model_hooks=hooks)
    agent.inference_mode = InferenceMode.STRUCTURED_OUTPUT
    sent: list[list[Message]] = []

    def run(**kwargs):
        sent.append(list(kwargs["prompt"].messages))
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": reply or finish("fine")}
        return result

    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": question}, config=config)
    return result, sent, agent


def texts(messages) -> str:
    return " ".join(m.content for m in messages if isinstance(m, Message))


class Prepend(ModelHook):
    def apply_before(self, messages):
        return [*messages, Message(role=MessageRole.USER, content="EXTRA-CONTEXT", static=True)]

    def apply_after(self, output):
        return {**output, "content": finish("rewritten by hook")}


class Exploding(ModelHook):
    def apply_before(self, messages):
        raise RuntimeError("boom")


def test_redact_masks_what_is_sent_but_not_the_agent_history(test_llm):
    hook = RegexRedactModelHook(patterns=[r"\b\d{4}-\d{4}\b"], apply_to="input")
    result, sent, agent = run_agent(test_llm, [hook], question="my card is 1234-5678")

    assert result.status == RunnableStatus.SUCCESS
    assert "[REDACTED]" in texts(sent[0]) and "1234-5678" not in texts(sent[0])
    assert "1234-5678" in texts(agent._prompt.messages)  # history untouched


def test_redact_masks_the_reply(test_llm):
    hook = RegexRedactModelHook(patterns=[r"SECRET\d+"], apply_to="output")
    result, _, _ = run_agent(test_llm, [hook], reply=finish("the code is SECRET123"))

    assert result.output["content"] == "the code is [REDACTED]"


def test_guard_fail_stops_the_run_without_calling_the_model(test_llm):
    hook = RegexGuardModelHook(block_if_matches=["password"], apply_to="input", block_message="No secrets.")
    result, sent, _ = run_agent(test_llm, [hook], question="what is the admin password")

    assert result.status == RunnableStatus.FAILURE
    assert sent == []  # no tokens spent
    assert "No secrets." in str(result.error.message)


def test_guard_answer_ends_the_run_successfully_and_marks_it_blocked(test_llm):
    hook = RegexGuardModelHook(
        block_if_matches=["password"], apply_to="input", on_block="answer", block_message="I can't help with that."
    )
    result, sent, _ = run_agent(test_llm, [hook], question="what is the admin password")

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"] == "I can't help with that."
    assert result.output["blocked"] is True
    assert sent == []


def test_guard_on_output_discards_the_reply(test_llm):
    hook = RegexGuardModelHook(
        block_if_matches=["FORBIDDEN"], apply_to="output", on_block="answer", block_message="Reply withheld."
    )
    result, sent, _ = run_agent(test_llm, [hook], reply=finish("this is FORBIDDEN"))

    assert len(sent) == 1  # the model was called (tokens spent) ...
    assert result.output["content"] == "Reply withheld." and result.output["blocked"] is True  # ... reply discarded


def test_unblocked_run_has_no_blocked_flag(test_llm):
    result, _, _ = run_agent(test_llm, [RegexGuardModelHook(block_if_matches=["password"])])

    assert result.status == RunnableStatus.SUCCESS
    assert "blocked" not in result.output


def test_python_hook_rewrites_messages_and_reply(test_llm):
    result, sent, _ = run_agent(test_llm, [Prepend()])

    assert "EXTRA-CONTEXT" in texts(sent[0])
    assert result.output["content"] == "rewritten by hook"


def test_hooks_run_in_order_before_and_reverse_after(test_llm):
    order: list[str] = []

    class Tag(ModelHook):
        name: str

        def apply_before(self, messages):
            order.append(f"before-{self.name}")
            return messages

        def apply_after(self, output):
            order.append(f"after-{self.name}")
            return output

    run_agent(test_llm, [Tag(name="a"), Tag(name="b")])
    assert order == ["before-a", "before-b", "after-b", "after-a"]


def test_error_block_policy_fails_closed_per_on_block(test_llm):
    failed, sent, _ = run_agent(test_llm, [Exploding(on_error="block")])
    assert failed.status == RunnableStatus.FAILURE and sent == []

    answered, _, _ = run_agent(test_llm, [Exploding(on_error="block", on_block="answer")])
    assert answered.status == RunnableStatus.SUCCESS and answered.output["blocked"] is True
    assert "boom" in answered.output["content"]


def test_error_stop_policy_fails_the_run(test_llm):
    result, _, _ = run_agent(test_llm, [Exploding(on_error="stop")])
    assert result.status == RunnableStatus.FAILURE


def test_error_skip_policy_continues_with_unmodified_messages(test_llm):
    result, sent, _ = run_agent(test_llm, [Exploding(on_error="skip")])
    assert result.status == RunnableStatus.SUCCESS and len(sent) == 1


def test_model_hook_activity_is_traced(test_llm):
    tracing = TracingCallbackHandler()
    hooks = [
        RegexRedactModelHook(patterns=["hello"], apply_to="input"),
        Exploding(on_error="skip"),
        RegexGuardModelHook(block_if_matches=["FORBIDDEN"], apply_to="output", on_block="answer"),
    ]
    run_agent(test_llm, hooks, reply=finish("FORBIDDEN"), config=RunnableConfig(callbacks=[tracing]))

    events = next(r for r in tracing.runs.values() if "model_hooks" in r.metadata).metadata["model_hooks"]
    phases = [(e["phase"], e["hook"]) for e in events]
    assert ("input", 0) in phases and ("error", 1) in phases and ("block_output", 2) in phases
    assert all("FORBIDDEN" not in json.dumps(e) for e in events)  # the original reply is never traced


def test_streamed_accumulation_does_not_override_a_rewritten_reply(test_llm):
    agent = Agent(id="agent", name="a", llm=test_llm, role="r", model_hooks=[Prepend()])
    agent.inference_mode = InferenceMode.STRUCTURED_OUTPUT
    accumulated = MagicMock(accumulated_content=finish("original streamed text"))

    def run(**kwargs):
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": finish("original streamed text")}
        return result

    with (
        patch.object(agent.llm, "run", side_effect=run),
        patch.object(agent, "_setup_streaming_callback", return_value=(accumulated, RunnableConfig(), False)),
    ):
        result = agent.run(input_data={"input": "hi"})

    assert result.output["content"] == "rewritten by hook"


def test_invalid_regex_is_rejected_at_construction():
    with pytest.raises(ValueError, match="invalid regex"):
        RegexRedactModelHook(patterns=["("])
    with pytest.raises(ValueError, match="invalid regex"):
        RegexGuardModelHook(block_if_matches=["[a"])


def test_builtin_hooks_dump_type_and_resolve_on_agent(test_llm):
    hook = RegexGuardModelHook(block_if_matches=["x"], on_block="answer")
    dumped = hook.model_dump()
    assert dumped["type"] == "dynamiq.nodes.agents.model_hooks.RegexGuardModelHook"
    assert "type" not in ModelHook().model_dump()

    agent = Agent(llm=test_llm, role="r", model_hooks=[dumped, {"block_message": "plain"}])
    assert agent.model_hooks[0] == hook
    assert type(agent.model_hooks[1]) is ModelHook

    with pytest.raises(ValueError, match="not a ModelHook subclass"):
        Agent(llm=test_llm, role="r", model_hooks=[{"type": "dynamiq.nodes.agents.hooks.ToolHook"}])
