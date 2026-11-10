import json
from unittest.mock import MagicMock

import pytest

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import Groq as GroqConnection
from dynamiq.nodes.llms.groq import Groq
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig


def _groq_error(code: str, failed_generation: str) -> Exception:
    body = {"error": {"message": "Please adjust your prompt.", "code": code, "failed_generation": failed_generation}}
    return Exception(f"litellm.BadRequestError: GroqException - {json.dumps(body)}")


def _run(node: Groq, completion, callbacks=()) -> tuple[dict, list[dict]]:
    sent = []

    def record(**params):
        sent.append(params)
        return completion(**params)

    node._completion = record
    result = node.execute(MagicMock(messages=None, files=None), config=RunnableConfig(callbacks=list(callbacks)))
    return result, sent


def _node(**kwargs) -> Groq:
    return Groq(
        model="groq/openai/gpt-oss-120b",
        connection=GroqConnection(api_key="x"),
        prompt=Prompt(messages=[{"role": "user", "content": "Fetch invoice INV-88."}]),
        **kwargs,
    )


@pytest.mark.parametrize("code", ["json_validate_failed", "tool_use_failed"])
def test_a_reply_groq_rejects_reaches_the_caller_as_text(code):
    reply = '{"vendor": "Nordwind GmbH", "total_eur": 1234.5}'

    def completion(**params):
        raise _groq_error(code, reply)

    result, _ = _run(_node(), completion)

    assert result["content"] == reply


def test_a_streamed_reply_groq_rejects_is_requested_again_without_streaming():
    reply = "Your locker code is 6360CF."

    def completion(**params):
        raise _groq_error("tool_use_failed", reply)

    result, sent = _run(
        _node(streaming=StreamingConfig(enabled=True)), completion, callbacks=[StreamingIteratorCallbackHandler()]
    )

    assert result["content"] == reply
    assert [params.get("stream") for params in sent] == [True, False]


def test_another_bad_request_still_fails():
    def completion(**params):
        raise Exception('litellm.BadRequestError: GroqException - {"error": {"code": "model_not_found"}}')

    with pytest.raises(Exception, match="model_not_found"):
        _run(_node(), completion)
