import uuid

import pytest

from dynamiq import connections, prompts
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.registry import ModelRegistry
from dynamiq.nodes.llms.togetherai import TogetherAI
from dynamiq.nodes.types import InferenceMode

UNKNOWN_MODEL = "moonshotai/kimi-k2.6"
PREFIXED_MODEL = f"together_ai/{UNKNOWN_MODEL}"


def _make_llm():
    return TogetherAI(
        name="TestLLM",
        model=UNKNOWN_MODEL,
        connection=connections.TogetherAI(id=str(uuid.uuid4()), api_key="fake-key"),
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )


@pytest.fixture()
def _litellm_unknown(mocker):
    """Force litellm to not recognise the model (get_model_info raises)."""
    mocker.patch("dynamiq.nodes.llms.base.get_model_info", side_effect=Exception("Unknown"))


def test_function_calling_unknown_litellm_but_registry_supports_does_not_raise(mocker, _litellm_unknown):
    registry = ModelRegistry()
    registry.register(PREFIXED_MODEL, {"supports_function_calling": True})
    mocker.patch("dynamiq.nodes.llms.base.model_registry", registry)

    agent = Agent(name="a", llm=_make_llm(), tools=[], inference_mode=InferenceMode.FUNCTION_CALLING)
    assert agent.inference_mode == InferenceMode.FUNCTION_CALLING


def test_function_calling_truly_unknown_model_warns_and_allows(mocker, _litellm_unknown):
    mocker.patch("dynamiq.nodes.llms.base.model_registry", ModelRegistry())

    agent = Agent(name="a", llm=_make_llm(), tools=[], inference_mode=InferenceMode.FUNCTION_CALLING)
    assert agent.inference_mode == InferenceMode.FUNCTION_CALLING


def test_function_calling_litellm_known_unsupported_still_raises(mocker):
    """When litellm KNOWS the model and says no FC, construction must still fail."""
    mocker.patch("dynamiq.nodes.llms.base.get_model_info", return_value={"supports_function_calling": False})
    mocker.patch("dynamiq.nodes.llms.base.supports_function_calling", return_value=False)

    with pytest.raises(ValueError, match="does not support function calling"):
        Agent(name="a", llm=_make_llm(), tools=[], inference_mode=InferenceMode.FUNCTION_CALLING)
