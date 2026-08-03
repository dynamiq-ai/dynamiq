"""litellm drops `tools` for models it does not recognise; these cover the fix.

Completions dispatch with ``drop_params=True``, so litellm removes any param missing
from its per-model allowlist rather than erroring. For ``tools`` that is silent and
fatal: the model never sees a tool, the agent loops, and the run fails with a max-loops
error that names nothing. The fix forces ``tools``/``tool_choice`` through litellm's
per-request ``allowed_openai_params`` escape hatch.

Two properties matter beyond "it works":

* **Scope.** ``drop_params=True`` is what lets one uniform param set reach 25 providers --
  ``temperature`` on an o-series model or ``seed`` on Claude would otherwise raise. Only
  ``tools`` is forced through, because only ``tools`` changes what the request *is*.
* **No shared state.** The value rides on the request, so parallel workflows stay
  independent. The rejected alternative, ``litellm.register_model``, mutates the global
  ``litellm.model_cost`` without locking.
"""

from unittest.mock import patch

from litellm import get_supported_openai_params

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import TogetherAI as TogetherAIConnection
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.llms.togetherai import TogetherAI
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
PROMPT = Prompt(messages=[Message(role="user", content="What is the weather in Paris?")])

# A model on a gating provider that neither litellm nor model_registry.json knows. A
# fictional name keeps the test independent of the registry -- a real model could be
# added there later and silently stop exercising this path.
UNKNOWN_MODEL = "vendor/model-that-litellm-does-not-know"


def _together(model: str = UNKNOWN_MODEL) -> TogetherAI:
    return TogetherAI(name="llm", model=model, connection=TogetherAIConnection(api_key="x"))


def _params(llm, **kwargs) -> dict:
    return llm._build_completion_params(
        messages=[{"role": "user", "content": "hi"}],
        config=RunnableConfig(callbacks=[]),
        prompt=PROMPT,
        **kwargs,
    )


class TestToolParamsToForce:
    def test_forces_tools_for_a_model_litellm_would_strip(self):
        llm = _together()
        assert "tools" not in (get_supported_openai_params(model=llm.model) or [])
        assert llm._tool_params_to_force() == ["tools", "tool_choice"]

    def test_forces_nothing_when_litellm_already_forwards_tools(self):
        """No override where litellm is already correct -- keep its filtering intact."""
        llm = OpenAI(name="llm", model="gpt-4o", connection=OpenAIConnection(api_key="x"))
        assert llm._tool_params_to_force() == []

    def test_never_raises(self):
        """A metadata check must not be able to break dispatch."""
        with patch(
            "dynamiq.nodes.llms.base.get_supported_openai_params", side_effect=RuntimeError("boom")
        ):
            assert _together()._tool_params_to_force() == []


class TestBuildCompletionParams:
    """The regression that would return silently: tools present but stripped in flight."""

    def test_unknown_model_gets_allowed_openai_params(self):
        params = _params(_together(), tools=[TOOL])
        assert params["tools"], "tools must reach the request"
        assert params["allowed_openai_params"] == ["tools", "tool_choice"]

    def test_known_model_gets_no_override(self):
        llm = OpenAI(name="llm", model="gpt-4o", connection=OpenAIConnection(api_key="x"))
        assert "allowed_openai_params" not in _params(llm, tools=[TOOL])

    def test_no_override_without_tools(self):
        """XML and DEFAULT inference modes send no tools; nothing should be forced."""
        assert "allowed_openai_params" not in _params(_together())

    def test_drop_params_still_enabled(self):
        """Only `tools` is forced. Everything else must keep degrading gracefully, or
        `temperature` on an o-series model and `seed` on Claude become hard failures."""
        params = _params(_together(), tools=[TOOL])
        assert params["drop_params"] is True

    def test_shared_litellm_state_is_untouched(self):
        """Parallel workflows must not race: the override rides on the request."""
        import litellm

        llm = _together("vendor/unknown-model-no-global-writes")
        _params(llm, tools=[TOOL])

        assert llm.model not in litellm.model_cost
        assert "tools" not in (get_supported_openai_params(model=llm.model) or [])
