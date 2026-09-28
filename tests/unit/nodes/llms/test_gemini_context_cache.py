"""Gemini context caching: the cache markers Dynamiq hands LiteLLM.

LiteLLM turns the leading block of marked messages into a Google ``cachedContent``
resource. A marker in the wrong place fails silently -- the request succeeds uncached,
or with the default TTL -- so these assert the exact messages sent.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from dynamiq import connections, prompts
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Gemini, GeminiCacheControl, VertexAI
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig

MODEL = "gemini/gemini-2.5-pro"
SYSTEM = {"role": "system", "content": "You are a support bot."}
USER = {"role": "user", "content": "hi"}
TOO_SMALL = "Cached content is too small. total_token_count=1385, min_total_token_count=2048"
VERTEX_TOO_SMALL = "The cached content is of 3998 tokens. The minimum token count to start explicit caching is 4096."


def _cfg(**kwargs) -> GeminiCacheControl:
    """A config without the size gate, for tests that use a one-line system prompt."""
    return GeminiCacheControl(min_tokens=1, **kwargs)


def _gemini(model=MODEL, **kwargs) -> Gemini:
    return Gemini(
        name="g",
        model=model,
        connection=connections.Gemini(api_key="k"),
        is_postponed_component_init=True,
        **kwargs,
    )


def _vertex(model, **kwargs) -> VertexAI:
    return VertexAI(
        name="v",
        model=model,
        connection=connections.VertexAI(vertex_project_id="p", vertex_project_location="us-central1"),
        is_postponed_component_init=True,
        **kwargs,
    )


def _messages(llm, messages) -> list[dict]:
    return llm.update_completion_params({"model": llm.model, "messages": messages})["messages"]


def _cached_system(content="You are a support bot.", control=None) -> dict:
    control = control or {"type": "ephemeral"}
    return {"role": "system", "content": [{"type": "text", "text": content, "cache_control": control}]}


def _mock_response(content="ok"):
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    response = MagicMock()
    response.choices = [choice]
    response.model_extra = {}
    usage = MagicMock()
    usage.prompt_tokens = usage.completion_tokens = usage.total_tokens = 0
    usage.prompt_tokens_details = None
    usage.cache_read_input_tokens = usage.cache_creation_input_tokens = None
    response.usage = usage
    return response


class TestDisabledByDefault:
    @pytest.mark.parametrize("cache_control", [None, False])
    def test_messages_are_sent_untouched(self, cache_control):
        messages = [SYSTEM, USER]

        assert _messages(_gemini(cache_control=cache_control), messages) is messages

    def test_agent_does_not_turn_it_on(self):
        """Explicit caching bills storage per hour, so unlike Claude it stays opt-in."""
        llm = _gemini(prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]))
        llm.connection.id = str(uuid.uuid4())

        Agent(name="a", llm=llm, tools=[], max_loops=2)

        assert llm.cache_control is None

    def test_agent_does_not_turn_it_on_for_claude_on_vertex(self):
        """The model name says Claude and the node has a `cache_control` field -- but the
        field takes a Gemini config, which Claude would reject."""
        llm = _vertex(
            "vertex_ai/claude-sonnet-4-6",
            prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
        )
        llm.connection.id = str(uuid.uuid4())

        Agent(name="a", llm=llm, tools=[], max_loops=2)

        assert llm.cache_control is None


class TestTheTwoNodesStayInSync:
    """`Gemini` and `VertexAI` each declare the field and hooks, as `Anthropic` and `Bedrock` do."""

    def test_same_cache_control_field(self):
        assert Gemini.model_fields["cache_control"].annotation == VertexAI.model_fields["cache_control"].annotation

    @pytest.mark.parametrize(
        "hook",
        [
            "supports_prompt_caching",
            "update_completion_params",
            "_recover_completion_params",
        ],
    )
    def test_both_nodes_override_the_hook(self, hook):
        assert hook in vars(Gemini) and hook in vars(VertexAI)


class TestMarkers:
    def test_string_system_prompt_becomes_a_marked_text_block(self):
        """On the block, not the message: LiteLLM reads the TTL only from a block."""
        assert _messages(_gemini(cache_control=_cfg()), [SYSTEM, USER]) == [_cached_system(), USER]

    def test_ttl_is_sent_in_seconds(self):
        messages = _messages(_gemini(cache_control=_cfg(ttl_seconds=300)), [SYSTEM, USER])

        assert messages[0] == _cached_system(control={"type": "ephemeral", "ttl": "300s"})

    @pytest.mark.parametrize("field", ["ttl_seconds", "min_tokens"])
    def test_numbers_must_be_positive(self, field):
        with pytest.raises(ValueError):
            GeminiCacheControl(**{field: 0})

    def test_only_the_leading_system_messages_are_marked(self):
        """LiteLLM caches one contiguous block; a later system message must stay out of it."""
        late_system = {"role": "system", "content": "Summary so far."}
        second = {"role": "system", "content": "Be brief."}

        messages = _messages(_gemini(cache_control=_cfg()), [SYSTEM, second, USER, late_system])

        assert messages == [_cached_system(), _cached_system("Be brief."), USER, late_system]

    def test_list_content_marks_only_the_last_block(self):
        system = {
            "role": "system",
            "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}],
        }

        messages = _messages(_gemini(cache_control=_cfg()), [system, USER])

        assert messages[0]["content"] == [
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two", "cache_control": {"type": "ephemeral"}},
        ]

    def test_caller_messages_are_not_mutated(self):
        """The messages may belong to the node's prompt and are reused on the next call."""
        block = {"type": "text", "text": "part"}
        system = {"role": "system", "content": [block]}
        messages = [system, USER]

        _messages(_gemini(cache_control=_cfg()), messages)

        assert messages == [{"role": "system", "content": [{"type": "text", "text": "part"}]}, USER]
        assert "cache_control" not in block

    def test_no_system_prompt_sends_nothing_cached(self):
        assert _messages(_gemini(cache_control=_cfg()), [USER]) == [USER]

    @pytest.mark.parametrize("model", ["vertex_ai/gemini-2.5-pro", "vertex_ai/gemini-3-flash-preview"])
    def test_vertex_gemini_is_marked(self, model):
        assert _messages(_vertex(model, cache_control=_cfg()), [SYSTEM, USER])[0] == _cached_system()

    @pytest.mark.parametrize(
        "llm",
        [
            # Claude on Vertex takes Anthropic's cache_control, where "300s" is not a valid TTL.
            lambda: _vertex("vertex_ai/claude-sonnet-4-6", cache_control=_cfg(ttl_seconds=300)),
            lambda: _gemini("gemini/gemma-3-27b-it", cache_control=_cfg()),
        ],
    )
    def test_non_gemini_models_are_left_alone(self, llm):
        messages = [SYSTEM, USER]

        assert _messages(llm(), messages) is messages


class TestRejectionRecovery:
    """Google rejects a cache below the model's minimum, and Vertex `us-central1` sometimes
    rejects one of any size, failing the whole request instead of skipping the cache."""

    def _node(self):
        return Gemini(
            name="g",
            model=MODEL,
            connection=connections.Gemini(api_key="k"),
            prompt=Prompt(messages=[SYSTEM, USER]),
            cache_control=_cfg(),
        )

    @pytest.mark.parametrize("error", [TOO_SMALL, VERTEX_TOO_SMALL])
    def test_retries_this_request_uncached_and_keeps_caching_on(self, error):
        """The Vertex rejection is intermittent, so the next request tries the cache again."""
        with patch("litellm.completion"), patch("litellm.stream_chunk_builder"):
            node = self._node()
            calls = []

            def fake_completion(**params):
                calls.append(params["messages"])
                content = params["messages"][0]["content"]
                if isinstance(content, list) and content[-1].get("cache_control"):
                    raise Exception(error)
                return _mock_response("ok")

            node._completion = fake_completion
            result = node.execute(MagicMock(messages=None, files=None), config=RunnableConfig(callbacks=[]))

            assert result["content"] == "ok"
            assert calls[1][0] == {"role": "system", "content": [{"type": "text", "text": "You are a support bot."}]}
            assert node.cache_control == _cfg()

            node.execute(MagicMock(messages=None, files=None), config=RunnableConfig(callbacks=[]))
            assert len(calls) == 4, "the next request tries the cache again"

    def test_other_errors_are_not_retried_uncached(self):
        with patch("litellm.completion"), patch("litellm.stream_chunk_builder"):
            node = self._node()
            node._completion = MagicMock(side_effect=Exception("quota exceeded"))

            with pytest.raises(Exception, match="quota exceeded"):
                node.execute(MagicMock(messages=None, files=None), config=RunnableConfig(callbacks=[]))

            assert node._completion.call_count == 1
            assert node.cache_control == _cfg()

    def test_uncached_request_is_not_recovered(self):
        """Without markers the error is not ours to handle; no pointless identical retry."""
        params = {"messages": [SYSTEM, USER]}

        assert _gemini(cache_control=_cfg())._recover_completion_params(Exception(TOO_SMALL), params) is None


class TestSizeGate:
    """Below `min_tokens` the request goes out uncached, so it cannot hit Google's minimum."""

    BIG = "Rule: answer precisely and cite the rule. " * 700  # ~6k tokens

    def test_default_gate_leaves_a_short_prompt_uncached(self):
        messages = [SYSTEM, USER]

        assert _messages(_gemini(cache_control=GeminiCacheControl()), messages) is messages

    def test_long_prompt_is_cached_with_the_default_gate(self):
        messages = _messages(
            _gemini(cache_control=GeminiCacheControl()), [{"role": "system", "content": self.BIG}, USER]
        )

        assert messages[0] == _cached_system(self.BIG)

    def test_tools_count_towards_the_gate(self):
        """Google caches the tool schemas with the system prompt, so they count too."""
        tools = [{"type": "function", "function": {"name": "search", "description": self.BIG, "parameters": {}}}]
        llm = _gemini(cache_control=GeminiCacheControl())

        params = llm.update_completion_params({"model": llm.model, "messages": [SYSTEM, USER], "tools": tools})

        assert params["messages"][0] == _cached_system()

    def test_only_the_cached_part_counts(self):
        """A long user message is not cached, so it must not lift a short system prompt over the gate."""
        messages = [SYSTEM, {"role": "user", "content": self.BIG}]

        assert _messages(_gemini(cache_control=GeminiCacheControl()), messages) is messages

    def test_token_count_failure_sends_uncached(self):
        messages = [{"role": "system", "content": self.BIG}, USER]

        with patch("dynamiq.nodes.llms.gemini.token_counter", side_effect=RuntimeError("boom")):
            assert _messages(_gemini(cache_control=GeminiCacheControl()), messages) is messages
