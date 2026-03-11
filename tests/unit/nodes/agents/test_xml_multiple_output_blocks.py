"""Tests for handling multiple <output> blocks in XML mode.

Covers:
- Agent._extract_first_output_block: only the first block is kept for parsing.
- AgentStreamingParserCallback._xml_output_complete: streaming stops after the first block.
"""

import uuid

import pytest

from dynamiq import connections
from dynamiq.callbacks.streaming import AgentStreamingParserCallback
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_node():
    conn = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-key")
    return OpenAI(name="LLM", model="gpt-4o-mini", connection=conn)


@pytest.fixture
def xml_agent(openai_node, mock_llm_executor):
    return Agent(
        name="TestXML",
        llm=openai_node,
        tools=[],
        inference_mode=InferenceMode.XML,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    )


# ---------------------------------------------------------------------------
# _extract_first_output_block
# ---------------------------------------------------------------------------


class TestExtractFirstOutputBlock:
    def test_single_output_block_unchanged(self):
        text = "<output><thought>hello</thought><action>tool</action><action_input>{}</action_input></output>"
        assert Agent._extract_first_output_block(text) == text

    def test_multiple_output_blocks_keeps_first(self):
        first = '<output><thought>first</thought><action>tool</action><action_input>{"a":1}</action_input></output>'
        second = "<output><thought>second</thought><answer>done</answer></output>"
        combined = first + "\n" + second
        assert Agent._extract_first_output_block(combined) == first

    def test_no_output_tags_unchanged(self):
        text = "<thought>hi</thought><answer>bye</answer>"
        assert Agent._extract_first_output_block(text) == text

    def test_three_blocks_keeps_first(self):
        blocks = [
            f"<output><thought>block{i}</thought><action>t{i}</action><action_input>{{}}</action_input></output>"
            for i in range(3)
        ]
        combined = "\n".join(blocks)
        assert Agent._extract_first_output_block(combined) == blocks[0]


# ---------------------------------------------------------------------------
# Streaming: _xml_output_complete stops further emission
# ---------------------------------------------------------------------------


class TestStreamingXmlOutputComplete:
    @staticmethod
    def _build_callback(agent) -> AgentStreamingParserCallback:
        return AgentStreamingParserCallback(
            agent=agent,
            config=RunnableConfig(),
            loop_num=1,
        )

    @staticmethod
    def _feed(cb: AgentStreamingParserCallback, text: str):
        """Simulate incremental token arrival by feeding one char at a time."""
        for ch in text:
            cb.accumulated_content += ch
            cb._buffer += ch
            cb._process_xml_mode(final_answer_only=False)

    def test_flag_set_after_answer_close(self, xml_agent):
        cb = self._build_callback(xml_agent)
        self._feed(cb, "<thought>reasoning</thought><answer>result</answer>")
        assert cb._xml_output_complete is True

    def test_flag_set_after_action_input_close(self, xml_agent):
        cb = self._build_callback(xml_agent)
        self._feed(cb, '<action>my_tool</action><action_input>{"x": 1}</action_input>')
        assert cb._xml_output_complete is True

    def test_second_block_not_streamed(self, xml_agent):
        cb = self._build_callback(xml_agent)
        emitted: list[tuple[str, str]] = []
        original_emit = cb._emit

        def capture_emit(content, step=None):
            emitted.append((step, content))
            original_emit(content, step=step)

        cb._emit = capture_emit

        first_block = "<thought>first thought</thought><answer>first answer</answer>"
        second_block = "<thought>second thought</thought><answer>second answer</answer>"
        self._feed(cb, first_block + second_block)

        all_text = "".join(content for _, content in emitted)
        assert "first thought" in all_text
        assert "first answer" in all_text
        assert "second thought" not in all_text
        assert "second answer" not in all_text

    def test_second_output_tag_not_streamed(self, xml_agent):
        """When the LLM wraps each action in <output> tags, only the first block is streamed."""
        cb = self._build_callback(xml_agent)
        emitted: list[tuple[str, str]] = []
        original_emit = cb._emit

        def capture_emit(content, step=None):
            emitted.append((step, content))
            original_emit(content, step=step)

        cb._emit = capture_emit

        first = (
            '<output><thought>plan A</thought><action>search</action><action_input>{"q":"x"}</action_input></output>'
        )
        second = "<output><thought>plan B</thought><answer>final</answer></output>"
        self._feed(cb, first + "\n" + second)

        all_text = "".join(content for _, content in emitted)
        assert "plan A" in all_text
        assert "plan B" not in all_text
        assert cb._xml_output_complete is True

    def test_flag_false_before_completion(self, xml_agent):
        cb = self._build_callback(xml_agent)
        self._feed(cb, "<thought>still thinking")
        assert cb._xml_output_complete is False
