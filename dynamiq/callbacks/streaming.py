import asyncio
import threading
from enum import Enum
from queue import Queue
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.callbacks.base import get_run_id
from dynamiq.types.streaming import StreamingEntitySource, StreamingEventMessage, StreamingMode, StreamingThought
from dynamiq.utils import format_value
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.nodes.agents import Agent


FINAL_ANSWER_FUNCTION_NAME = "provide_final_answer"
TAIL_GUARD_SIZE = 16
STREAMING_SEGMENT_SIZE = 8
FIND_JSON_FIELD_MAX_OFFSET = 64
WHITESPACE_PATTERNS = (" ", "\n", "\r", "\t")


class DefaultModeTag(str, Enum):
    """
    Enumeration of default mode tags.
    """

    THOUGHT = "Thought:"
    ACTION = "Action:"
    ANSWER = "Answer:"


class XMLModeTag(str, Enum):
    """
    Enumeration of XML mode tags.
    """

    OPEN_THOUGHT = "<thought>"
    CLOSE_THOUGHT = "</thought>"
    OPEN_ACTION = "<action>"
    OPEN_ANSWER = "<answer>"
    CLOSE_ANSWER = "</answer>"


class JSONStreamingField(str, Enum):
    """
    Enumeration of JSON streaming fields in FUNCTION_CALLING mode.
    """

    THOUGHT = "thought"
    ACTION = "action"
    ACTION_INPUT = "action_input"
    ANSWER = "answer"


class StreamingState(str, Enum):
    """
    Enumeration of streaming states.
    """

    REASONING = "reasoning"
    ANSWER = "answer"


class InferenceMode(str, Enum):
    """
    Enumeration of inference types.
    """

    DEFAULT = "DEFAULT"
    XML = "XML"
    FUNCTION_CALLING = "FUNCTION_CALLING"
    STRUCTURED_OUTPUT = "STRUCTURED_OUTPUT"


class BaseStreamingCallbackHandler(BaseCallbackHandler):
    """Base callback handler for streaming events."""


class StreamingQueueCallbackHandler(BaseStreamingCallbackHandler):
    """Callback handler for streaming events to a queue.

    Attributes:
        queue (asyncio.Queue | Queue | None): Queue for streaming events.
        done_event (asyncio.Event | threading.Event | None): Event to signal completion.
    """

    def __init__(
        self,
        queue: asyncio.Queue | Queue | None = None,
        done_event: asyncio.Event | threading.Event | None = None,
    ) -> None:
        """Initialize StreamingQueueCallbackHandler.

        Args:
            queue (asyncio.Queue | Queue | None): Queue for streaming events.
            done_event (asyncio.Event | threading.Event | None): Event to signal completion.
        """
        self.queue = queue
        self.done_event = done_event

    def on_workflow_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Called when the workflow starts.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            prompts (list[str]): List of prompts.
            **kwargs (Any): Additional arguments.
        """
        self.done_event.clear()

    def on_node_execute_stream(
        self, serialized: dict[str, Any], chunk: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Called when the node execute streams.

        Args:
            serialized (dict[str, Any]): Serialized node data.
            chunk (dict[str, Any] | None): Stream chunk data.
            **kwargs (Any): Additional arguments.
        """
        event = kwargs.get("event") or StreamingEventMessage(
            run_id=str(get_run_id(kwargs)),
            wf_run_id=kwargs.get("wf_run_id"),
            entity_id=serialized.get("id"),
            data=format_value(chunk),
            event=serialized.get("streaming", {}).get("event"),
            source=StreamingEntitySource(
                name=serialized.get("name", None),
                group=serialized.get("group", None),
                type=serialized.get("type", None),
            ),
        )
        self.send_to_queue(event)

    def on_workflow_end(
        self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any
    ) -> None:
        """Called when the workflow ends.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            output_data (dict[str, Any]): Output data from the workflow.
            **kwargs (Any): Additional arguments.
        """
        event = StreamingEventMessage(
            run_id=str(get_run_id(kwargs)),
            wf_run_id=kwargs.get("wf_run_id"),
            entity_id=serialized.get("id"),
            data=format_value(output_data),
            event=serialized.get("streaming", {}).get("event"),
            source=StreamingEntitySource(
                name=serialized.get("name", None),
                group=serialized.get("group", None),
                type=serialized.get("type", None),
            ),
        )
        self.send_to_queue(event)
        self.done_event.set()

    def on_workflow_error(
        self, serialized: dict[str, Any], error: BaseException, **kwargs: Any
    ) -> None:
        """Called when the workflow errors.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            error (BaseException): Error encountered.
            **kwargs (Any): Additional arguments.
        """
        self.done_event.set()

    def send_to_queue(self, event: StreamingEventMessage):
        """Send the event to the queue."""
        self.queue.put_nowait(event)


class StreamingIteratorCallbackHandler(StreamingQueueCallbackHandler):
    """Callback handler for streaming events using an iterator."""

    def __init__(
        self,
        queue: Queue | None = None,
        done_event: threading.Event | None = None,
    ) -> None:
        """Initialize StreamingIteratorCallbackHandler.

        Args:
            queue (Queue | None): Queue for streaming events.
            done_event (threading.Event | None): Event to signal completion.
        """
        if queue is None:
            queue = Queue()
        if done_event is None:
            done_event = threading.Event()
        super().__init__(queue, done_event)
        self._iterator = self._iter_queue_events()

    def _iter_queue_events(self) -> Iterator[StreamingEventMessage]:
        """Iterate over queue events.

        Returns:
            Iterator[StreamingEventMessage]: Iterator for streaming events.
        """
        try:
            while not self.queue.empty() or not self.done_event.is_set():
                event = self.queue.get()
                yield event
        except Exception as e:
            logger.error(f"Event streaming failed. Error: {e}")

    def __next__(self) -> StreamingEventMessage:
        """Get the next streaming event.

        Returns:
            StreamingEventMessage: Next streaming event.
        """
        return self._iterator.__next__()

    def __iter__(self) -> Iterator[StreamingEventMessage]:
        """Get the iterator for streaming events.

        Returns:
            Iterator[StreamingEventMessage]: Iterator for streaming events.
        """
        yield from self._iterator


class AsyncStreamingIteratorCallbackHandler(StreamingQueueCallbackHandler):
    """Callback handler for streaming events using an async iterator."""

    def __init__(
        self,
        queue: asyncio.Queue | None = None,
        done_event: asyncio.Event | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Initialize AsyncStreamingIteratorCallbackHandler.

        Args:
            queue (asyncio.Queue | None): Queue for streaming events.
            done_event (asyncio.Event | None): Event to signal completion.
            loop (asyncio.AbstractEventLoop | None): Event loop.
        """
        if queue is None:
            queue = asyncio.Queue()
        if done_event is None:
            done_event = asyncio.Event()
        super().__init__(queue, done_event)
        self._iterator = self._iter_queue_events()
        self.loop = loop or asyncio.get_event_loop()

    def send_to_queue(self, event: StreamingEventMessage):
        """Send the event to the queue."""
        asyncio.run_coroutine_threadsafe(self.queue.put(event), self.loop)

    async def _iter_queue_events(self) -> AsyncIterator[StreamingEventMessage]:
        """Async iterate over queue events.

        Returns:
            AsyncIterator[StreamingEventMessage]: Async iterator for streaming events.
        """
        try:
            while not self.queue.empty() or not self.done_event.is_set():
                event = await self.queue.get()
                yield event
        except Exception as e:
            logger.error(f"Event streaming failed. Error: {e}")

    async def __anext__(self) -> StreamingEventMessage:
        """Get the next async streaming event.

        Returns:
            StreamingEventMessage: Next async streaming event.
        """
        return await self._iterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[StreamingEventMessage]:
        """Get the async iterator for streaming events.

        Returns:
            AsyncIterator[StreamingEventMessage]: Async iterator for streaming events.
        """
        async for item in self._iterator:
            yield item


class AgentStreamingParserCallback(BaseStreamingCallbackHandler):
    """Agent callback that parses LLM streaming output in real time and streams structured chunks.

    This callback attaches to the underlying LLM node (group == 'llms'), incrementally parses the
    provider streaming chunks to detect logical sections like reasoning and final answer based on
    the selected inference mode, and forwards the relevant segments via the `stream_content()` API
    with appropriate `step` labels (reasoning/answer) included.
    """

    def __init__(self, agent: "Agent", config, loop_num: int, **kwargs):
        self.agent = agent
        self.config = config
        self.loop_num = loop_num
        self.kwargs = kwargs

        # Aggregate streamed text from the LLM in the current loop for proper tracing inside the agent
        self.accumulated_content: str = ""

        self._buffer: str = ""
        self._current_state: str | None = None
        self._state_start_index: int = 0
        self._state_last_emit_index: int = 0
        self._answer_started: bool = False
        self._state_has_emitted: dict[str, bool] = {
            StreamingState.REASONING: False,
            StreamingState.ANSWER: False,
        }

        # Set a tail guard to avoid streaming parts of the next tag that may arrive in next chunk
        self._tail_guard: int = TAIL_GUARD_SIZE

        self.mode_name = getattr(self.agent.inference_mode, "name", str(self.agent.inference_mode)).upper()

    def on_node_execute_stream(self, serialized: dict[str, Any], chunk: dict[str, Any] | None = None, **kwargs: Any):
        if not chunk or not self.agent.streaming.enabled:
            return

        # Only process the chunks from the LLM node
        if serialized.get("group") != "llms":
            return

        agent_run_id = kwargs.get("run_id") or kwargs.get("parent_run_id")
        if agent_run_id and serialized.get("id") != getattr(self.agent, "llm", object()).id:
            return

        if self.mode_name == InferenceMode.FUNCTION_CALLING.value:
            text_delta, function_name = self._extract_function_calling_text(chunk)

            if function_name and function_name == FINAL_ANSWER_FUNCTION_NAME:
                self._answer_started = True
        else:
            text_delta = self._extract_text_delta(chunk)

        if not text_delta:
            return

        self.accumulated_content += text_delta
        self._buffer += text_delta

        final_answer_only = self.agent.streaming.mode == StreamingMode.FINAL

        if self.mode_name == InferenceMode.DEFAULT.value:
            self._process_default_mode(final_answer_only)
        elif self.mode_name == InferenceMode.XML.value:
            self._process_xml_mode(final_answer_only)
        elif self.mode_name == InferenceMode.STRUCTURED_OUTPUT.value:
            self._process_structured_output_mode(final_answer_only)
        elif self.mode_name == InferenceMode.FUNCTION_CALLING.value:
            self._process_function_calling_mode(final_answer_only)

        self._trim_buffer()

    def on_node_execute_end(self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any):
        # Clear the remaining buffer content when the LLM streaming ends
        if serialized.get("group") != "llms":
            return

        if not self.agent.streaming.enabled:
            return

        self._flush_buffer()
        self._trim_buffer(force=True)

    def _flush_buffer(self) -> None:
        """Flush the remaining buffer content by streaming it as one chunk."""
        if not self._buffer or len(self._buffer) <= self._state_last_emit_index:
            return

        if self._current_state in (StreamingState.REASONING, StreamingState.ANSWER):
            remaining_content = self._buffer[self._state_last_emit_index :]
            if remaining_content.strip():
                self._emit(remaining_content, step=self._current_state)
                self._state_last_emit_index = len(self._buffer)

    def _extract_text_delta(self, chunk: dict[str, Any]) -> str:
        """Extract textual content from streaming chunk received from the LLM.

        Returns:
            str: The extracted content.
        """
        extracted_content = ""

        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if isinstance(content, str):
                extracted_content = content

        if not extracted_content and isinstance(chunk.get("content"), str):
            extracted_content = chunk.get("content")

        return extracted_content

    def _extract_function_calling_text(self, chunk: dict[str, Any]) -> tuple[str, str | None]:
        """
        Extract incremental JSON values (arguments) and function name
        from the LLM streaming chunks in FUNCTION_CALLING inference mode.

        Returns:
            tuple[str, str | None]: (arguments_text, function_name)
        """
        arguments_text = ""
        function_name = None

        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta", {})
            tool_calls = delta.get("tool_calls")

            if tool_calls and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                if tool_call.get("type") == "function" and "function" in tool_call:
                    function_data = tool_call["function"]

                    if function_data.get("name"):
                        function_name = function_data["name"]

                    if function_data.get("arguments"):
                        arguments_text = function_data["arguments"]

        return arguments_text, function_name

    def _emit(self, content: str, step: str) -> None:
        """Emit the parsed content using the agent's stream_content method.

        Args:
            content (str): The content to stream.
            step (str): The step to stream the content to.
        """
        if not content:
            return

        # Skip streaming if in FINAL mode and not in answer step
        if self.agent.streaming.mode == StreamingMode.FINAL and step != StreamingState.ANSWER:
            return

        if step in self._state_has_emitted:
            if not self._state_has_emitted[step]:
                trimmed = content.lstrip("\r\n ")
                if not trimmed:
                    return
                content = trimmed

        # Format content based on the step type
        if step == StreamingState.REASONING:
            thought_model = StreamingThought(thought=content, loop_num=self.loop_num)
            content_to_stream = thought_model.to_dict()
        elif step == StreamingState.ANSWER:
            content_to_stream = content

        self.agent.stream_content(
            content=content_to_stream,
            source=self.agent.name,
            step=step,
            config=self.config,
            **(self.kwargs | {"loop_num": self.loop_num}),
        )
        if step in self._state_has_emitted:
            self._state_has_emitted[step] = True

    def _process_default_mode(self, final_answer_only: bool) -> None:
        if self._current_state is None:
            start = self._state_last_emit_index
            idx_thought = self._buffer.find(DefaultModeTag.THOUGHT, start) if not final_answer_only else -1
            idx_answer = self._buffer.find(DefaultModeTag.ANSWER, start)

            if not final_answer_only and idx_thought != -1 and (idx_answer == -1 or idx_thought < idx_answer):
                self._current_state = StreamingState.REASONING
                self._state_start_index = idx_thought + len(DefaultModeTag.THOUGHT)
                self._state_last_emit_index = self._state_start_index
            elif idx_answer != -1:
                self._current_state = StreamingState.ANSWER
                self._answer_started = True
                self._state_start_index = idx_answer + len(DefaultModeTag.ANSWER)
                self._state_last_emit_index = self._state_start_index

        # If the state was not detected, nothing to emit yet
        if self._current_state is None:
            return

        search_start = self._state_last_emit_index

        if self._current_state == StreamingState.REASONING:
            # Check if there is a transition to Action or Answer
            next_tag_pos = -1
            next_tag_name = None
            for tag_text, name in ((DefaultModeTag.ACTION, "action"), (DefaultModeTag.ANSWER, "answer")):
                pos = self._buffer.find(tag_text, search_start)
                if pos != -1 and (next_tag_pos == -1 or pos < next_tag_pos):
                    next_tag_pos, next_tag_name = pos, name

            if next_tag_pos != -1:
                # If a complete next tag is found, emit everything up to it
                if next_tag_pos > self._state_last_emit_index:
                    self._emit(self._buffer[self._state_last_emit_index : next_tag_pos], step=StreamingState.REASONING)
                if next_tag_name == "answer":
                    self._current_state = StreamingState.ANSWER
                    self._answer_started = True
                    self._state_start_index = next_tag_pos + len(DefaultModeTag.ANSWER)
                    self._state_last_emit_index = self._state_start_index
                else:
                    # Wait for the next state after `action`
                    self._current_state = None
                    self._state_last_emit_index = next_tag_pos + len(DefaultModeTag.ACTION)
                return

            # If there is no next tag yet, emit incrementally using a tail guard
            safe_end = max(self._state_last_emit_index, len(self._buffer) - self._tail_guard)
            if safe_end > self._state_last_emit_index:
                self._emit(self._buffer[self._state_last_emit_index : safe_end], step=StreamingState.REASONING)
                self._state_last_emit_index = safe_end
            return

        # If the current state is 'answer', stream up to the end
        safe_end = max(self._state_last_emit_index, len(self._buffer) - self._tail_guard)
        if safe_end > self._state_last_emit_index:
            self._emit(self._buffer[self._state_last_emit_index : safe_end], step=StreamingState.ANSWER)
            self._state_last_emit_index = safe_end

    def _process_xml_mode(self, final_answer_only: bool) -> None:
        if self._current_state is None:
            start = self._state_last_emit_index
            idx_thought = self._buffer.find(XMLModeTag.OPEN_THOUGHT, start) if not final_answer_only else -1
            idx_answer = self._buffer.find(XMLModeTag.OPEN_ANSWER, start)

            if not final_answer_only and idx_thought != -1 and (idx_answer == -1 or idx_thought < idx_answer):
                self._current_state = StreamingState.REASONING
                self._state_start_index = idx_thought + len(XMLModeTag.OPEN_THOUGHT)
                self._state_last_emit_index = self._state_start_index
            elif idx_answer != -1:
                self._current_state = StreamingState.ANSWER
                self._answer_started = True
                self._state_start_index = idx_answer + len(XMLModeTag.OPEN_ANSWER)
                self._state_last_emit_index = self._state_start_index

        if self._current_state is None:
            return

        search_start = self._state_last_emit_index

        if self._current_state == StreamingState.REASONING:
            # Check for the next boundary: either </thought>, <action>, or <answer>
            next_pos = -1
            next_tag = None
            for tag in (XMLModeTag.CLOSE_THOUGHT, XMLModeTag.OPEN_ACTION, XMLModeTag.OPEN_ANSWER):
                pos = self._buffer.find(tag, search_start)
                if pos != -1 and (next_pos == -1 or pos < next_pos):
                    next_pos, next_tag = pos, tag

            if next_pos != -1:
                # Emit everything up to the next tag
                if next_pos > self._state_last_emit_index:
                    self._emit(self._buffer[self._state_last_emit_index : next_pos], step=StreamingState.REASONING)

                if next_tag == XMLModeTag.OPEN_ANSWER:
                    self._current_state = StreamingState.ANSWER
                    self._answer_started = True
                    self._state_start_index = next_pos + len(XMLModeTag.OPEN_ANSWER)
                    self._state_last_emit_index = self._state_start_index
                else:
                    # Stop reasoning stream due to either </thought> or <action>
                    self._current_state = None
                    self._state_last_emit_index = next_pos + len(next_tag)
                return

            # If there is no next tag yet, emit incrementally using a tail guard
            safe_end = max(self._state_last_emit_index, len(self._buffer) - self._tail_guard)
            if safe_end > self._state_last_emit_index:
                self._emit(self._buffer[self._state_last_emit_index : safe_end], step=StreamingState.REASONING)
                self._state_last_emit_index = safe_end
            return

        # If the current state is 'answer', stream up to the </answer> tag
        end_pos = self._buffer.find(XMLModeTag.CLOSE_ANSWER, search_start)
        if end_pos != -1:
            if end_pos > self._state_last_emit_index:
                self._emit(self._buffer[self._state_last_emit_index : end_pos], step=StreamingState.ANSWER)
            # Close the answer
            self._current_state = None
            self._state_last_emit_index = end_pos + len(XMLModeTag.CLOSE_ANSWER)
            return

        safe_end = max(self._state_last_emit_index, len(self._buffer) - self._tail_guard)
        if safe_end > self._state_last_emit_index:
            self._emit(self._buffer[self._state_last_emit_index : safe_end], step=StreamingState.ANSWER)
            self._state_last_emit_index = safe_end

    def _process_structured_output_mode(self, final_answer_only: bool) -> None:
        """Process structured output mode."""
        self._process_json_mode(final_answer_only, is_function_calling=False)

    def _process_function_calling_mode(self, final_answer_only: bool) -> None:
        """Process function calling mode."""
        self._process_json_mode(final_answer_only, is_function_calling=True)

    def _find_unescaped_quote_end(self, input_string: str, start_quote_index: int) -> int:
        """
        Return index of the next unescaped '"' after start_quote_index, or -1 if not complete yet.

        Args:
            input_string (str): The string to search in.
            start_quote_index (int): The index of the starting quote.

        Returns:
            int: The index of the next unescaped quote, or -1 if not found.
        """
        current_index = start_quote_index + 1
        while current_index < len(input_string):
            if input_string[current_index] == '"':
                # Count preceding backslashes
                backslash_count = 0
                previous_index = current_index - 1
                while previous_index >= 0 and input_string[previous_index] == "\\":
                    backslash_count += 1
                    previous_index -= 1
                if backslash_count % 2 == 0:
                    return current_index
            current_index += 1
        return -1

    def _find_field_string_value_start(self, input_string: str, field_name: str, start_index: int = 0) -> int:
        """
        Find the index of the first character inside the opening quote of a string field value.
        Returns -1 if field or opening quote is not fully present yet.

        Args:
            input_string (str): The string to search in.
            field_name (str): The name of the field to search for.
            start_index (int): The index to start searching from.

        Returns:
            int: The index of the first character inside the opening quote of the string field value,
                 or -1 if not found.
        """
        key = f'"{field_name}"'
        position = input_string.find(key, start_index)
        if position == -1:
            return -1

        colon_index = input_string.find(":", position + len(key))
        if colon_index == -1:
            return -1

        # Skip the whitespace after the colon
        index = colon_index + 1
        while index < len(input_string) and input_string[index] in WHITESPACE_PATTERNS:
            index += 1
        if index >= len(input_string) or input_string[index] != '"':
            return -1
        return index + 1

    def _initialize_json_field_state(
        self, buf: str, field_name: str, state: str, final_answer_only: bool = False
    ) -> bool:
        """
        Initialize streaming state for a JSON field if not already set.

        Args:
            buf: Buffer containing JSON content
            field_name: Name of the JSON field to look for
            state: State to set if field is found ("reasoning" or "answer")
            final_answer_only: Whether we're in final answer only mode

        Returns:
            bool: True if state was initialized, False otherwise
        """
        if self._current_state is not None:
            return False

        # Skip reasoning fields in final answer only mode
        if final_answer_only and state == StreamingState.REASONING:
            return False

        field_start = self._find_field_string_value_start(
            buf, field_name, max(0, self._state_last_emit_index - FIND_JSON_FIELD_MAX_OFFSET)
        )

        # If the field is found, set the state and indices
        if field_start != -1:
            self._current_state = state
            self._state_start_index = field_start
            self._state_last_emit_index = max(self._state_last_emit_index, field_start)
            return True
        return False

    def _process_json_mode(self, final_answer_only: bool, is_function_calling: bool = False) -> None:
        """
        Unified processing for JSON-like modes (structured output and function calling).

        Args:
            final_answer_only: Whether to stream only final answers
            is_function_calling: Whether this is function calling mode (vs structured output)
        """
        buf = self._buffer

        if not is_function_calling and not self._answer_started:
            # If there is a "finish" action, enable answer streaming
            action_key_pos = buf.find(
                f'"{JSONStreamingField.ACTION.value}"', max(0, self._state_last_emit_index - FIND_JSON_FIELD_MAX_OFFSET)
            )
            if action_key_pos != -1:
                colon_pos = buf.find(":", action_key_pos)
                if colon_pos != -1:
                    v_start = self._skip_whitespace(buf, colon_pos + 1)
                    if v_start < len(buf) and buf[v_start] == '"':
                        end_quote = self._find_unescaped_quote_end(buf, v_start)
                        if end_quote != -1:
                            action_value = buf[v_start + 1 : end_quote]
                            if action_value.strip().lower() == "finish":
                                self._answer_started = True
                                # Try to find the action_input field
                                action_input_start = self._find_field_string_value_start(
                                    buf, JSONStreamingField.ACTION_INPUT.value, end_quote + 1
                                )
                                if action_input_start != -1:
                                    self._current_state = StreamingState.ANSWER
                                    self._state_start_index = action_input_start
                                    self._state_last_emit_index = max(self._state_last_emit_index, action_input_start)

        self._initialize_json_field_state(
            buf, JSONStreamingField.THOUGHT.value, StreamingState.REASONING, final_answer_only
        )

        if self._answer_started:
            answer_field = (
                JSONStreamingField.ANSWER.value if is_function_calling else JSONStreamingField.ACTION_INPUT.value
            )
            self._initialize_json_field_state(buf, answer_field, StreamingState.ANSWER)

        if self._current_state == StreamingState.REASONING:
            self._emit_json_field_content(buf, StreamingState.REASONING)
        elif self._current_state == StreamingState.ANSWER:
            self._emit_json_field_content(buf, StreamingState.ANSWER)

    def _skip_whitespace(self, text: str, start: int) -> int:
        """Skip whitespace characters starting from the given position."""
        while start < len(text) and text[start] in WHITESPACE_PATTERNS:
            start += 1
        return start

    def _emit_json_field_content(self, buf: str, step: str) -> bool:
        """
        Emit JSON field content in segments, handling complete and partial values.

        Args:
            buf: Buffer containing the JSON content
            step: The streaming step ("reasoning" or "answer")

        Returns:
            bool: True if field is complete and state should be reset, False otherwise
        """
        # Find the closing quote of the current JSON field
        end_quote = self._find_unescaped_quote_end(buf, self._state_start_index - 1)
        if end_quote != -1:
            # If the field is complete, emit it
            if end_quote > self._state_last_emit_index:
                segment_start = self._state_last_emit_index
                while segment_start < end_quote:
                    segment_end = min(end_quote, segment_start + STREAMING_SEGMENT_SIZE)
                    self._emit(buf[segment_start:segment_end], step=step)
                    segment_start = segment_end
                self._state_last_emit_index = end_quote
            # Reset the state
            self._current_state = None
            return True

        # Emit incrementally if the field is not complete
        if len(buf) > self._state_last_emit_index:
            segment_start = self._state_last_emit_index
            segment_end_target = len(buf)
            while segment_start < segment_end_target:
                segment_end = min(segment_end_target, segment_start + STREAMING_SEGMENT_SIZE)
                self._emit(buf[segment_start:segment_end], step=step)
                segment_start = segment_end
            self._state_last_emit_index = segment_end_target
        return False

    def _trim_buffer(self, force: bool = False) -> None:
        """Trim already-emitted prefix of buffer to prevent re-detection."""
        if not self._buffer:
            return

        if self.mode_name == InferenceMode.STRUCTURED_OUTPUT.value:
            return

        if force:
            keep_from = self._state_last_emit_index
        else:
            if (
                self._current_state in (StreamingState.REASONING, StreamingState.ANSWER)
                and self._state_start_index != -1
            ):
                keep_from = max(0, min(self._state_last_emit_index - self._tail_guard, self._state_start_index - 1))
            else:
                keep_from = max(0, self._state_last_emit_index - self._tail_guard)
        if keep_from <= 0:
            return
        self._buffer = self._buffer[keep_from:]

        # Rebase the indices
        self._state_start_index = max(0, self._state_start_index - keep_from)
        self._state_last_emit_index = max(0, self._state_last_emit_index - keep_from)
