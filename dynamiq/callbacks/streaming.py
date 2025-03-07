import asyncio
import threading
from queue import Queue
from typing import Any, AsyncIterator, Iterator

from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.callbacks.base import get_run_id
from dynamiq.types.streaming import StreamingEventMessage
from dynamiq.utils import format_value
from dynamiq.utils.logger import logger


class StreamingQueueCallbackHandler(BaseCallbackHandler):
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
            data=format_value(chunk)[0],
            event=serialized.get("streaming", {}).get("event"),
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
            data=format_value(output_data)[0],
            event=serialized.get("streaming", {}).get("event"),
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
