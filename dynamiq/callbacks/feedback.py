import asyncio
from typing import Any, AsyncIterator

from dynamiq.runnables import RunnableConfig
from dynamiq.types.message import EventMessage
from dynamiq.utils import format_value
from dynamiq.utils.logger import logger

from .base import BaseCallbackHandler, get_run_id
from .streaming import AsyncStreamingIteratorCallbackHandler


class AsyncFeedbackIteratorCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming events using an async iterator."""

    def __init__(
        self,
        queue: asyncio.Queue | None = None,
        done_event: asyncio.Event | None = None,
    ) -> None:
        """Initialize AsyncStreamingIteratorCallbackHandler.

        Args:
            queue (asyncio.Queue | None): Queue for streaming events.
            done_event (asyncio.Event | None): Event to signal completion.
        """
        super().__init__()
        if queue is None:
            self.queue = asyncio.Queue()
        if done_event is None:
            self.done_event = asyncio.Event()

        self._iterator = self._iter_queue_events()

    async def _iter_queue_events(self) -> AsyncIterator[EventMessage]:
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

    async def __anext__(self) -> EventMessage:
        """Get the next async streaming event.

        Returns:
            StreamingEventMessage: Next async streaming event.
        """
        return await self._iterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[EventMessage]:
        """Get the async iterator for streaming events.

        Returns:
            AsyncIterator[StreamingEventMessage]: Async iterator for streaming events.
        """
        async for item in self._iterator:
            yield item

    def on_node_end(self, serialized, output_data, **kwargs):
        if kwargs.get("send_output", False):
            event = kwargs.get("event") or EventMessage(
                run_id=str(get_run_id(kwargs)),
                wf_run_id=kwargs.get("wf_run_id"),
                entity_id=serialized.get("id"),
                data=format_value(output_data),
            )
            self.queue.put_nowait(event)

    def on_workflow_end(self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any) -> None:
        """Called when the workflow ends.

        Args:
            serialized (dict[str, Any]): Serialized workflow data.
            output_data (dict[str, Any]): Output data from the workflow.
            **kwargs (Any): Additional arguments.
        """
        self.done_event.set()


def send_message(message: EventMessage, config: RunnableConfig) -> None:
    """Emits message to the queue of AsyncFeedbackIteratorCallbackHandler or AsyncStreamingIteratorCallbackHandler

    Args:
        message (EventMessage): Message to send.
        config (RunnableConfig): RunnableConfig with callbacks.
    """
    sender_callback = None
    sender_callback_streaming = None

    for callback in config.callbacks:
        if isinstance(callback, AsyncStreamingIteratorCallbackHandler):
            sender_callback_streaming = callback
        elif isinstance(callback, AsyncFeedbackIteratorCallbackHandler):
            sender_callback = callback

    if sender_callback:
        sender_callback.queue.put_nowait(message)
    elif sender_callback_streaming:
        sender_callback_streaming.queue.put_nowait(message)
    else:
        raise ValueError(
            "Error: To use 'send_message' function either AsyncFeedbackIteratorCallbackHandler "
            "or AsyncStreamingIteratorCallbackHandler has to be present in config callbacks. "
        )
