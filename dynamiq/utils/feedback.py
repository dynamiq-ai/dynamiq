from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler, StreamingEventMessage
from dynamiq.runnables import RunnableConfig


def send_message(message: StreamingEventMessage, config: RunnableConfig) -> None:
    """Emits message to the queue of AsyncStreamingIteratorCallbackHandler

    Args:
        message (StreamingEventMessage): Message to send.
        config (RunnableConfig): RunnableConfig with callbacks (AsyncStreamingIteratorCallbackHandler is required).
    """
    sender_callback = None

    sender_callback = next(
        (callback for callback in config.callbacks if isinstance(callback, AsyncStreamingIteratorCallbackHandler)), None
    )

    if sender_callback:
        sender_callback.queue.put_nowait(message)
    else:
        raise ValueError(
            "Error: To use 'send_message' function AsyncStreamingIteratorCallbackHandler "
            "has to be present in config callbacks."
        )
