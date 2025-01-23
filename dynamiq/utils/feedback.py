from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler, StreamingEventMessage
from dynamiq.runnables import RunnableConfig
from dynamiq.types.feedback import FeedbackMethod


def send_message(
    event_message: StreamingEventMessage, config: RunnableConfig, approval_type: FeedbackMethod = FeedbackMethod.STREAM
) -> None:
    """Emits message to the queue of AsyncStreamingIteratorCallbackHandler

    Args:
        message (StreamingEventMessage): Message to send.
        config (RunnableConfig): RunnableConfig with callbacks (AsyncStreamingIteratorCallbackHandler is required).
    """

    match approval_type:
        case FeedbackMethod.CONSOLE:
            print(event_message.data)
        case FeedbackMethod.STREAM:
            sender_callback = next(
                (
                    callback
                    for callback in config.callbacks
                    if isinstance(callback, AsyncStreamingIteratorCallbackHandler)
                ),
                None,
            )

            if sender_callback:
                sender_callback.on_node_execute_stream({}, event=event_message)
            else:
                raise ValueError(
                    "Error: To use 'send_message' function in streaming mode AsyncStreamingIteratorCallbackHandler "
                    "has to be present in config callbacks."
                )
