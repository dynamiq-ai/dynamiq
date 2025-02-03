from dynamiq.callbacks.streaming import StreamingEventMessage
from dynamiq.runnables import RunnableConfig
from dynamiq.types.feedback import FeedbackMethod


def send_message(
    event_message: StreamingEventMessage,
    config: RunnableConfig,
    feedback_method: FeedbackMethod = FeedbackMethod.STREAM,
) -> None:
    """Emits message

    Args:
        message (StreamingEventMessage): Message to send.
        config (RunnableConfig): Configuration for the runnable.
        feedback_method (FeedbackMethod, optional): Sets up where message is sent. Defaults to "stream".
    """

    match feedback_method:
        case FeedbackMethod.CONSOLE:
            print(event_message.data)
        case FeedbackMethod.STREAM:
            for callback in config.callbacks:
                callback.on_node_execute_stream({}, event=event_message)
