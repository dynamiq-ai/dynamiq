import asyncio
import threading
import time
from queue import Queue
from threading import Event
from typing import ClassVar

import pytest
from pydantic import BaseModel, Field

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.streaming import StreamingConfig, StreamingEventMessage

TEST_ENTITY_ID = "test-entity"
TEST_EVENT = "streaming"
TEST_INPUT_DATA = {"test": "data"}

SHORT_TIMEOUT = 0.5
EXECUTE_TIMEOUT = 0.5
STREAMING_TIMEOUT_LONG = 5.0
SHORT_STREAMING_TIMEOUT = 1.0
MAX_ELAPSED_TIME = 5.0
CANCEL_DELAY = 0.1
BACKGROUND_THREAD_WAIT = 3.0
SHORT_POLL_INTERVAL = 0.2
LONG_POLL_INTERVAL = 2.0

TIMING_TOLERANCE = 0.2


class SlowExecuteNode(Node):
    """Test node that simulates slow execution."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "SlowExecuteNode"
    sleep_time: float = 5.0

    def execute(self, input_data: dict, config: RunnableConfig = None, **kwargs) -> dict:
        time.sleep(self.sleep_time)
        return {"result": "completed"}


class BlockingQueueNode(Node):
    """Test node that blocks on input queue."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "BlockingQueueNode"

    def execute(self, input_data: dict, config: RunnableConfig = None, **kwargs) -> dict:
        event_msg = self.get_input_streaming_event(config=config)
        return {"result": event_msg.data}


class TrackableBlockingNode(Node):
    """Test node that tracks when execution finishes in background."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "TrackableBlockingNode"
    finished_event: Event | None = None

    def execute(self, input_data: dict, config: RunnableConfig = None, **kwargs) -> dict:
        try:
            event_msg = self.get_input_streaming_event(config=config)
            return {"result": event_msg.data}
        finally:
            if self.finished_event:
                self.finished_event.set()


def create_streaming_message(data: dict) -> StreamingEventMessage:
    return StreamingEventMessage(entity_id=TEST_ENTITY_ID, data=data, event=TEST_EVENT)


@pytest.fixture
def node_sync_result():
    return RunnableResult(
        status=RunnableStatus.SUCCESS, input={"input": "test_input"}, output={"output": "sync_output"}
    )


@pytest.fixture
def node_async_result():
    return RunnableResult(
        status=RunnableStatus.SUCCESS, input={"input": "test_input"}, output={"output": "async_output"}
    )


@pytest.fixture
def openai_node(mocker, node_sync_result, node_async_result):
    mocker.patch("dynamiq.nodes.llms.base.BaseLLM.run_sync", return_value=node_sync_result)
    mocker.patch("dynamiq.nodes.node.Node.run_async", return_value=node_async_result)
    yield OpenAI(model="gpt-4", connection=OpenAIConnection(api_key="test_api_key"))


def test_run_in_sync_context(openai_node, node_sync_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    result = openai_node.run(input_data, config)

    openai_node.run_sync.assert_called_once_with(input_data, config)
    openai_node.run_async.assert_not_called()
    assert result == node_sync_result


def test_run_in_sync_runtime_and_async_context(openai_node, node_async_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    async def run_async(*args, **kwargs):
        return await openai_node.run(*args, **kwargs)

    result = asyncio.run(run_async(input_data, config))

    openai_node.run_async.assert_called_once_with(input_data, config)
    openai_node.run_sync.assert_not_called()
    assert result == node_async_result


@pytest.mark.asyncio
async def test_run_in_async_context(openai_node, node_async_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    result = await openai_node.run(input_data, config)

    openai_node.run_async.assert_called_once_with(input_data, config)
    openai_node.run_sync.assert_not_called()
    assert result == node_async_result


@pytest.mark.asyncio
async def test_run_in_async_runtime_and_sync_context(openai_node, node_sync_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    result = await asyncio.to_thread(openai_node.run, input_data, config)

    openai_node.run_sync.assert_called_once_with(input_data, config)
    openai_node.run_async.assert_not_called()
    assert result == node_sync_result


def test_run_with_explicit_sync_flag(openai_node, node_sync_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    result = openai_node.run(input_data, config, is_async=False)

    openai_node.run_sync.assert_called_once_with(input_data, config)
    openai_node.run_async.assert_not_called()
    assert result == node_sync_result


@pytest.mark.asyncio
async def test_run_with_explicit_async_flag(openai_node, node_async_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    result = await openai_node.run(input_data, config, is_async=True)

    openai_node.run_async.assert_called_once_with(input_data, config)
    openai_node.run_sync.assert_not_called()
    assert result == node_async_result


def test_run_with_timeout_success():
    node = SlowExecuteNode(sleep_time=0.1)
    node.error_handling.timeout_seconds = 10.0

    result = node.run(input_data=TEST_INPUT_DATA, config=RunnableConfig())

    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"result": "completed"}


def test_run_with_timeout_returns_failure_on_timeout():
    node = SlowExecuteNode(sleep_time=30.0)
    node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
    node.error_handling.max_retries = 0

    result = node.run(input_data=TEST_INPUT_DATA, config=RunnableConfig())

    assert result.status == RunnableStatus.FAILURE
    assert result.error is not None


def test_run_with_timeout_does_not_hang():
    node = SlowExecuteNode(sleep_time=30.0)
    node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
    node.error_handling.max_retries = 0

    start_time = time.time()
    node.run(input_data=TEST_INPUT_DATA, config=RunnableConfig())
    elapsed_time = time.time() - start_time

    assert elapsed_time < MAX_ELAPSED_TIME, (
        f"run took {elapsed_time:.2f}s, expected < {MAX_ELAPSED_TIME}s. "
        "This suggests executor.shutdown is waiting for threads to complete."
    )


def test_get_input_streaming_event_with_short_timeout():
    node = BlockingQueueNode()
    input_queue = Queue()
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=SHORT_TIMEOUT)

    config = RunnableConfig()

    start_time = time.time()
    with pytest.raises(ValueError) as exc_info:
        node.get_input_streaming_event(config=config)
    elapsed_time = time.time() - start_time

    assert "timeout" in str(exc_info.value).lower()
    assert elapsed_time >= SHORT_TIMEOUT - TIMING_TOLERANCE
    assert elapsed_time < MAX_ELAPSED_TIME


def test_get_input_streaming_event_returns_valid_message():
    node = BlockingQueueNode()
    input_queue = Queue()
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=STREAMING_TIMEOUT_LONG)

    config = RunnableConfig()

    message = create_streaming_message({"content": "test content"})
    input_queue.put(message.model_dump_json())

    result = node.get_input_streaming_event(config=config)

    assert result.entity_id == TEST_ENTITY_ID
    assert result.data == {"content": "test content"}


def test_get_input_streaming_event_raises_when_not_enabled():
    node = BlockingQueueNode()
    node.streaming = StreamingConfig(enabled=False)

    config = RunnableConfig()

    with pytest.raises(ValueError) as exc_info:
        node.get_input_streaming_event(config=config)

    assert "not enabled" in str(exc_info.value).lower()


def test_get_input_streaming_event_exits_when_done_event_set():
    node = BlockingQueueNode()
    input_queue = Queue()
    done_event = Event()
    node.streaming = StreamingConfig(
        enabled=True, input_queue=input_queue, input_queue_done_event=done_event, timeout=SHORT_TIMEOUT
    )

    config = RunnableConfig()
    done_event.set()

    with pytest.raises(ValueError) as exc_info:
        node.get_input_streaming_event(config=config)

    assert "completed" in str(exc_info.value).lower()


def test_get_input_streaming_event_skips_invalid_messages():
    node = BlockingQueueNode()
    input_queue = Queue()
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=STREAMING_TIMEOUT_LONG)

    config = RunnableConfig()

    input_queue.put("invalid json")
    message = create_streaming_message({"content": "valid"})
    input_queue.put(message.model_dump_json())

    result = node.get_input_streaming_event(config=config)

    assert result.entity_id == TEST_ENTITY_ID
    assert result.data == {"content": "valid"}


def test_blocking_queue_node_with_streaming_timeout_does_not_hang():
    node = BlockingQueueNode()
    input_queue = Queue()
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=SHORT_TIMEOUT)

    start_time = time.time()
    result = node.run(input_data=TEST_INPUT_DATA, config=RunnableConfig())
    elapsed_time = time.time() - start_time

    assert result.status == RunnableStatus.FAILURE
    assert "timeout" in str(result.error.message).lower()
    assert elapsed_time >= SHORT_TIMEOUT - TIMING_TOLERANCE
    assert elapsed_time < MAX_ELAPSED_TIME


def test_blocking_queue_node_with_execute_timeout_does_not_hang():
    node = BlockingQueueNode()
    input_queue = Queue()
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=STREAMING_TIMEOUT_LONG)
    node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
    node.error_handling.max_retries = 0

    start_time = time.time()
    result = node.run(input_data=TEST_INPUT_DATA, config=RunnableConfig())
    elapsed_time = time.time() - start_time

    assert result.status == RunnableStatus.FAILURE
    assert elapsed_time < MAX_ELAPSED_TIME, (
        f"run took {elapsed_time:.2f}s, expected < {MAX_ELAPSED_TIME}s. "
        "The executor might be waiting for the blocked thread."
    )


@pytest.mark.asyncio
async def test_blocking_queue_node_run_async_does_not_hang():
    node = BlockingQueueNode()
    input_queue = Queue()
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=STREAMING_TIMEOUT_LONG)
    node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
    node.error_handling.max_retries = 0

    start_time = time.time()
    result = await node.run_async(input_data=TEST_INPUT_DATA, config=RunnableConfig())
    elapsed_time = time.time() - start_time

    assert result.status == RunnableStatus.FAILURE
    assert result.error is not None

    assert elapsed_time < MAX_ELAPSED_TIME, (
        f"run_async took {elapsed_time:.2f}s, expected < {MAX_ELAPSED_TIME}s. "
        "This indicates the executor is waiting for blocked streaming thread."
    )


def test_blocking_queue_node_receives_data_before_timeout():
    node = BlockingQueueNode()
    input_queue = Queue()
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=STREAMING_TIMEOUT_LONG)

    message = create_streaming_message({"content": "test data"})
    input_queue.put(message.model_dump_json())

    result = node.run(input_data=TEST_INPUT_DATA, config=RunnableConfig())

    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"result": {"content": "test data"}}


@pytest.mark.asyncio
async def test_run_async_does_not_hang_on_timeout():
    node = SlowExecuteNode(sleep_time=30.0)
    node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
    node.error_handling.max_retries = 0

    start_time = time.time()
    result = await node.run_async(input_data=TEST_INPUT_DATA, config=RunnableConfig())
    elapsed_time = time.time() - start_time

    assert result.status == RunnableStatus.FAILURE
    assert result.error is not None

    assert elapsed_time < MAX_ELAPSED_TIME, (
        f"run_async took {elapsed_time:.2f}s, expected < {MAX_ELAPSED_TIME}s. "
        "This indicates a potential infinite wait issue."
    )


@pytest.mark.asyncio
async def test_multiple_async_nodes_with_timeout_do_not_block_each_other():
    nodes = [SlowExecuteNode(id=f"node-{i}", sleep_time=30.0) for i in range(3)]

    for node in nodes:
        node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
        node.error_handling.max_retries = 0

    start_time = time.time()

    tasks = [node.run_async(input_data=TEST_INPUT_DATA, config=RunnableConfig()) for node in nodes]
    results = await asyncio.gather(*tasks)

    elapsed_time = time.time() - start_time

    for result in results:
        assert result.status == RunnableStatus.FAILURE

    assert elapsed_time < MAX_ELAPSED_TIME, (
        f"Concurrent run_async took {elapsed_time:.2f}s, expected < {MAX_ELAPSED_TIME}s. "
        "This indicates blocking between node executions."
    )


def test_workflow_cancellation_while_waiting_for_input_does_not_hang():
    """
    Simulates workflow cancellation while node is waiting for input messages.

    This test verifies that when:
    1. A node is blocked on get_input_streaming_event waiting for input
    2. The execute timeout triggers (simulating workflow cancellation)

    The main flow returns quickly without waiting for the background thread.
    The background thread will eventually terminate when the streaming timeout expires.
    """
    node = BlockingQueueNode()
    input_queue = Queue()
    # Streaming timeout is longer than execute timeout
    # This simulates a node waiting for user input that never arrives
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=STREAMING_TIMEOUT_LONG)
    # Short execute timeout simulates workflow cancellation/timeout
    node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
    node.error_handling.max_retries = 0

    start_time = time.time()
    result = node.run(input_data=TEST_INPUT_DATA, config=RunnableConfig())
    elapsed_time = time.time() - start_time

    assert result.status == RunnableStatus.FAILURE

    # Critical: main flow should return quickly, NOT wait for streaming timeout
    assert elapsed_time < EXECUTE_TIMEOUT + MAX_ELAPSED_TIME, (
        f"run took {elapsed_time:.2f}s, expected < {EXECUTE_TIMEOUT + MAX_ELAPSED_TIME}s. "
        f"If it took ~{STREAMING_TIMEOUT_LONG}s, the executor.shutdown is waiting for the blocked thread."
    )


@pytest.mark.asyncio
async def test_workflow_cancellation_async_while_waiting_for_input_does_not_hang():
    """
    Async version: simulates workflow cancellation while node is waiting for input messages.
    """
    node = BlockingQueueNode()
    input_queue = Queue()
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=STREAMING_TIMEOUT_LONG)
    node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
    node.error_handling.max_retries = 0

    start_time = time.time()
    result = await node.run_async(input_data=TEST_INPUT_DATA, config=RunnableConfig())
    elapsed_time = time.time() - start_time

    assert result.status == RunnableStatus.FAILURE
    assert result.error is not None

    assert elapsed_time < EXECUTE_TIMEOUT + MAX_ELAPSED_TIME, (
        f"run_async took {elapsed_time:.2f}s, expected < {EXECUTE_TIMEOUT + MAX_ELAPSED_TIME}s. "
        "This indicates the executor is waiting for the blocked streaming thread."
    )


@pytest.mark.asyncio
async def test_multiple_workflow_cancellations_do_not_accumulate_blocked_threads():
    """
    Verifies that multiple workflow cancellations don't cause thread accumulation issues.

    When multiple nodes are cancelled while waiting for input, they should all
    return quickly without blocking each other.
    """
    nodes = []
    for i in range(3):
        node = BlockingQueueNode(id=f"blocking-node-{i}")
        input_queue = Queue()
        node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=STREAMING_TIMEOUT_LONG)
        node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
        node.error_handling.max_retries = 0
        nodes.append(node)

    start_time = time.time()

    tasks = [node.run_async(input_data=TEST_INPUT_DATA, config=RunnableConfig()) for node in nodes]
    results = await asyncio.gather(*tasks)

    elapsed_time = time.time() - start_time

    for result in results:
        assert result.status == RunnableStatus.FAILURE

    assert elapsed_time < MAX_ELAPSED_TIME, (
        f"Multiple cancellations took {elapsed_time:.2f}s, expected < {MAX_ELAPSED_TIME}s. "
        "Background threads might be blocking each other or causing accumulation issues."
    )


@pytest.mark.asyncio
async def test_asyncio_task_cancel_does_not_hang():
    """
    Verifies that when an asyncio task running run_async is explicitly cancelled,
    the cancellation is propagated and doesn't hang.

    Note: The underlying thread will continue running until its timeout expires.
    This is a Python limitation - threads cannot be interrupted.
    """
    node = BlockingQueueNode()
    input_queue = Queue()
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=STREAMING_TIMEOUT_LONG)
    # No execute timeout - we rely on task.cancel()
    node.error_handling.timeout_seconds = None
    node.error_handling.max_retries = 0

    start_time = time.time()

    task = asyncio.create_task(node.run_async(input_data=TEST_INPUT_DATA, config=RunnableConfig()))

    # Give the task time to start and block on input_queue
    await asyncio.sleep(CANCEL_DELAY)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    elapsed_time = time.time() - start_time

    assert elapsed_time < MAX_ELAPSED_TIME, (
        f"Task cancellation took {elapsed_time:.2f}s, expected < {MAX_ELAPSED_TIME}s. "
        f"If it took ~{STREAMING_TIMEOUT_LONG}s, the task didn't respond to cancel()."
    )


@pytest.mark.asyncio
async def test_asyncio_task_cancel_with_slow_node_does_not_hang():
    """
    Verifies that cancelling an asyncio task with a slow node
    returns immediately without waiting for the node to complete.
    """
    node = SlowExecuteNode(sleep_time=30.0)
    node.error_handling.timeout_seconds = None
    node.error_handling.max_retries = 0

    start_time = time.time()

    task = asyncio.create_task(node.run_async(input_data=TEST_INPUT_DATA, config=RunnableConfig()))

    await asyncio.sleep(CANCEL_DELAY)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    elapsed_time = time.time() - start_time

    assert elapsed_time < MAX_ELAPSED_TIME, (
        f"Task cancellation took {elapsed_time:.2f}s, expected < {MAX_ELAPSED_TIME}s. "
        "Task didn't respond to cancel()."
    )


@pytest.mark.asyncio
async def test_multiple_asyncio_task_cancels_do_not_hang():
    """
    Verifies that cancelling multiple asyncio tasks at once
    all respond quickly to cancellation.
    """
    nodes = []
    for i in range(3):
        node = BlockingQueueNode(id=f"cancel-node-{i}")
        input_queue = Queue()
        node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=STREAMING_TIMEOUT_LONG)
        node.error_handling.timeout_seconds = None
        node.error_handling.max_retries = 0
        nodes.append(node)

    start_time = time.time()

    tasks = [asyncio.create_task(node.run_async(input_data=TEST_INPUT_DATA, config=RunnableConfig())) for node in nodes]

    await asyncio.sleep(CANCEL_DELAY)

    for task in tasks:
        task.cancel()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed_time = time.time() - start_time
    for result in results:
        assert isinstance(result, asyncio.CancelledError)

    assert (
        elapsed_time < MAX_ELAPSED_TIME
    ), f"Multiple task cancellations took {elapsed_time:.2f}s, expected < {MAX_ELAPSED_TIME}s."


def test_background_thread_runs_until_streaming_timeout_after_execute_timeout():
    """
    Verifies that when execute timeout triggers, the background thread
    continues running until the streaming timeout expires.

    This confirms the Python limitation: threads cannot be interrupted.
    The fix ensures the main flow returns immediately, but the background
    thread will keep running until streaming.timeout expires.
    """
    finished_event = Event()
    node = TrackableBlockingNode()
    node.finished_event = finished_event
    input_queue = Queue()

    # Streaming timeout is longer than execute timeout
    node.streaming = StreamingConfig(enabled=True, input_queue=input_queue, timeout=SHORT_STREAMING_TIMEOUT)
    node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
    node.error_handling.max_retries = 0

    start_time = time.time()
    result = node.run(input_data=TEST_INPUT_DATA, config=RunnableConfig())
    main_flow_elapsed = time.time() - start_time

    # Main flow returns quickly with timeout
    assert result.status == RunnableStatus.FAILURE
    assert main_flow_elapsed < MAX_ELAPSED_TIME

    # Background thread should NOT have finished yet (it's waiting on streaming timeout)
    assert (
        not finished_event.is_set()
    ), "Background thread finished too early. It should still be waiting on streaming timeout."

    # Wait for background thread to finish (streaming timeout)
    finished = finished_event.wait(timeout=BACKGROUND_THREAD_WAIT)
    background_elapsed = time.time() - start_time

    assert finished, (
        f"Background thread did not finish within {BACKGROUND_THREAD_WAIT}s. "
        "It might be stuck or streaming timeout is not working."
    )

    # Background should have waited for the streaming timeout
    assert background_elapsed >= SHORT_STREAMING_TIMEOUT - TIMING_TOLERANCE, (
        f"Background thread finished in {background_elapsed:.2f}s, "
        f"expected to wait for streaming timeout (~{SHORT_STREAMING_TIMEOUT}s)."
    )


def test_background_thread_can_be_unblocked_by_done_event():
    """
    Verifies that setting input_queue_done_event can unblock the background thread
    sooner than the streaming timeout.

    This provides a way to gracefully terminate background threads when a workflow is cancelled.
    """
    finished_event = Event()
    done_event = Event()
    node = TrackableBlockingNode()
    node.finished_event = finished_event
    input_queue = Queue()

    # Long streaming timeout, but we'll signal done_event to unblock
    # Use short poll interval so done_event is detected quickly
    node.streaming = StreamingConfig(
        enabled=True,
        input_queue=input_queue,
        input_queue_done_event=done_event,
        input_queue_poll_interval=SHORT_POLL_INTERVAL,
        timeout=STREAMING_TIMEOUT_LONG,
    )
    node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
    node.error_handling.max_retries = 0

    start_time = time.time()
    result = node.run(input_data=TEST_INPUT_DATA, config=RunnableConfig())
    main_flow_elapsed = time.time() - start_time

    assert result.status == RunnableStatus.FAILURE
    assert main_flow_elapsed < MAX_ELAPSED_TIME

    # Background thread should still be waiting
    assert not finished_event.is_set()

    done_event.set()

    # Background thread should finish quickly after done_event is set
    finished = finished_event.wait(timeout=BACKGROUND_THREAD_WAIT)
    background_elapsed = time.time() - start_time

    assert finished, "Background thread did not respond to done_event."
    assert background_elapsed < STREAMING_TIMEOUT_LONG, (
        f"Background thread took {background_elapsed:.2f}s, "
        f"should have finished quickly after done_event (~{SHORT_STREAMING_TIMEOUT}s)."
    )


def test_input_queue_poll_interval_default_value():
    """Verify default poll interval is 5.0 seconds."""
    config = StreamingConfig(enabled=True, input_queue=Queue())
    assert config.input_queue_poll_interval == 5.0


def test_input_queue_poll_interval_custom_value():
    """Verify custom poll interval is respected."""
    config = StreamingConfig(enabled=True, input_queue=Queue(), input_queue_poll_interval=SHORT_POLL_INTERVAL)
    assert config.input_queue_poll_interval == SHORT_POLL_INTERVAL


def test_done_event_response_time_within_poll_interval():
    """
    Verifies that done_event is detected within poll_interval.

    When done_event is set, the polling loop should exit within one poll interval,
    not wait for the full streaming timeout.
    """
    node = BlockingQueueNode()
    input_queue = Queue()
    done_event = Event()

    node.streaming = StreamingConfig(
        enabled=True,
        input_queue=input_queue,
        input_queue_done_event=done_event,
        input_queue_poll_interval=SHORT_POLL_INTERVAL,
        timeout=STREAMING_TIMEOUT_LONG,
    )

    config = RunnableConfig()

    def set_done_after_delay():
        time.sleep(0.1)
        done_event.set()

    threading.Thread(target=set_done_after_delay, daemon=True).start()

    start_time = time.time()
    with pytest.raises(ValueError) as exc_info:
        node.get_input_streaming_event(config=config)
    elapsed_time = time.time() - start_time

    assert "completed" in str(exc_info.value).lower()
    # Should respond within poll_interval + some tolerance, not wait for streaming timeout
    assert elapsed_time < SHORT_POLL_INTERVAL + 0.3, (
        f"Took {elapsed_time:.2f}s to respond to done_event, "
        f"expected within poll_interval ({SHORT_POLL_INTERVAL}s) + tolerance."
    )


def test_longer_poll_interval_delays_done_event_detection():
    """
    Verifies that a longer poll_interval delays done_event detection.

    This confirms the polling mechanism is actually using the configured interval.
    """
    node = BlockingQueueNode()
    input_queue = Queue()
    done_event = Event()

    node.streaming = StreamingConfig(
        enabled=True,
        input_queue=input_queue,
        input_queue_done_event=done_event,
        input_queue_poll_interval=LONG_POLL_INTERVAL,
        timeout=STREAMING_TIMEOUT_LONG,
    )

    config = RunnableConfig()

    def set_done_after_delay():
        time.sleep(0.1)
        done_event.set()

    threading.Thread(target=set_done_after_delay, daemon=True).start()

    start_time = time.time()
    with pytest.raises(ValueError) as exc_info:
        node.get_input_streaming_event(config=config)
    elapsed_time = time.time() - start_time

    assert "completed" in str(exc_info.value).lower()
    # With longer poll interval, detection should take longer (at least close to poll interval)
    # We set done_event after 0.1s, but detection happens after the current poll completes
    assert (
        elapsed_time < LONG_POLL_INTERVAL + 0.5
    ), f"Took {elapsed_time:.2f}s, expected less than poll_interval ({LONG_POLL_INTERVAL}s) + tolerance."


def test_execute_timeout_with_custom_poll_interval():
    """
    Verifies that execute timeout works correctly with custom poll_interval.

    The execute timeout should trigger regardless of the poll_interval setting.
    """
    finished_event = Event()
    node = TrackableBlockingNode()
    node.finished_event = finished_event
    input_queue = Queue()

    node.streaming = StreamingConfig(
        enabled=True,
        input_queue=input_queue,
        input_queue_poll_interval=SHORT_POLL_INTERVAL,
        timeout=STREAMING_TIMEOUT_LONG,
    )
    node.error_handling.timeout_seconds = EXECUTE_TIMEOUT
    node.error_handling.max_retries = 0

    start_time = time.time()
    result = node.run(input_data=TEST_INPUT_DATA, config=RunnableConfig())
    main_flow_elapsed = time.time() - start_time

    assert result.status == RunnableStatus.FAILURE
    assert main_flow_elapsed < MAX_ELAPSED_TIME

    # Background thread should still be running (blocked on polling)
    assert not finished_event.is_set()


def test_streaming_timeout_respects_poll_interval_accumulation():
    """
    Verifies that streaming timeout is correctly accumulated across poll iterations.

    With a short poll interval and short streaming timeout, the timeout should
    still trigger at the correct total time.
    """
    node = BlockingQueueNode()
    input_queue = Queue()
    streaming_timeout = 0.5

    node.streaming = StreamingConfig(
        enabled=True,
        input_queue=input_queue,
        input_queue_poll_interval=SHORT_POLL_INTERVAL,
        timeout=streaming_timeout,
    )

    config = RunnableConfig()

    start_time = time.time()
    with pytest.raises(ValueError) as exc_info:
        node.get_input_streaming_event(config=config)
    elapsed_time = time.time() - start_time

    assert "timeout" in str(exc_info.value).lower()
    assert (
        elapsed_time >= streaming_timeout - TIMING_TOLERANCE
    ), f"Timed out in {elapsed_time:.2f}s, expected around {streaming_timeout}s."
    assert (
        elapsed_time < streaming_timeout + TIMING_TOLERANCE + SHORT_POLL_INTERVAL
    ), f"Timed out in {elapsed_time:.2f}s, expected around {streaming_timeout}s."


class SharedResource(BaseModel):
    """A resource marked as clone-shared; Node.clone() should reuse by reference."""

    _clone_shared: ClassVar[bool] = True
    name: str = "shared"


class RegularResource(BaseModel):
    """A normal BaseModel without _clone_shared; Node.clone() should deep-copy it."""

    name: str = "regular"


class NodeWithSharedField(Node):
    """Minimal concrete Node that holds both shared and regular resources."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "NodeWithSharedField"
    shared: SharedResource = Field(default_factory=SharedResource)
    regular: RegularResource = Field(default_factory=RegularResource)

    def execute(self, input_data: dict, config: RunnableConfig = None, **kwargs) -> dict:
        return {"ok": True}


class NodeWithNestedShared(Node):
    """Node where the shared resource lives inside a list and a dict."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "NodeWithNestedShared"
    shared_list: list[SharedResource] = Field(default_factory=lambda: [SharedResource(name="in-list")])
    shared_dict: dict[str, SharedResource] = Field(default_factory=lambda: {"key": SharedResource(name="in-dict")})

    def execute(self, input_data: dict, config: RunnableConfig = None, **kwargs) -> dict:
        return {"ok": True}


def test_clone_regular_object_is_different_reference():
    """Regular BaseModel fields (no _clone_shared) must be deep-copied."""
    node = NodeWithSharedField()
    cloned = node.clone()

    assert (
        cloned.regular is not node.regular
    ), "RegularResource without _clone_shared should have been deep-copied but was shared."
    assert cloned.regular.name == node.regular.name


def test_clone_shared_in_list_is_same_reference():
    """_clone_shared objects nested inside a list must stay shared."""
    node = NodeWithNestedShared()
    cloned = node.clone()

    assert (
        cloned.shared_list[0] is node.shared_list[0]
    ), "SharedResource inside a list was deep-copied instead of shared by reference."


def test_clone_shared_in_dict_is_same_reference():
    """_clone_shared objects nested inside a dict must stay shared."""
    node = NodeWithNestedShared()
    cloned = node.clone()

    assert (
        cloned.shared_dict["key"] is node.shared_dict["key"]
    ), "SharedResource inside a dict was deep-copied instead of shared by reference."


def test_clone_shared_survives_multiple_clones():
    """Cloning several times should always yield the same shared reference."""
    node = NodeWithSharedField()
    clone1 = node.clone()
    clone2 = node.clone()
    clone3 = clone1.clone()

    assert clone1.shared is node.shared
    assert clone2.shared is node.shared
    assert clone3.shared is node.shared, "Shared reference was lost when cloning a clone."
