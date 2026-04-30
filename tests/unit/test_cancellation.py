import threading
import time
from queue import Queue
from threading import Event
from uuid import uuid4

import pytest

from dynamiq.callbacks.tracing import RunStatus, TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.node import Node, NodeDependency, NodeGroup
from dynamiq.nodes.operators.operators import Map
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.cancellation import CanceledException, CancellationConfig, CancellationToken, check_cancellation
from dynamiq.types.streaming import StreamingConfig
from dynamiq.workflow import Workflow

SHORT_SLEEP = 0.1
CANCEL_DELAY = 0.3
THREAD_JOIN_TIMEOUT = 10.0


class InstantNode(Node):
    """Node that completes immediately."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "instant-node"

    def execute(self, input_data, config=None, **kwargs):
        return {"result": "instant"}


class SlowNode(Node):
    """Node that sleeps in small increments with cancellation checks.

    Unlike a real blocking node, this checks cancellation between
    short sleeps so tests can cancel it promptly.
    """

    group: NodeGroup = NodeGroup.UTILS
    name: str = "slow-node"
    sleep_time: float = 5.0

    def execute(self, input_data, config=None, **kwargs):
        elapsed = 0.0
        while elapsed < self.sleep_time:
            check_cancellation(config)
            time.sleep(0.05)
            elapsed += 0.05
        return {"result": "completed"}


class LoopingNode(Node):
    """Node that loops with cancellation checks between iterations."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "looping-node"
    iterations: int = 100
    completed: int = 0

    def execute(self, input_data, config=None, **kwargs):
        self.completed = 0
        for i in range(self.iterations):
            check_cancellation(config)
            time.sleep(0.02)
            self.completed = i + 1
        return {"result": f"completed {self.iterations} iterations"}


class CountingNode(Node):
    """Node that counts up and records how many items it processed."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "counting-node"

    def execute(self, input_data, config=None, **kwargs):
        check_cancellation(config)
        return {"result": f"processed-{input_data.get('index', '?')}"}


class FailingNode(Node):
    """Node that always raises."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "failing-node"

    def execute(self, input_data, config=None, **kwargs):
        raise ValueError("intentional failure")


class RetryableNode(Node):
    """Node that fails N times then succeeds, for testing retry cancellation."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "retryable-node"
    fail_count: int = 3
    _attempt: int = 0

    def execute(self, input_data, config=None, **kwargs):
        self._attempt += 1
        if self._attempt <= self.fail_count:
            raise ValueError(f"attempt {self._attempt} failed")
        return {"result": "finally succeeded"}


class TestCancellationToken:
    def test_initial_state(self):
        token = CancellationToken()
        assert token.is_canceled is False

    def test_cancel_sets_state(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_canceled is True

    def test_cancel_idempotent(self):
        token = CancellationToken()
        token.cancel()
        token.cancel()  # second call should be a no-op
        assert token.is_canceled is True

    def test_thread_safety_concurrent_cancels(self):
        token = CancellationToken()
        barrier = threading.Barrier(10)

        def cancel_from_thread():
            barrier.wait()
            token.cancel()

        threads = [threading.Thread(target=cancel_from_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert token.is_canceled is True

    def test_cancel_from_different_thread(self):
        token = CancellationToken()
        done = Event()

        def background():
            time.sleep(0.1)
            token.cancel()
            done.set()

        threading.Thread(target=background, daemon=True).start()
        done.wait(timeout=2.0)
        assert token.is_canceled is True


class TestCancellationConfig:
    def test_default_config_has_fresh_token(self):
        config = CancellationConfig()
        assert config.token is not None
        assert config.token.is_canceled is False

    def test_each_default_config_gets_unique_token(self):
        c1 = CancellationConfig()
        c2 = CancellationConfig()
        assert c1.token is not c2.token

    def test_uncanceled_token_check_is_noop(self):
        config = CancellationConfig(token=CancellationToken())
        config.check()  # Must NOT raise

    def test_canceled_token_check_raises(self):
        token = CancellationToken()
        token.cancel()
        config = CancellationConfig(token=token)
        with pytest.raises(CanceledException):
            config.check()

    def test_is_canceled_property_tracks_token(self):
        token = CancellationToken()
        config = CancellationConfig(token=token)
        assert config.is_canceled is False
        token.cancel()
        assert config.is_canceled is True

    def test_to_dict_not_canceled(self):
        config = CancellationConfig(token=CancellationToken())
        assert config.to_dict() == {"canceled": False}

    def test_to_dict_canceled(self):
        token = CancellationToken()
        token.cancel()
        config = CancellationConfig(token=token)
        assert config.to_dict() == {"canceled": True}

    def test_custom_token_can_be_passed(self):
        token = CancellationToken()
        config = CancellationConfig(token=token)
        assert config.token is token


class TestCheckCancellation:
    def test_none_config_is_noop(self):
        check_cancellation(None)

    def test_default_config_with_uncanceled_token_is_noop(self):
        """Default RunnableConfig() has a fresh token that is not canceled — no-op."""
        check_cancellation(RunnableConfig())

    def test_config_with_canceled_token_raises(self):
        token = CancellationToken()
        token.cancel()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        with pytest.raises(CanceledException):
            check_cancellation(config)

    def test_config_with_active_uncanceled_token_is_noop(self):
        config = RunnableConfig(cancellation=CancellationConfig(token=CancellationToken()))
        check_cancellation(config)  # Must NOT raise


class TestRunnableConfigCancellation:
    def test_cancellation_always_on_by_default(self):
        """Every RunnableConfig has a fresh CancellationConfig with a token."""
        config = RunnableConfig()
        assert config.cancellation is not None
        assert config.cancellation.token is not None
        assert config.cancellation.is_canceled is False

    def test_each_runnable_config_gets_unique_token(self):
        """Each RunnableConfig instance gets its own CancellationToken."""
        config1 = RunnableConfig()
        config2 = RunnableConfig()
        assert config1.cancellation.token is not config2.cancellation.token

    def test_flow_setup_cancellation_when_config_none(self):
        """Flow._setup_cancellation creates a config with cancellation when passed None."""
        config = Flow._setup_cancellation(None)
        assert config.cancellation is not None
        assert config.cancellation.token is not None

    def test_flow_setup_cancellation_preserves_existing_token(self):
        """Flow._setup_cancellation does not overwrite an existing cancellation config."""
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        result = Flow._setup_cancellation(config)
        assert result.cancellation.token is token

    def test_cancellation_field_accepts_custom_config(self):
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        assert config.cancellation.token is token

    def test_config_copy_shares_token(self):
        """Shallow copy of RunnableConfig should share the same CancellationToken."""
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        config2 = config.model_copy(deep=False)
        token.cancel()
        assert config2.cancellation.is_canceled is True


class TestNodeCancellation:
    def test_pre_cancelled_returns_cancelled_status(self):
        token = CancellationToken()
        token.cancel()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))

        node = InstantNode()
        result = node.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.CANCELED
        assert result.error is not None
        assert result.error.type is CanceledException
        assert result.error.message
        assert result.output is None
        assert result.input is None

    def test_mid_execution_looping_node_cancelled(self):
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        node = LoopingNode(iterations=1000)
        result_holder = {}

        def run():
            result_holder["result"] = node.run_sync(input_data={}, config=config)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(CANCEL_DELAY)
        token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        result = result_holder["result"]
        assert result.status == RunnableStatus.CANCELED

    def test_no_cancellation_config_works_normally(self):
        node = InstantNode()
        result = node.run_sync(input_data={}, config=RunnableConfig())
        assert result.status == RunnableStatus.SUCCESS

    def test_uncanceled_token_does_not_block_execution(self):
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        node = InstantNode()
        result = node.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.SUCCESS

    def test_canceled_result_has_descriptive_error(self):
        token = CancellationToken()
        token.cancel()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        node = InstantNode()
        result = node.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.CANCELED
        assert result.error is not None
        assert result.error.type is CanceledException
        assert result.error.message  # has a descriptive message
        assert result.output is None
        assert result.input is None


class TestExecuteWithRetryCancellation:
    def test_cancel_between_retries(self):
        """Cancel should be checked before each retry attempt."""
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        node = RetryableNode(
            fail_count=5,
            error_handling={"max_retries": 10, "retry_interval_seconds": 0.5},
        )
        result_holder = {}

        def run():
            result_holder["result"] = node.run_sync(input_data={}, config=config)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(1.0)
        token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        result = result_holder["result"]
        # Should be CANCELED (caught CanceledException) not FAILURE from the retries
        assert result.status == RunnableStatus.CANCELED


class TestFlowCancellation:
    def test_pre_cancelled_flow(self):
        token = CancellationToken()
        token.cancel()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        flow = Flow(nodes=[InstantNode()])
        result = flow.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.CANCELED

    def test_flow_cancelled_between_sequential_nodes(self):
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))

        fast = InstantNode(name="fast")
        slow = SlowNode(name="slow", sleep_time=10.0)
        slow.depends = [NodeDependency(node=fast)]

        flow = Flow(nodes=[fast, slow])
        result_holder = {}

        def run():
            result_holder["result"] = flow.run_sync(input_data={}, config=config)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(CANCEL_DELAY)
        token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        result = result_holder["result"]
        assert result.status == RunnableStatus.CANCELED
        # Cancellation returns no output, no error
        assert result.output is None
        assert result.error is not None
        assert result.error.type is CanceledException
        assert result.error.message  # has a descriptive message

    def test_flow_canceled_result_has_no_output(self):
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))

        node1 = InstantNode(name="node1")
        node2 = InstantNode(name="node2")
        node3 = SlowNode(name="node3", sleep_time=10.0)
        node2.depends = [NodeDependency(node=node1)]
        node3.depends = [NodeDependency(node=node2)]

        flow = Flow(nodes=[node1, node2, node3])
        result_holder = {}

        def run():
            result_holder["result"] = flow.run_sync(input_data={}, config=config)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.5)
        token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        result = result_holder["result"]
        assert result.status == RunnableStatus.CANCELED
        # Canceled result is fully empty: no input, output, or error
        assert result.output is None
        assert result.error is not None
        assert result.error.type is CanceledException
        assert result.error.message  # has a descriptive message
        assert result.input is None

    def test_flow_not_cancelled_completes_normally(self):
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        flow = Flow(nodes=[InstantNode()])
        result = flow.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.SUCCESS


class TestFlowAsyncCancellation:
    @pytest.mark.asyncio
    async def test_flow_async_pre_cancelled(self):
        token = CancellationToken()
        token.cancel()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        flow = Flow(nodes=[InstantNode()])
        result = await flow.run_async(input_data={}, config=config)
        assert result.status == RunnableStatus.CANCELED

    @pytest.mark.asyncio
    async def test_flow_async_not_cancelled(self):
        config = RunnableConfig(cancellation=CancellationConfig(token=CancellationToken()))
        flow = Flow(nodes=[InstantNode()])
        result = await flow.run_async(input_data={}, config=config)
        assert result.status == RunnableStatus.SUCCESS


class TestWorkflowCancellation:
    def test_workflow_propagates_cancelled_status(self):
        token = CancellationToken()
        token.cancel()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        workflow = Workflow(flow=Flow(nodes=[InstantNode()]))
        result = workflow.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.CANCELED

    def test_workflow_not_cancelled(self):
        config = RunnableConfig(cancellation=CancellationConfig(token=CancellationToken()))
        workflow = Workflow(flow=Flow(nodes=[InstantNode()]))
        result = workflow.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_workflow_async_cancelled(self):
        token = CancellationToken()
        token.cancel()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        workflow = Workflow(flow=Flow(nodes=[InstantNode()]))
        result = await workflow.run_async(input_data={}, config=config)
        assert result.status == RunnableStatus.CANCELED


class TestTracingCancellationCallbacks:
    def test_on_node_canceled(self):
        handler = TracingCallbackHandler()
        run_id = uuid4()
        handler.on_node_start(
            serialized={"name": "test-node", "group": "utils"},
            input_data={},
            run_id=run_id,
        )
        handler.on_node_canceled(
            serialized={"name": "test-node", "group": "utils"},
            run_id=run_id,
        )
        run = handler.runs[run_id]
        assert run.status == RunStatus.CANCELED
        assert run.end_time is not None

    def test_on_node_canceled_creates_run_if_missing(self):
        handler = TracingCallbackHandler()
        run_id = uuid4()
        handler.on_node_canceled(
            serialized={"name": "late-node", "group": "utils"},
            run_id=run_id,
        )
        assert run_id in handler.runs
        assert handler.runs[run_id].status == RunStatus.CANCELED

    def test_on_flow_canceled(self):
        handler = TracingCallbackHandler()
        run_id = uuid4()
        handler.on_flow_start(serialized={"id": "f1"}, input_data={}, run_id=run_id)
        handler.on_flow_canceled(
            serialized={"id": "f1"},
            run_id=run_id,
        )
        run = handler.runs[run_id]
        assert run.status == RunStatus.CANCELED

    def test_on_workflow_canceled(self):
        handler = TracingCallbackHandler()
        run_id = uuid4()
        handler.on_workflow_start(serialized={"id": "wf1"}, input_data={}, run_id=run_id)
        handler.on_workflow_canceled(
            serialized={"id": "wf1"},
            run_id=run_id,
        )
        run = handler.runs[run_id]
        assert run.status == RunStatus.CANCELED


class TestHITLCancellation:
    def test_cancel_during_streaming_input_wait(self):
        """Cancelling while node is blocked on get_input_streaming_event."""
        token = CancellationToken()
        config = RunnableConfig(
            cancellation=CancellationConfig(token=token),
        )
        queue = Queue()
        done_event = Event()
        node = InstantNode(
            streaming=StreamingConfig(
                enabled=True,
                input_queue=queue,
                input_queue_done_event=done_event,
                timeout=30.0,
                input_queue_poll_interval=0.2,
            ),
        )
        error_holder = {}

        def run():
            try:
                node.get_input_streaming_event(config=config)
            except CanceledException as e:
                error_holder["canceled"] = e
            except Exception as e:
                error_holder["other"] = e

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.5)
        token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        assert "canceled" in error_holder
        assert isinstance(error_holder["canceled"], CanceledException)

    def test_hitl_responsive_polling(self):
        """With cancellation active, polling should use shorter intervals."""
        token = CancellationToken()
        config = RunnableConfig(
            cancellation=CancellationConfig(token=token),
        )
        queue = Queue()
        done_event = Event()
        node = InstantNode(
            streaming=StreamingConfig(
                enabled=True,
                input_queue=queue,
                input_queue_done_event=done_event,
                timeout=30.0,
                # Long poll interval, but cancellation should override to 0.5s
                input_queue_poll_interval=10.0,
            ),
        )
        error_holder = {}

        def run():
            try:
                node.get_input_streaming_event(config=config)
            except CanceledException as e:
                error_holder["canceled"] = e
            except Exception as e:
                error_holder["other"] = e

        start = time.time()
        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.3)
        token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)
        elapsed = time.time() - start

        assert "canceled" in error_holder
        # Should have canceled within ~1s despite 10s poll interval
        assert elapsed < 3.0


class TestCancellationTracingIntegration:
    def test_cancelled_node_fires_on_node_canceled_callback(self):
        token = CancellationToken()
        token.cancel()
        handler = TracingCallbackHandler()
        config = RunnableConfig(
            cancellation=CancellationConfig(token=token),
            callbacks=[handler],
        )
        node = InstantNode()
        result = node.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.CANCELED

        canceled = [r for r in handler.runs.values() if r.status == RunStatus.CANCELED]
        assert len(canceled) >= 1

    def test_cancelled_flow_fires_on_flow_canceled_callback(self):
        token = CancellationToken()
        token.cancel()
        handler = TracingCallbackHandler()
        config = RunnableConfig(
            cancellation=CancellationConfig(token=token),
            callbacks=[handler],
        )
        flow = Flow(nodes=[InstantNode()])
        result = flow.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.CANCELED

        canceled = [r for r in handler.runs.values() if r.status == RunStatus.CANCELED]
        assert len(canceled) >= 1

    def test_cancelled_workflow_fires_on_workflow_canceled_callback(self):
        token = CancellationToken()
        token.cancel()
        handler = TracingCallbackHandler()
        config = RunnableConfig(
            cancellation=CancellationConfig(token=token),
            callbacks=[handler],
        )
        workflow = Workflow(flow=Flow(nodes=[InstantNode()]))
        result = workflow.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.CANCELED

        canceled = [r for r in handler.runs.values() if r.status == RunStatus.CANCELED]
        assert len(canceled) >= 1


class TestMapCancellation:
    def test_map_pre_cancelled(self):
        token = CancellationToken()
        token.cancel()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))

        inner_node = CountingNode()
        map_node = Map(node=inner_node, max_workers=2)
        result = map_node.run_sync(
            input_data={"input": [{"index": i} for i in range(10)]},
            config=config,
        )
        assert result.status == RunnableStatus.CANCELED

    def test_map_not_cancelled(self):
        config = RunnableConfig(cancellation=CancellationConfig(token=CancellationToken()))
        inner_node = CountingNode()
        map_node = Map(node=inner_node, max_workers=2)
        result = map_node.run_sync(
            input_data={"input": [{"index": i} for i in range(3)]},
            config=config,
        )
        assert result.status == RunnableStatus.SUCCESS


class TestCancellationEdgeCases:
    def test_cancel_after_completion_is_harmless(self):
        """Cancelling a token after execution completes has no effect."""
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        node = InstantNode()
        result = node.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.SUCCESS
        token.cancel()
        assert token.is_canceled is True
        # The result is already returned and doesn't change
        assert result.status == RunnableStatus.SUCCESS

    def test_failing_node_with_cancellation_returns_failure_not_cancelled(self):
        """A node that fails normally should return FAILURE, not CANCELED."""
        config = RunnableConfig(cancellation=CancellationConfig(token=CancellationToken()))
        node = FailingNode()
        result = node.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.FAILURE

    def test_empty_flow_with_cancellation(self):
        """A flow with no nodes should complete normally even with cancellation."""
        config = RunnableConfig(cancellation=CancellationConfig(token=CancellationToken()))
        flow = Flow(nodes=[])
        result = flow.run_sync(input_data={}, config=config)
        assert result.status == RunnableStatus.SUCCESS
