import threading
import time

from dynamiq import Workflow, connections, flows
from dynamiq.nodes import llms
from dynamiq.nodes.node import InputTransformer, Node, NodeDependency, NodeGroup
from dynamiq.nodes.tools import Python
from dynamiq.nodes.utils import Input, Output
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.cancellation import CanceledException, CancellationConfig, CancellationToken, check_cancellation

from .conftest import mock_llm_success

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"

CANCEL_DELAY = 0.4
THREAD_JOIN_TIMEOUT = 15.0


def make_llm(node_id: str = "llm", depends: list | None = None) -> llms.OpenAI:
    return llms.OpenAI(
        id=node_id,
        name="LLM",
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        prompt=Prompt(messages=[Message(role="user", content="Process: {{query}}")]),
        is_postponed_component_init=True,
        depends=depends or [],
    )


def make_pipeline():
    """Input -> LLM -> Output."""
    inp = Input(id="input", name="Input")
    llm = make_llm(depends=[NodeDependency(inp)])
    out = Output(id="output", name="Output", depends=[NodeDependency(llm)])
    return [inp, llm, out]


class CooperativeSlowNode(Node):
    """Sleeps in increments while checking cancellation between each."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "cooperative-slow"
    sleep_seconds: float = 5.0

    def execute(self, input_data, config=None, **kwargs):
        elapsed = 0.0
        while elapsed < self.sleep_seconds:
            check_cancellation(config)
            time.sleep(0.05)
            elapsed += 0.05
        return {"result": "done"}


def make_slow_pipeline(sleep_seconds: float = 5.0):
    """Input -> CooperativeSlowNode -> Output (cancellable)."""
    inp = Input(id="input", name="Input")
    slow = CooperativeSlowNode(
        id="slow",
        name="Slow",
        sleep_seconds=sleep_seconds,
        depends=[NodeDependency(inp)],
        input_transformer=InputTransformer(selector={"query": "$.input.output.query"}),
    )
    out = Output(id="output", name="Output", depends=[NodeDependency(slow)])
    return [inp, slow, out]


def make_python_pipeline_slow():
    """Input -> Python -> CooperativeSlowNode -> Output."""
    inp = Input(id="input", name="Input")
    py = Python(
        id="python",
        name="Python",
        code='def run(input_data): return {"processed": input_data.get("value", 0) * 10}',
        depends=[NodeDependency(inp)],
    )
    slow = CooperativeSlowNode(
        id="slow",
        name="Slow",
        sleep_seconds=5.0,
        depends=[NodeDependency(py)],
    )
    out = Output(id="output", name="Output", depends=[NodeDependency(slow)])
    return [inp, py, slow, out]


def run_in_thread(target):
    holder = {}

    def runner():
        try:
            holder["result"] = target()
        except Exception as e:
            holder["exception"] = e

    thread = threading.Thread(target=runner)
    thread.start()
    return holder, thread


class TestPreCanceledFlow:
    def test_pre_canceled_pipeline_returns_canceled_immediately(self, mocker, cancellation_token, runnable_config):
        """LLM should never be invoked if the token is pre-canceled."""
        llm_call = mock_llm_success(mocker)
        cancellation_token.cancel()

        flow = flows.Flow(nodes=make_pipeline())
        result = flow.run_sync(input_data={"query": "test"}, config=runnable_config)

        assert result.status == RunnableStatus.CANCELED
        assert result.input is None
        assert result.output is None
        assert result.error is not None
        assert result.error.type is CanceledException
        assert result.error.message  # has a descriptive message
        # LLM should never have been called
        assert llm_call.call_count == 0

    def test_pre_canceled_workflow_returns_canceled(self, mocker, cancellation_token, runnable_config):
        mock_llm_success(mocker)
        cancellation_token.cancel()

        wf = Workflow(flow=flows.Flow(nodes=make_pipeline()))
        result = wf.run_sync(input_data={"query": "test"}, config=runnable_config)

        assert result.status == RunnableStatus.CANCELED


class TestMidExecutionCancellation:
    def test_cancel_during_slow_node_in_pipeline(self, cancellation_token, runnable_config):
        """Input -> CooperativeSlowNode -> Output: cancel while slow node is running."""
        flow = flows.Flow(nodes=make_slow_pipeline(sleep_seconds=5.0))

        def go():
            return flow.run_sync(input_data={"query": "test"}, config=runnable_config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        result = holder["result"]
        assert result.status == RunnableStatus.CANCELED
        # Cancellation produces a fully empty result - no partial output
        assert result.input is None
        assert result.output is None
        assert result.error is not None
        assert result.error.type is CanceledException
        assert result.error.message  # has a descriptive message

    def test_cancel_after_first_node_completes(self, mocker, cancellation_token, runnable_config):
        """Input + Python complete fast; cancel during slow node."""
        flow = flows.Flow(nodes=make_python_pipeline_slow())

        def go():
            return flow.run_sync(input_data={"value": 5}, config=runnable_config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        result = holder["result"]
        assert result.status == RunnableStatus.CANCELED

    def test_cancel_workflow_wrapper(self, cancellation_token, runnable_config):
        """Workflow wrapper should also return CANCELED status."""
        wf = Workflow(flow=flows.Flow(nodes=make_slow_pipeline()))

        def go():
            return wf.run_sync(input_data={"query": "test"}, config=runnable_config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        assert holder["result"].status == RunnableStatus.CANCELED


class TestDefaultCancellationWiring:
    def test_default_runnable_config_has_token(self):
        config = RunnableConfig()
        assert config.cancellation is not None
        assert config.cancellation.token is not None

    def test_workflow_with_default_config_can_be_canceled(self):
        """No explicit CancellationConfig setup needed - just grab the token."""
        config = RunnableConfig()
        token = config.cancellation.token
        wf = Workflow(flow=flows.Flow(nodes=make_slow_pipeline()))

        def go():
            return wf.run_sync(input_data={"query": "test"}, config=config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        assert holder["result"].status == RunnableStatus.CANCELED

    def test_each_runnable_config_has_unique_token(self):
        c1 = RunnableConfig()
        c2 = RunnableConfig()
        assert c1.cancellation.token is not c2.cancellation.token

        c1.cancellation.token.cancel()
        assert c1.cancellation.is_canceled is True
        assert c2.cancellation.is_canceled is False


class TestMultiRunIsolation:
    def test_canceling_one_run_does_not_affect_another(self, mocker):
        """Two parallel runs with separate tokens - cancel one, verify other completes."""
        mock_llm_success(mocker)
        flow = flows.Flow(nodes=make_pipeline())

        token1 = CancellationToken()
        token2 = CancellationToken()
        config1 = RunnableConfig(cancellation=CancellationConfig(token=token1))
        config2 = RunnableConfig(cancellation=CancellationConfig(token=token2))

        # Cancel run 1 immediately, run 2 should succeed
        token1.cancel()

        result1 = flow.run_sync(input_data={"query": "run1"}, config=config1)
        result2 = flow.run_sync(input_data={"query": "run2"}, config=config2)

        assert result1.status == RunnableStatus.CANCELED
        assert result2.status == RunnableStatus.SUCCESS
        assert token2.is_canceled is False

    def test_token_cancel_after_completion_is_noop(self, mocker, cancellation_token, runnable_config):
        """If we cancel after the workflow finishes successfully, the result is unchanged."""
        mock_llm_success(mocker)

        flow = flows.Flow(nodes=make_pipeline())
        result = flow.run_sync(input_data={"query": "test"}, config=runnable_config)
        assert result.status == RunnableStatus.SUCCESS

        cancellation_token.cancel()
        assert result.status == RunnableStatus.SUCCESS


class TestPerRunOverrides:
    def test_cancel_via_per_run_config(self, mocker):
        """Even if no cancellation is set on the workflow, the runtime config controls it."""
        mock_llm_success(mocker)

        token = CancellationToken()
        token.cancel()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))

        flow = flows.Flow(nodes=make_pipeline())
        result = flow.run_sync(input_data={"query": "test"}, config=config)
        assert result.status == RunnableStatus.CANCELED
