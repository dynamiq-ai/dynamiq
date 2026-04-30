import asyncio
import time

import pytest

from dynamiq import Workflow, connections, flows
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.node import Node, NodeDependency, NodeGroup
from dynamiq.nodes.tools import Python
from dynamiq.nodes.utils import Input, Output
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.cancellation import CanceledException, CancellationConfig, CancellationToken, check_cancellation

from .conftest import mock_llm_react_loop, mock_llm_success

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"


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
    """Input -> CooperativeSlowNode -> Output."""
    inp = Input(id="input", name="Input")
    slow = CooperativeSlowNode(
        id="slow",
        name="Slow",
        sleep_seconds=sleep_seconds,
        depends=[NodeDependency(inp)],
    )
    out = Output(id="output", name="Output", depends=[NodeDependency(slow)])
    return [inp, slow, out]


class TestAsyncTokenCancellation:
    @pytest.mark.asyncio
    async def test_pre_canceled_async_pipeline(self, mocker, cancellation_token, runnable_config):
        llm_call = mock_llm_success(mocker)
        cancellation_token.cancel()

        flow = flows.Flow(nodes=make_pipeline())
        result = await flow.run_async(input_data={"query": "test"}, config=runnable_config)

        assert result.status == RunnableStatus.CANCELED
        assert result.input is None
        assert result.output is None
        assert result.error is not None
        assert result.error.type is CanceledException
        assert result.error.message  # has a descriptive message
        # LLM never called - flow stopped at first cancellation check
        assert llm_call.call_count == 0

    @pytest.mark.asyncio
    async def test_async_token_cancel_during_slow_node(self, cancellation_token, runnable_config):
        flow = flows.Flow(nodes=make_slow_pipeline(sleep_seconds=5.0))

        task = asyncio.create_task(flow.run_async(input_data={"query": "test"}, config=runnable_config))
        await asyncio.sleep(0.3)
        cancellation_token.cancel()
        result = await task

        assert result.status == RunnableStatus.CANCELED
        assert result.input is None
        assert result.output is None
        assert result.error is not None
        assert result.error.type is CanceledException
        assert result.error.message  # has a descriptive message

    @pytest.mark.asyncio
    async def test_async_workflow_token_cancel(self, cancellation_token, runnable_config):
        wf = Workflow(flow=flows.Flow(nodes=make_slow_pipeline(sleep_seconds=5.0)))

        task = asyncio.create_task(wf.run_async(input_data={"query": "test"}, config=runnable_config))
        await asyncio.sleep(0.3)
        cancellation_token.cancel()
        result = await task

        assert result.status == RunnableStatus.CANCELED


class TestAsyncTaskCancel:
    """task.cancel() raises CancelledError inside the coroutine. The framework
    must catch it at all 3 layers (Node/Flow/Workflow) and convert to CANCELED."""

    @pytest.mark.asyncio
    async def test_task_cancel_on_workflow_returns_canceled(self):
        wf = Workflow(flow=flows.Flow(nodes=make_slow_pipeline(sleep_seconds=5.0)))
        config = RunnableConfig()  # default cancellation auto-wired

        task = asyncio.create_task(wf.run_async(input_data={"query": "test"}, config=config))
        await asyncio.sleep(0.3)
        task.cancel()

        try:
            result = await task
        except asyncio.CancelledError:
            pytest.fail("asyncio.CancelledError leaked out — framework should catch it")

        assert result.status == RunnableStatus.CANCELED
        assert result.input is None
        assert result.output is None
        assert result.error is not None
        assert result.error.type is CanceledException
        assert result.error.message  # has a descriptive message

    @pytest.mark.asyncio
    async def test_task_cancel_on_flow_returns_canceled(self):
        flow = flows.Flow(nodes=make_slow_pipeline(sleep_seconds=5.0))
        config = RunnableConfig()

        task = asyncio.create_task(flow.run_async(input_data={"query": "test"}, config=config))
        await asyncio.sleep(0.3)
        task.cancel()

        try:
            result = await task
        except asyncio.CancelledError:
            pytest.fail("asyncio.CancelledError leaked out from flow.run_async")

        assert result.status == RunnableStatus.CANCELED

    @pytest.mark.asyncio
    async def test_task_cancel_signals_shared_token(self):
        """When task.cancel() fires, the framework should set the shared token
        so any background threading.Thread (asyncio.to_thread) stops."""
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
        wf = Workflow(flow=flows.Flow(nodes=make_slow_pipeline(sleep_seconds=5.0)))

        task = asyncio.create_task(wf.run_async(input_data={"query": "test"}, config=config))
        await asyncio.sleep(0.3)
        assert not token.is_canceled

        task.cancel()
        await task

        # After task.cancel(), the token must be set so background threads stop
        assert token.is_canceled

    @pytest.mark.asyncio
    async def test_default_config_supports_task_cancel(self):
        """No explicit CancellationConfig — RunnableConfig default auto-provides it."""
        wf = Workflow(flow=flows.Flow(nodes=make_slow_pipeline(sleep_seconds=5.0)))

        task = asyncio.create_task(wf.run_async(input_data={"query": "test"}))
        await asyncio.sleep(0.3)
        task.cancel()
        result = await task

        assert result.status == RunnableStatus.CANCELED


class TestConcurrentAsyncRuns:
    @pytest.mark.asyncio
    async def test_canceling_one_run_does_not_affect_another(self, mocker):
        """Two parallel async runs with separate tokens. Cancel one, verify other completes."""
        mock_llm_success(mocker)

        token1 = CancellationToken()
        token2 = CancellationToken()
        config1 = RunnableConfig(cancellation=CancellationConfig(token=token1))
        config2 = RunnableConfig(cancellation=CancellationConfig(token=token2))

        flow1 = flows.Flow(nodes=make_slow_pipeline(sleep_seconds=3.0))
        flow2 = flows.Flow(nodes=make_pipeline())

        task1 = asyncio.create_task(flow1.run_async(input_data={"query": "run1"}, config=config1))
        task2 = asyncio.create_task(flow2.run_async(input_data={"query": "run2"}, config=config2))

        await asyncio.sleep(0.3)
        token1.cancel()  # cancel only the slow one

        result1, result2 = await asyncio.gather(task1, task2)

        assert result1.status == RunnableStatus.CANCELED
        assert result2.status == RunnableStatus.SUCCESS
        assert token2.is_canceled is False

    @pytest.mark.asyncio
    async def test_multiple_concurrent_canceled_runs(self):
        """Five concurrent async slow workflows, all canceled. Each gets its own Flow
        instance to avoid topological-sort state corruption from shared mutation."""
        flows_list = [flows.Flow(nodes=make_slow_pipeline(sleep_seconds=5.0)) for _ in range(5)]
        tokens = [CancellationToken() for _ in range(5)]
        configs = [RunnableConfig(cancellation=CancellationConfig(token=t)) for t in tokens]
        tasks = [
            asyncio.create_task(f.run_async(input_data={"query": f"run{i}"}, config=cfg))
            for i, (f, cfg) in enumerate(zip(flows_list, configs))
        ]

        await asyncio.sleep(0.3)
        for t in tokens:
            t.cancel()

        results = await asyncio.gather(*tasks)
        assert all(r.status == RunnableStatus.CANCELED for r in results)


class TestAsyncCancellationEdgeCases:
    @pytest.mark.asyncio
    async def test_cancel_after_completion_is_noop(self, mocker, cancellation_token, runnable_config):
        mock_llm_success(mocker)

        flow = flows.Flow(nodes=make_pipeline())
        result = await flow.run_async(input_data={"query": "test"}, config=runnable_config)
        assert result.status == RunnableStatus.SUCCESS

        cancellation_token.cancel()  # too late
        assert result.status == RunnableStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_immediate_task_cancel_before_any_work(self):
        """task.cancel() called immediately after create_task returns CANCELED."""
        wf = Workflow(flow=flows.Flow(nodes=make_slow_pipeline(sleep_seconds=5.0)))
        config = RunnableConfig()

        task = asyncio.create_task(wf.run_async(input_data={"query": "test"}, config=config))
        # Yield once to let the task start
        await asyncio.sleep(0)
        task.cancel()

        try:
            result = await task
        except asyncio.CancelledError:
            pytest.fail("asyncio.CancelledError leaked - should be caught")

        assert result.status == RunnableStatus.CANCELED


class TestAsyncAgentCancellation:
    @pytest.mark.asyncio
    async def test_async_agent_canceled_via_task_cancel(self, mocker):
        """Agent running in async mode + task.cancel() returns CANCELED."""
        mock_llm_react_loop(mocker, tool_calls=20)

        slow_tool = Python(
            id="slow-tool",
            name="calculator",
            description="Calculator. Input: {'step': <n>}.",
            code=(
                "import time\n"
                "def run(input_data):\n"
                "    elapsed = 0.0\n"
                "    while elapsed < 1.5:\n"
                "        time.sleep(0.05)\n"
                "        elapsed += 0.05\n"
                "    return {'content': 'ok'}\n"
            ),
        )
        agent = Agent(
            id="agent",
            name="ReAct Agent",
            llm=llms.OpenAI(
                id="agent-llm",
                model=LLM_MODEL,
                connection=connections.OpenAI(api_key=TEST_API_KEY),
                is_postponed_component_init=True,
            ),
            tools=[slow_tool],
            role="Test agent",
            max_loops=20,
        )
        flow = flows.Flow(nodes=[agent])
        config = RunnableConfig()

        task = asyncio.create_task(flow.run_async(input_data={"input": "test"}, config=config))
        await asyncio.sleep(0.3)
        task.cancel()

        try:
            result = await task
        except asyncio.CancelledError:
            pytest.fail("CancelledError leaked from agent")

        assert result.status == RunnableStatus.CANCELED
