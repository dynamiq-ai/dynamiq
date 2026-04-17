import threading
import time

from litellm import ModelResponse

from dynamiq import Workflow, connections, flows
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.nodes.operators.operators import Map
from dynamiq.nodes.tools import Python
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.cancellation import check_cancellation

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"

CANCEL_DELAY = 0.4
THREAD_JOIN_TIMEOUT = 15.0


def make_llm(node_id: str = "llm") -> llms.OpenAI:
    return llms.OpenAI(
        id=node_id,
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        is_postponed_component_init=True,
    )


SLOW_TOOL_CODE = """
import time
def run(input_data):
    elapsed = 0.0
    while elapsed < 1.5:
        time.sleep(0.05)
        elapsed += 0.05
    return {"content": "ok"}
"""


def make_slow_tool(name: str = "slow-tool", node_id: str | None = None) -> Python:
    return Python(
        id=node_id or name,
        name=name,
        description=f"{name}. Input: {{'step': <n>}}.",
        code=SLOW_TOOL_CODE,
    )


class CooperativeSlowNode(Node):
    """Sleeps in increments while checking cancellation between each.
    Allowed to run in parallel inside Map / Agent parallel-tool execution."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "cooperative-slow"
    sleep_seconds: float = 5.0
    is_parallel_execution_allowed: bool = True

    def execute(self, input_data, config=None, **kwargs):
        elapsed = 0.0
        while elapsed < self.sleep_seconds:
            check_cancellation(config)
            time.sleep(0.05)
            elapsed += 0.05
        return {"result": "done"}


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


def mock_react_loop(mocker, *, tool_name: str = "calculator", tool_calls: int = 5, final: str = "Done"):
    call_count = {"value": 0}

    def side_effect(stream: bool, *args, **kwargs):
        call_count["value"] += 1
        r = ModelResponse()
        if call_count["value"] <= tool_calls:
            r["choices"][0]["message"]["content"] = (
                f"Thought: Step {call_count['value']}.\n"
                f"Action: {tool_name}\n"
                f"Action Input: {{\"step\": {call_count['value']}}}"
            )
        else:
            r["choices"][0]["message"]["content"] = f"Thought: Done.\nAnswer: {final}"
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count


def mock_parallel_xml_react(mocker, *, parallel_rounds: int = 3, final: str = "Done"):
    """Mock LLM that returns XML-formatted parallel tool calls."""
    call_count = {"value": 0}

    def side_effect(stream: bool, *args, **kwargs):
        call_count["value"] += 1
        r = ModelResponse()
        if call_count["value"] <= parallel_rounds:
            r["choices"][0]["message"]["content"] = (
                "<output>\n"
                f"    <thought>Round {call_count['value']}.</thought>\n"
                "    <action>run-parallel</action>\n"
                '    <action_input>{"tools": ['
                '{"name": "tool-a", "input": {"step": 1}}, '
                '{"name": "tool-b", "input": {"step": 2}}'
                "]}</action_input>\n"
                "</output>"
            )
        else:
            r["choices"][0]["message"][
                "content"
            ] = f"<output>\n    <thought>Done.</thought>\n    <answer>{final}</answer>\n</output>"
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count


class TestMapNodeCancellation:
    """Map node uses ContextAwareThreadPoolExecutor.map for parallel iteration."""

    def test_map_pre_canceled_returns_canceled(self, cancellation_token, runnable_config):
        cancellation_token.cancel()
        inner = CooperativeSlowNode(sleep_seconds=2.0)
        map_node = Map(node=inner, max_workers=4)

        flow = flows.Flow(nodes=[map_node])
        result = flow.run_sync(
            input_data={"input": [{"i": i} for i in range(8)]},
            config=runnable_config,
        )
        assert result.status == RunnableStatus.CANCELED

    def test_map_canceled_mid_parallel_iteration(self, cancellation_token, runnable_config):
        """Cancel while Map is iterating parallel workers — they should stop promptly."""
        inner = CooperativeSlowNode(sleep_seconds=5.0)
        map_node = Map(node=inner, max_workers=3)

        flow = flows.Flow(nodes=[map_node])

        def go():
            return flow.run_sync(
                input_data={"input": [{"i": i} for i in range(6)]},
                config=runnable_config,
            )

        start = time.time()
        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)
        elapsed = time.time() - start

        assert holder["result"].status == RunnableStatus.CANCELED
        # Must finish much earlier than 5s * (6/3) = 10s without cancel
        assert elapsed < 6.0

    def test_map_with_default_config_supports_cancellation(self):
        """Map with default RunnableConfig() inherits the auto-wired cancellation token."""
        config = RunnableConfig()
        token = config.cancellation.token

        inner = CooperativeSlowNode(sleep_seconds=5.0)
        map_node = Map(node=inner, max_workers=2)
        flow = flows.Flow(nodes=[map_node])

        def go():
            return flow.run_sync(
                input_data={"input": [{"i": i} for i in range(4)]},
                config=config,
            )

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        assert holder["result"].status == RunnableStatus.CANCELED


class TestParallelAgentsInFlow:
    """Two independent agents (no NodeDependency between them) execute in parallel
    via the flow's executor. Cancel signal stops both."""

    def test_two_parallel_agents_canceled_together(self, mocker, cancellation_token, runnable_config):
        mock_react_loop(mocker, tool_calls=10)

        agent_a = Agent(
            id="agent-a",
            name="Agent A",
            llm=make_llm("llm-a"),
            tools=[make_slow_tool("calculator", "tool-a")],
            role="Agent A",
            max_loops=10,
        )
        agent_b = Agent(
            id="agent-b",
            name="Agent B",
            llm=make_llm("llm-b"),
            tools=[make_slow_tool("calculator", "tool-b")],
            role="Agent B",
            max_loops=10,
        )
        flow = flows.Flow(nodes=[agent_a, agent_b])

        def go():
            return flow.run_sync(input_data={"input": "test"}, config=runnable_config)

        start = time.time()
        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)
        elapsed = time.time() - start

        assert holder["result"].status == RunnableStatus.CANCELED
        # Should finish quickly after cancel — well before max iteration time
        assert elapsed < 10.0

    def test_three_parallel_agents_workflow_canceled(self, mocker, cancellation_token, runnable_config):
        mock_react_loop(mocker, tool_calls=20)

        agents = [
            Agent(
                id=f"agent-{i}",
                name=f"Agent {i}",
                llm=make_llm(f"llm-{i}"),
                tools=[make_slow_tool("calculator", f"tool-{i}")],
                role=f"Agent {i}",
                max_loops=20,
            )
            for i in range(3)
        ]
        wf = Workflow(flow=flows.Flow(nodes=agents))

        def go():
            return wf.run_sync(input_data={"input": "test"}, config=runnable_config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        assert holder["result"].status == RunnableStatus.CANCELED


class TestAgentParallelToolCalls:
    """Agent with parallel_tool_calls_enabled=True executes tools concurrently
    via ContextAwareThreadPoolExecutor inside _execute_tools."""

    def test_parallel_tool_calls_canceled(self, mocker, cancellation_token, runnable_config):
        mock_parallel_xml_react(mocker, parallel_rounds=3)

        tool_a = make_slow_tool("tool-a", "tool-a")
        tool_a.is_parallel_execution_allowed = True
        tool_b = make_slow_tool("tool-b", "tool-b")
        tool_b.is_parallel_execution_allowed = True

        agent = Agent(
            id="agent",
            name="Parallel Agent",
            llm=make_llm(),
            tools=[tool_a, tool_b],
            role="Parallel agent",
            max_loops=5,
            parallel_tool_calls_enabled=True,
            inference_mode=InferenceMode.XML,
        )
        flow = flows.Flow(nodes=[agent])

        def go():
            return flow.run_sync(input_data={"input": "test"}, config=runnable_config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        assert holder["result"].status == RunnableStatus.CANCELED


class TestSubAgentCancellation:
    """Parent agent delegates to a child agent via SubAgentTool. Cancel must
    propagate from parent through SubAgentTool through child agent.

    Both parent and child share the same _completion mock. We drive each LLM
    call to the matching tool name so each iteration actually executes the slow
    tool (giving time for cancel to fire)."""

    def test_subagent_canceled_propagates_to_parent(self, mocker, cancellation_token, runnable_config):
        # Mock distinguishes parent vs child by inspecting the messages payload.
        # Parent gets "Action: calc-sub-agent" → invokes sub-tool.
        # Child gets "Action: slow-calc" → invokes its own slow tool.
        call_count = {"value": 0}

        def side_effect(stream: bool, *args, **kwargs):
            call_count["value"] += 1
            messages = kwargs.get("messages") or []
            sys_content = ""
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "system":
                    sys_content = m.get("content", "")
                    break
            r = ModelResponse()
            # Parent's system prompt mentions delegation; route to sub-agent
            if "Parent" in sys_content or "delegate" in sys_content.lower():
                r["choices"][0]["message"]["content"] = (
                    "Thought: Delegate.\nAction: calc-sub-agent\n" 'Action Input: {"input": "calculate"}'
                )
            else:
                # Child: invoke its own slow tool
                r["choices"][0]["message"]["content"] = (
                    "Thought: Calculate.\nAction: slow-calc\n" 'Action Input: {"step": 1}'
                )
            return r

        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)

        # Child agent with a slow tool that will block, giving cancel a window
        child = Agent(
            id="child-agent",
            name="Child Agent",
            llm=make_llm("child-llm"),
            tools=[make_slow_tool("slow-calc", "child-tool")],
            role="Child agent that calculates.",
            max_loops=10,
        )
        sub_tool = SubAgentTool(
            id="sub-tool",
            name="calc-sub-agent",
            description="Delegates to child agent.",
            agent=child,
        )
        parent = Agent(
            id="parent-agent",
            name="Parent Agent",
            llm=make_llm("parent-llm"),
            tools=[sub_tool],
            role="Parent agent that must delegate everything.",
            max_loops=10,
        )
        flow = flows.Flow(nodes=[parent])

        def go():
            return flow.run_sync(input_data={"input": "delegate"}, config=runnable_config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        result = holder["result"]
        assert result.status == RunnableStatus.CANCELED
        assert call_count["value"] >= 1
