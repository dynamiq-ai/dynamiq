"""A human-input timeout inside an agent run: a pause, never a failure worth retrying."""

from queue import Queue

from dynamiq.nodes.agents.hooks import ToolPolicyHook
from dynamiq.nodes.node import ErrorHandling
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.types.streaming import InputStreamingTimeoutError, StreamingConfig, StreamingMode
from tests.helpers.agent_hooks import TOOL_CALLS, SearchTool, build_agent, final_step, scripted_stateless, tool_step


def test_an_approval_timeout_is_not_retried_by_rerunning_the_agent():
    # A retry would run every earlier tool again and ask the human a second time.
    agent = build_agent(
        [ToolPolicyHook(tools=["send-email"], approval=True, feedback_method=FeedbackMethod.STREAM)],
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.FINAL, input_queue=Queue(), timeout=0.2),
        error_handling=ErrorHandling(timeout_seconds=60, max_retries=2, retry_interval_seconds=0),
    )
    steps = [tool_step("search", query="x"), tool_step("send-email", to="a@b.c", body="hi"), final_step("ok")]
    TOOL_CALLS.clear()

    with scripted_stateless(InferenceMode.FUNCTION_CALLING, steps, [agent]):
        result = agent.run(input_data={"input": "hello"})

    assert result.status == RunnableStatus.FAILURE
    assert issubclass(result.error.type, InputStreamingTimeoutError)
    assert [name for name, _ in TOOL_CALLS] == ["search"]


class AskHuman(SearchTool):
    """Waits for an answer that never comes, as HumanFeedbackTool does when its input times out."""

    name: str = "ask-human"

    def execute(self, input_data, config=None, **kwargs):
        TOOL_CALLS.append((self.name, input_data.model_dump()))
        raise InputStreamingTimeoutError(node_id=self.id, timeout=0.2)


def test_a_tool_input_timeout_keeps_its_type_and_is_not_retried():
    agent = build_agent(
        tools=[SearchTool(), AskHuman()],
        error_handling=ErrorHandling(timeout_seconds=60, max_retries=2, retry_interval_seconds=0),
    )
    steps = [tool_step("search", query="x"), tool_step("ask-human", query="which one?"), final_step("ok")]
    TOOL_CALLS.clear()

    with scripted_stateless(InferenceMode.FUNCTION_CALLING, steps, [agent]):
        result = agent.run(input_data={"input": "hello"})

    assert result.status == RunnableStatus.FAILURE
    assert issubclass(result.error.type, InputStreamingTimeoutError)
    assert result.error.message == "Input streaming timeout: 0.2 seconds exceeded."
    assert [name for name, _ in TOOL_CALLS] == ["search", "ask-human"]


def test_a_standalone_input_wait_that_times_out_is_still_retried():
    # Outside an agent a retry repeats nothing but the question, and it is how such a node keeps waiting.
    tool = AskHuman(error_handling=ErrorHandling(timeout_seconds=60, max_retries=2, retry_interval_seconds=0))
    TOOL_CALLS.clear()

    result = tool.run(input_data={"query": "which one?"})

    assert result.status == RunnableStatus.FAILURE
    assert len(TOOL_CALLS) == 3
