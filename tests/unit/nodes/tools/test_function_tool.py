from dynamiq.nodes.tools.function_tool import function_tool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus
from tests.helpers.agent_hooks import build_agent, final_step, run_scripted, tool_step


@function_tool
def order_status(order_id: str) -> str:
    """Return the status of an order."""
    return f"order {order_id} shipped"


def test_a_plain_function_runs_as_an_agent_tool():
    # The agent runs tools with the run's config and ids, which a function without **kwargs does not take.
    agent = build_agent(tools=[order_status()])
    steps = [tool_step("order_status", order_id="A-7"), final_step("done")]

    result, recorder = run_scripted(agent, InferenceMode.FUNCTION_CALLING, steps)

    assert result.status == RunnableStatus.SUCCESS
    assert "order A-7 shipped" in recorder.sent_text(1)


def test_a_function_that_takes_keywords_still_gets_the_run_config():
    received = {}

    @function_tool
    def remember(note: str, **kwargs) -> str:
        """Store a note."""
        received.update(kwargs)
        return "stored"

    result = remember().run({"note": "x"})

    assert result.status == RunnableStatus.SUCCESS
    assert "config" in received


def test_an_argument_named_like_a_run_keyword_is_the_models_to_give():
    # The node passes its own run_id with every call.
    @function_tool
    def run_status(run_id: str = "latest") -> str:
        """Return the status of a CI run."""
        return f"run {run_id} passed"

    named = run_status().run({"run_id": "ci-123"})
    default = run_status().run({})

    assert named.output["content"] == "run ci-123 passed"
    assert default.output["content"] == "run latest passed"
