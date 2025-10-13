import json
import os

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

CHILD_ROLE = """
You are a collaborative math specialist with access to two tools:
1. "Range Sum" — expects input {"n": int} and returns sum(1..n)
2. "Range Product" — expects input {"m": int} and returns product(1..m)

When possible, execute both tools in parallel and combine their outputs into a clear summary for the caller.
"""


PARENT_ROLE = """
You are a project manager who delegates computation-heavy work to the Math Specialist agent tool.
For multi-step numeric requests, pass a concise subtask via {"input": "..."} and ask the tool to solve each
part using parallel tool calls when appropriate.
Summarize the specialist's findings for the user.
"""


def make_child_agent(llm):
    sum_tool = Python(
        name="Range Sum",
        description='Returns sum(1..n). Expects input {"n": int}.',
        code="""
def run(params: dict):
    n = int(params.get("n", 1))
    total = sum(range(1, n + 1))
    return {"content": f"sum(1..{n}) = {total}", "n": n, "sum": total}
""",
    )

    product_tool = Python(
        name="Range Product",
        description='Returns product(1..m). Expects input {"m": int}.',
        code="""
import math

def run(params: dict):
    m = int(params.get("m", 1))
    product = math.prod(range(1, m + 1)) if m > 0 else 1
    return {"content": f"product(1..{m}) = {product}", "m": m, "product": product}
""",
    )

    return Agent(
        name="Math Specialist",
        description='Call with {"input": "Compute sums/products ..."}',
        role=CHILD_ROLE,
        llm=llm,
        tools=[sum_tool, product_tool],
        max_loops=3,
        parallel_tool_calls_enabled=True,
    )


def make_parent_agent(llm, child_agent):
    return Agent(
        name="Delegator Agent",
        description="Delegates range calculations to the Math Specialist agent tool.",
        role=PARENT_ROLE,
        llm=llm,
        tools=[child_agent],
        max_loops=3,
        parallel_tool_calls_enabled=True,
    )


def _resolve_trace_runs(callbacks: list) -> dict:
    for cb in callbacks:
        runs = getattr(cb, "runs", None)
        if runs is not None:
            return runs
    return {}


def run_workflow(callbacks: list | None = None):
    """Run the workflow and return (content, trace_runs)."""
    llm = setup_llm()
    child = make_child_agent(llm)
    parent = make_parent_agent(llm, child)
    default_tracing = None
    if callbacks is None:
        default_tracing = TracingCallbackHandler()
        callbacks = [default_tracing]

    wf = Workflow(flow=Flow(nodes=[parent]))

    result = wf.run(
        input_data={
            "input": (
                "Coordinate with the math specialist. Ask it to compute sum(1..6) and product(1..4) in parallel."
                " Provide the results with a concise summary."
            ),
            "tool_params": {
                "by_name": {"Math Specialist": {"by_name": {"Range Sum": {"n": 6}, "Range Product": {"m": 4}}}}
            },
        },
        config=RunnableConfig(callbacks=callbacks),
    )

    trace_runs = _resolve_trace_runs(callbacks)
    if trace_runs:
        json.dumps({"runs": [run.to_dict() for run in trace_runs.values()]}, cls=JsonWorkflowEncoder)

    content = result.output[parent.id]["output"]["content"]
    return content, trace_runs


def run_workflow_with_ui_tracing(
    base_url: str = os.environ.get("DYNAMIQ_TRACE_BASE_URL", "https://collector.sandbox.getdynamiq.ai"),
    access_key: str | None = os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY"),
    **handler_kwargs,
):
    """Execute the workflow with Dynamiq UI tracing enabled."""

    tracing = DynamiqTracingCallbackHandler(base_url=base_url, access_key=access_key, **handler_kwargs)
    content, traces = run_workflow(callbacks=[tracing])
    return content, traces, tracing


if __name__ == "__main__":
    output, _, _ = run_workflow_with_ui_tracing()
    logger.info("=== PARALLEL AGENT TOOL OUTPUT ===")
    logger.info(output)
