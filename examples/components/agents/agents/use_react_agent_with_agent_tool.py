import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

CHILD_ROLE = """
You are a focused calculation assistant.
You have a tool named "Sum Calculator" that computes the sum of the first n integers.
STRICT: When using this tool, DO NOT pass arbitrary code. Provide only structured input like {"n": 7}.
Always return a concise explanation along with results.
"""


PARENT_ROLE = """
You are a senior assistant who can delegate work to specialized agent tools.
If a tool is itself an agent, pass it a concise subtask via {"input": "..."}.
Summarize final results for the user.
"""


def make_child_agent(llm):
    python_tool = Python(
        code="""
def run(params: dict):
    # expects optional n; defaults to 10
    n = int(params.get("n", 10))
    s = sum(range(1, n + 1))
    return {"content": f"sum(1..{n})={s}", "n": n, "sum": s}
""",
        name="Sum Calculator",
        description='Computes the sum of the first n integers; expects input {"n": int}',
    )

    child = ReActAgent(
        name="Coder Agent",
        description='Uses Sum Calculator tool; provide only {"n": int}',
        role=CHILD_ROLE,
        llm=llm,
        tools=[python_tool],
        max_loops=3,
    )
    return child


def make_parent_agent(llm, child_agent):
    parent = ReActAgent(
        name="Manager Agent",
        description="Delegates subtasks to a coding agent when computation is required.",
        role=PARENT_ROLE,
        llm=llm,
        tools=[child_agent],
        max_loops=3,
    )
    return parent


def run_workflow():
    """
    Build a simple workflow: parent agent that uses a child agent as a tool.
    Returns (content, traces) for graph drawing utilities.
    """
    llm = setup_llm()
    child = make_child_agent(llm)
    parent = make_parent_agent(llm, child)

    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[parent]))

    result = wf.run(
        input_data={
            "input": "Compute the sum of the first 7 integers. Use your coding agent if needed.",
            "tool_params": {"by_name": {"Coder Agent": {"global": {"n": 7}}}},
        },
        config=RunnableConfig(callbacks=[tracing]),
    )

    json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)

    content = result.output[parent.id]["output"]["content"]
    return content, tracing.runs


if __name__ == "__main__":
    output, _ = run_workflow()
    logger.info("=== AGENT OUTPUT ===")
    logger.info(output)
