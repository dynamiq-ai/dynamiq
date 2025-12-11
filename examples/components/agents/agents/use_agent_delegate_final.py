import json
import os

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools import TavilyTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

RESEARCHER_ROLE = """
You are a concise researcher.
Produce a short markdown brief:
- Start with a level-2 heading naming the topic.
- Follow with 3-5 bullet points of crisp facts.
- End with a one-sentence takeaway in bold.
Keep it tight—no filler, no extra sections.
"""


MANAGER_ROLE = """
You are a manager agent that hands research tasks to the Researcher Agent.
When you call that agent tool, include "delegate_final": true so its response is returned directly.
Do not summarize or rewrite its output yourself.
"""


def make_researcher_agent(llm):
    secondary_llm = setup_llm()
    tavily_connection = TavilyConnection()
    search_tool = TavilyTool(
        name="Tavily Search",
        description="Finds recent information on the web (powered by Tavily).",
        connection=tavily_connection,
    )

    return Agent(
        name="Researcher Agent",
        description='Call with {"input": "<topic to research>"}',
        role=RESEARCHER_ROLE,
        llm=secondary_llm,
        tools=[search_tool],
        max_loops=2,
        inference_mode=InferenceMode.XML,
    )


def make_manager_agent(llm, researcher):
    return Agent(
        name="Manager Agent",
        description="Delegates research questions to a markdown-focused researcher.",
        role=MANAGER_ROLE,
        llm=llm,
        tools=[researcher],
        max_loops=3,
        inference_mode=InferenceMode.XML,
    )


def run_workflow(callbacks: list | None = None):
    """
    Build a workflow where the manager delegates and returns the researcher's markdown verbatim.
    Returns (content, traces) for graph drawing utilities.
    """
    llm = setup_llm()
    researcher = make_researcher_agent(llm)
    manager = make_manager_agent(llm, researcher)

    default_tracing = None
    if callbacks is None:
        default_tracing = TracingCallbackHandler()
        callbacks = [default_tracing]
    wf = Workflow(flow=Flow(nodes=[manager]))

    result = wf.run(
        input_data={
            "input": "Give me a brief on the future of solid-state batteries.",
        },
        config=RunnableConfig(callbacks=callbacks),
    )

    trace_runs = {}
    for cb in callbacks:
        runs = getattr(cb, "runs", None)
        if runs is not None:
            trace_runs = runs
            break
    if trace_runs:
        json.dumps({"runs": [run.to_dict() for run in trace_runs.values()]}, cls=JsonWorkflowEncoder)

    content = result.output[manager.id]["output"]["content"]
    return content, trace_runs


if __name__ == "__main__":
    trace_base_url = os.environ.get("DYNAMIQ_TRACE_BASE_URL")
    trace_access_key = os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY")
    enable_ui_tracing = bool(trace_access_key)

    if enable_ui_tracing:
        tracing = DynamiqTracingCallbackHandler(base_url=trace_base_url, access_key=trace_access_key)
        output, traces = run_workflow(callbacks=[tracing])
    else:
        output, traces = run_workflow()

    logger.info("=== AGENT OUTPUT ===")
    logger.info(output)
