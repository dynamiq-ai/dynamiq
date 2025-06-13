import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B, Exa
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = (
    "You are a versatile AI assistant:\n"
    "  1. Plan your approach in a clear reasoning step.\n"
    "  2. Search  for any documentation or data you need.\n"
    "  3. Code: write a snippet that solves the user's request, use code interpreter\n"
    "  4. Execute & Validate:\n"
    "     - Run your code in the sandbox.\n"
    "     - Inspect stdout/stderr and the data structures returned.\n"
    "     - If there are errors or unexpected results, fix them.\n"
    "     - Optionally add simple assertions or checks to prove correctness.\n"
    "     - Repeat until the code runs cleanly and meets the spec.\n"
    "  5. Report: Return a concise Markdown‐formatted answer or code block,\n"
    "     including any tables or plots if relevant.\n\n"
    "Always think in terms of tool use: plan → search → code → test → refine → answer."
)

EXAMPLE_QUERY = (
    "Find the best conferences in europe for AI in 2025"
    "You have to use parallel tool calls, to perform search from different directions."
)


def setup_multi_tool_agent() -> ReActAgent:
    """
    Set up and return a ReAct agent with search and code interpreter tools.

    Returns:
        ReActAgent: Configured ReAct agent with multi-tool capabilities.
    """
    # Initialize connections
    conn_code = E2B()
    conn_search = Exa()

    # Create tools
    tool_code = E2BInterpreterTool(connection=conn_code)
    tool_search = ExaTool(connection=conn_search)

    # Setup LLM
    llm = setup_llm(model_provider="gpt", model_name="o3-mini", temperature=0.2)

    # Create agent
    agent = ReActAgent(
        name="MultiToolAgent",
        id="MultiToolAgent",
        llm=llm,
        tools=[tool_search, tool_code],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=True,
    )

    return agent


def run_workflow(agent: ReActAgent = setup_multi_tool_agent(), input_prompt: str = EXAMPLE_QUERY) -> tuple[str, dict]:
    """
    Execute a workflow using the multi-tool ReAct agent to process a query.

    Args:
        agent: The ReAct agent to use (defaults to a new multi-tool agent)
        input_prompt: The input prompt to process (defaults to example query)

    Returns:
        tuple[str, dict]: The generated content by the agent and the trace logs.

    Raises:
        Exception: Captures and prints any errors during workflow execution.
    """
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": input_prompt},
            config=RunnableConfig(callbacks=[tracing]),
        )
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    output, traces = run_workflow()
    logger.info("=== AGENT OUTPUT ===")
    logger.info(output)
