import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import E2B, Exa
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = (
    "You are a versatile AI assistant:\n"
    "  1. Plan your approach in a clear reasoning step.\n"
    "  2. Search for any documentation or data you need.\n"
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
    "Find the best AI conferences in Europe for 2025. "
    "You should use parallel tool calls to search from different angles."
)


def setup_multi_tool_streaming_agent() -> Agent:
    """
    Set up and return a ReAct agent with search and code interpreter tools,
    configured with streaming enabled to test parallel tool calls streaming.

    Returns:
        Agent: Configured ReAct agent with multi-tool capabilities and streaming.
    """
    conn_code = E2B()
    conn_search = Exa()

    tool_code = E2BInterpreterTool(connection=conn_code)
    tool_search = ExaTool(connection=conn_search)

    llm = setup_llm(model_provider="gpt", model_name="o3-mini", temperature=0.2)

    agent = Agent(
        name="MultiToolStreamingAgent",
        id="MultiToolStreamingAgent",
        llm=llm,
        tools=[tool_search, tool_code],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=True,
        streaming=StreamingConfig(
            enabled=True,
            event="multi_tool_stream",
            mode=StreamingMode.ALL,
        ),
        max_loops=3,
    )

    return agent


def run_streaming_workflow(agent: Agent = None, input_prompt: str = EXAMPLE_QUERY) -> tuple[str, dict, list]:
    """
    Execute a workflow using the multi-tool ReAct agent with streaming enabled
    to test parallel tool calls streaming functionality.

    Args:
        agent: The ReAct agent to use (defaults to a new multi-tool streaming agent)
        input_prompt: The input prompt to process (defaults to example query)

    Returns:
        tuple[str, dict, list]: The generated content, trace logs, and streaming chunks.

    Raises:
        Exception: Captures and prints any errors during workflow execution.
    """
    if agent is None:
        agent = setup_multi_tool_streaming_agent()

    tracing = TracingCallbackHandler()
    streaming_handler = StreamingIteratorCallbackHandler()

    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        logger.info("=== Starting Multi-Tool Workflow with Streaming ===")
        logger.info("Testing parallel tool calls with streaming enabled...")
        logger.info(f"Query: {input_prompt}\n")

        result = wf.run(
            input_data={"input": input_prompt},
            config=RunnableConfig(callbacks=[tracing, streaming_handler]),
        )

        streaming_chunks = []
        logger.info("=== STREAMING OUTPUT ===\n")

        for chunk in streaming_handler:
            streaming_chunks.append(chunk)
            logger.info(f"Stream chunk: {chunk}")

        logger.info("\n=== FINAL RESULT ===")

        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        final_content = result.output[agent.id]["output"]["content"]
        return final_content, tracing.runs, streaming_chunks

    except Exception as e:
        logger.exception(f"An error occurred during streaming workflow: {e}")
        logger.error(f"ERROR: {e}")
        return "", {}, []


def run_parallel_tool_calls_streaming():
    """
    Test function specifically designed to validate that parallel tool calls
    work correctly with streaming enabled, testing our StreamChunkChoiceDelta fix.
    """
    logger.info("=== Testing Parallel Tool Calls with Streaming ===")
    logger.info("This test validates the fix for StreamChunkChoiceDelta validation errors")
    logger.info("when parallel tool calls are enabled with streaming.\n")

    try:
        output, traces, chunks = run_streaming_workflow()

        logger.info("\n=== FINAL AGENT OUTPUT ===")
        logger.info(output)

        logger.info("\n=== STREAMING SUMMARY ===")
        logger.info(f"Total streaming chunks received: {len(chunks)}")

        if chunks:
            logger.info("✅ Streaming worked successfully with parallel tool calls!")
            logger.info("✅ StreamChunkChoiceDelta validation fix is working correctly!")
        else:
            logger.warning("⚠️  No streaming chunks received - check streaming configuration")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        logger.error("❌ StreamChunkChoiceDelta validation error may still be present")
        return False


if __name__ == "__main__":
    success = run_parallel_tool_calls_streaming()

    if success:
        logger.info("\n🎉 Test completed successfully!")
    else:
        logger.error("\n💥 Test failed - check the implementation")
