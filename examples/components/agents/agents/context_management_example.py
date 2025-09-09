from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B, ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from examples.llm_setup import setup_llm

AGENT_ROLE = """
You are an AI assistant specialized in data analysis and web research.

Your unique capabilities include:
1. **Smart Tool Caching**: Tool outputs are automatically cached and can be referenced directly
2. **Direct Tool Return**: You can return raw tool outputs using:
   - Quoted JSON: <answer type="tool_output" action="tool_name" action_input='{"key": "value"}'>
3. **Enhanced Memory**: Previous tool outputs are preserved and can be referenced by action and input

When to use direct tool return:
- User asks for raw data or code output
- Tool produces structured data that shouldn't be summarized
- User explicitly wants unprocessed results

Available tools:
- web-search: For finding current information online
- code-executor: For data analysis, calculations, and programming tasks

Always be explicit about which approach you're using and why.
"""

QUERY1 = """Return result of query search 'ai 2025'"""
QUERY2 = """Research the latest developments in
            AI agents and multi-agent systems at this months;
            then create a report for perspectives;
            regarding all areas of ML/DL and AI"""


def create_enhanced_context_agent() -> ReActAgent:
    """
    Create a ReAct agent with enhanced context management features.

    Returns:
        ReActAgent with context management enabled
    """
    exa_conn = ScaleSerp()
    e2b_conn = E2B()

    search_tool = ScaleSerpTool(connection=exa_conn, name="web-search")
    code_tool = E2BInterpreterTool(connection=e2b_conn, name="code-executor")

    llm = setup_llm(model_provider="claude", model_name="claude-3-5-sonnet-20240620")

    context_config = SummarizationConfig(
        enabled=True,
        max_token_context_length=10000,
        context_usage_ratio=0.8,
        context_history_length=10,
    )

    agent = ReActAgent(
        name="Context Agent",
        llm=llm,
        tools=[search_tool, code_tool],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=True,
        direct_tool_output_enabled=True,
        summarization_config=context_config,
        max_loops=15,
        verbose=True,
    )

    return agent


def demonstrate_direct_tool_return():
    """
    Specific demonstration of direct tool output return feature.
    """
    print("Direct Tool Return Feature Demo")
    print("=" * 50)

    agent = create_enhanced_context_agent()
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(input_data={"input": QUERY1}, config=RunnableConfig(callbacks=[tracing]))

        output = result.output[agent.id]["output"]["content"]

        print("Agent Response (Direct Tool Return):")
        print(output)

        print("\nRegistry Status:")
        print(f"   - Tools executed: {len(agent._tool_cache)}")
        available_tools = [(entry.action, entry.action_input) for entry in agent._tool_cache.keys()]
        print(f"   - Available tools: {available_tools}")
        if agent._tool_cache:
            print(f"   - Cached tool outputs: {len(agent._tool_cache)} entries")

    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    demonstrate_direct_tool_return()
