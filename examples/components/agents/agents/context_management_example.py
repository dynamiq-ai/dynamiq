from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B, ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.agents.utils import SummarizationConfig, ToolOutputConfig
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from examples.llm_setup import setup_llm

AGENT_ROLE = """
You are an AI assistant specialized in data analysis and web research.

Your unique capabilities include:
1. **Smart Context Management**: Tool outputs are automatically chunked and stored with unique IDs
2. **Direct Tool Return**: You can return raw tool outputs using: <answer type="tool_output" tool_id="tool_X">
3. **Enhanced Memory**: Previous tool outputs are preserved and can be referenced by ID

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
    tool_config = ToolOutputConfig(max_chars=100, chunk_strategy="smart", summary_strategy="structured")

    context_config = SummarizationConfig(
        enabled=True,
        max_token_context_length=10000,
        context_usage_ratio=0.8,
        context_history_length=10,
        tool_output=tool_config,
    )

    agent = ReActAgent(
        name="Context Agent",
        llm=llm,
        tools=[search_tool, code_tool],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.DEFAULT,
        parallel_tool_calls_enabled=True,
        direct_tool_output_enabled=True,
        summarization_config=context_config,
        max_loops=15,
        verbose=True,
    )

    return agent


def demonstrate_context_features():
    """
    Demonstrate various context management features with different scenarios.
    """
    agent = create_enhanced_context_agent()

    scenarios = [
        {
            "name": "📊 Data Analysis with Direct Return",
            "query": QUERY2,
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{scenario['name']} (Scenario {i})")
        print("-" * 50)
        print(f"Query: {scenario['query']}")
        print()

        tracing = TracingCallbackHandler()
        wf = Workflow(flow=Flow(nodes=[agent]))

        try:
            result = wf.run(input_data={"input": scenario["query"]}, config=RunnableConfig(callbacks=[tracing]))

            output = result.output[agent.id]["output"]["content"]
            print("Agent Response:")
            print(output[:1000] + "..." if len(output) > 1000 else output)

            print("Context Info:")
            print(f"   - Tool outputs in registry: {len(agent._tool_chunked_outputs)}")
            print(f"   - Available tool IDs: {list(agent._tool_chunked_outputs.keys())}")

            if agent._tool_chunked_outputs:
                latest_id = list(agent._tool_chunked_outputs.keys())[-1]
                latest_output = agent._tool_chunked_outputs[latest_id]
                print(f"   - Latest tool ({latest_id}): {latest_output.tool_chunks[0][:200]}...")

        except Exception as e:
            print(f"Error in scenario {i}: {e}")

        print("\n" + "=" * 60)


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
        print(f"   - Tools executed: {len(agent._tool_chunked_outputs)}")
        for tool_id, tool_output in agent._tool_chunked_outputs.items():
            print(
                f"   - {tool_id}: {len(tool_output.tool_chunks)} chunks, " f"{tool_output.original_size} chars original"
            )

    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    demonstrate_context_features()
    demonstrate_direct_tool_return()
