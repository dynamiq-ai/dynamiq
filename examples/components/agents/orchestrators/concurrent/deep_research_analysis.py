from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Exa
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.connections import ZenRows as ZenRowsConnection
from dynamiq.nodes.agents.orchestrators.concurrent import ConcurrentOrchestrator
from dynamiq.nodes.agents.orchestrators.concurrent_manager import ConcurrentAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

INPUT_TASK = (
    "Conduct a comprehensive analysis of the current state and future potential of "
    "quantum computing in 2025. I need insights on: "
    "1) Latest technological breakthroughs and research papers, "
    "2) Market landscape including key players and investments, "
    "3) Commercial applications currently in development, "
    "4) Regulatory and policy considerations, "
    "5) Projected timeline for widespread adoption. "
    "Please provide data analysis, trend predictions, and actionable recommendations."
)


ALTERNATIVE_TASKS = [
    (
        "Analyze the impact of AI on healthcare in 2025. Research current applications, "
        "regulatory challenges, market opportunities, ethical considerations, and "
        "provide predictions with supporting data analysis."
    ),
    (
        "Investigate the renewable energy transition: current technologies, policy landscape, "
        "investment trends, environmental impact, and economic implications. "
        "Include market analysis and future projections."
    ),
    (
        "Study the evolution of remote work technologies post-2024. Analyze collaboration tools, "
        "productivity metrics, security considerations, and workplace culture changes. "
        "Provide recommendations for organizations adapting to hybrid models."
    ),
]


def create_research_orchestrator():
    """Create and configure the ConcurrentOrchestrator for research tasks."""

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0.1, max_tokens=4000)

    search_tool = TavilyTool(
        connection=TavilyConnection(),
        name="Web Search",
    )

    academic_search_tool = ExaTool(
        connection=Exa(),
        name="Academic Search",
    )

    scraping_tool = ZenRowsTool(
        connection=ZenRowsConnection(),
        name="Web Scraper",
    )

    analysis_tool = E2BInterpreterTool(
        connection=E2BConnection(),
        name="Data Analyzer",
    )

    literature_agent = ReActAgent(
        name="Academic Literature Researcher",
        llm=llm,
        tools=[academic_search_tool, search_tool],
        role=(
            "Expert at finding and analyzing academic papers, research publications, "
            "and scientific literature. Focuses on peer-reviewed sources and "
            "technical documentation."
        ),
        max_loops=8,
        inference_mode=InferenceMode.XML,
    )

    market_agent = ReActAgent(
        name="Market Intelligence Analyst",
        llm=llm,
        tools=[search_tool, scraping_tool],
        role=(
            "Specialist in market research, competitive analysis, industry trends, "
            "and business intelligence. Focuses on commercial applications, "
            "investments, and market dynamics."
        ),
        max_loops=8,
        inference_mode=InferenceMode.XML,
    )

    data_agent = ReActAgent(
        name="Data Analysis Specialist",
        llm=llm,
        tools=[analysis_tool],
        role=(
            "Expert in data processing, statistical analysis, trend identification, "
            "and quantitative research. Processes numerical data and creates "
            "visualizations and predictions."
        ),
        max_loops=6,
        inference_mode=InferenceMode.XML,
    )

    policy_agent = SimpleAgent(
        name="Policy and Regulation Analyst",
        llm=llm,
        role=(
            "Specialist in regulatory frameworks, policy analysis, government initiatives, "
            "and compliance requirements. Analyzes legal and policy implications "
            "of emerging technologies and market trends."
        ),
    )

    synthesis_agent = SimpleAgent(
        name="Research Synthesis Expert",
        llm=llm,
        role=(
            "Expert at synthesizing complex research from multiple sources into "
            "coherent, actionable insights. Specializes in identifying patterns, "
            "drawing conclusions, and making evidence-based recommendations."
        ),
    )

    manager = ConcurrentAgentManager(
        llm=llm,
        name="Research Coordination Manager",
    )

    orchestrator = ConcurrentOrchestrator(
        name="Deep Research Orchestrator",
        manager=manager,
        agents=[literature_agent, market_agent, data_agent, policy_agent, synthesis_agent],
        tools=[search_tool, academic_search_tool, scraping_tool, analysis_tool],
        max_concurrency=4,
        task_timeout=600,
        enable_complexity_analysis=True,
        enable_context_sharing=True,
    )

    return orchestrator


def run_research_example():
    """Run the deep research analysis example."""
    print("=== Deep Research Analysis with ConcurrentOrchestrator ===\n")

    orchestrator = create_research_orchestrator()

    print("Created orchestrator with:")
    print(f"- {len(orchestrator.agents)} specialized agents")
    print(f"- {len(orchestrator.tools)} research tools")
    print(f"- Max concurrency: {orchestrator.max_concurrency}")
    print(f"- Task timeout: {orchestrator.task_timeout}s")
    print(f"- Complexity analysis: {orchestrator.enable_complexity_analysis}")
    print(f"- Context sharing: {orchestrator.enable_context_sharing}")
    print()

    print("Research Task:")
    print(f'"{INPUT_TASK[:200]}..."')
    print()

    print("Starting parallel research execution...")
    print("=" * 60)

    try:

        result = orchestrator.run(
            input_data={"input": INPUT_TASK},
            config=None,
        )

        print("\n" + "=" * 60)
        print("RESEARCH RESULTS")
        print("=" * 60)

        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print("No content in result output")
            print(f"Full result: {result}")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()


def run_market_analysis_example():
    """Run a focused market analysis example."""
    print("\n=== Market Analysis Example ===\n")

    market_task = (
        "Analyze the AI chip market in 2025. Research major players (NVIDIA, AMD, Intel, "
        "startups), recent developments, market size projections, and investment trends. "
        "Include competitive analysis and growth predictions."
    )

    orchestrator = create_research_orchestrator()

    print(f"Market Analysis Task: {market_task[:100]}...")
    print()

    try:
        result = orchestrator.run(
            input_data={"input": market_task},
            config=None,
        )

        print("MARKET ANALYSIS RESULTS:")
        print("-" * 40)
        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print(f"Full result: {result}")

    except Exception as e:
        print(f"Error: {e}")


def demonstrate_parallel_capabilities():
    """Demonstrate the parallel execution capabilities."""
    print("\n=== Parallel Execution Demonstration ===\n")

    parallel_task = (
        "Simultaneously research three topics: "
        "1) Latest developments in large language models, "
        "2) Current state of autonomous vehicle technology, "
        "3) Quantum computing breakthroughs in 2024-2025. "
        "For each topic, gather recent news, technical papers, and market data. "
        "Then analyze trends and provide a comparative analysis."
    )

    orchestrator = create_research_orchestrator()

    print("This task is designed to showcase parallel execution:")
    print(f'"{parallel_task}"')
    print()
    print("The orchestrator should identify the three independent research streams")
    print("and execute them in parallel for maximum efficiency.")
    print()

    try:
        result = orchestrator.run(
            input_data={"input": parallel_task},
            config=None,
        )

        print("PARALLEL EXECUTION RESULTS:")
        print("-" * 40)
        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print(f"Full result: {result}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":

    run_research_example()

    # run_market_analysis_example()
    # demonstrate_parallel_capabilities()

    print("\n" + "=" * 60)
    print("Example completed. The ConcurrentOrchestrator demonstrated:")
    print("✓ Complex task decomposition and parallel planning")
    print("✓ Intelligent agent and tool coordination")
    print("✓ Context sharing across parallel research streams")
    print("✓ Synthesis of results from multiple sources")
    print("✓ Efficient resource utilization through parallelization")
