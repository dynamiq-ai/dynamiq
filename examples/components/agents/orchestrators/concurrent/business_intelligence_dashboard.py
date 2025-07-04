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
    "Create a comprehensive business intelligence dashboard for a global technology company. "
    "The dashboard should include financial performance metrics, "
    "market trends analysis, competitive landscape overview, "
    "customer sentiment insights, and strategic recommendations. "
    "Focus on the latest market developments, emerging technologies, "
    "and potential growth opportunities. "
    "Incorporate data from financial reports, market research, "
    "social media sentiment, and industry publications. "
    "The dashboard should be suitable for executive-level decision-making, "
    "with clear visualizations and actionable insights. "
)

ALTERNATIVE_TASKS = [
    (
        "Build a retail analytics dashboard: sales performance, customer behavior, "
        "inventory analysis, seasonal trends, competitive pricing, and market opportunities. "
        "Include predictive models for demand forecasting and profit optimization."
    ),
    (
        "Create a cryptocurrency market intelligence report: price analysis, "
        "trading volumes, market sentiment, regulatory developments, "
        "institutional adoption trends, and risk assessment with predictions."
    ),
    (
        "Develop a startup ecosystem analysis dashboard: funding trends, "
        "unicorn companies, investor activity, sector performance, "
        "geographic distribution, and success pattern analysis."
    ),
]


def create_bi_orchestrator():
    """Create the ConcurrentOrchestrator for business intelligence tasks."""

    llm = setup_llm(model_provider="gpt", model_name="o4-mini", temperature=0.1, max_tokens=4000)

    web_search_tool = TavilyTool(
        connection=TavilyConnection(),
        name="Market Research Tool",
    )

    data_search_tool = ExaTool(
        connection=Exa(),
        name="Business Data Search",
    )

    web_scraper_tool = ZenRowsTool(
        connection=ZenRowsConnection(),
        name="Data Scraper",
    )

    analytics_tool = E2BInterpreterTool(
        connection=E2BConnection(),
        name="Analytics Engine",
    )

    market_data_agent = ReActAgent(
        name="Market Data Collector",
        llm=llm,
        tools=[web_search_tool, data_search_tool, web_scraper_tool],
        role=(
            "Expert at gathering market data, financial information, and business metrics "
            "from various sources including financial reports, market research, and "
            "industry publications. Focuses on accuracy and data quality."
        ),
        max_loops=8,
        inference_mode=InferenceMode.XML,
    )

    financial_analyst = ReActAgent(
        name="Financial Analysis Specialist",
        llm=llm,
        tools=[analytics_tool, data_search_tool],
        role=(
            "Expert in financial analysis, ratio calculations, trend analysis, and "
            "performance metrics. Processes financial data to extract meaningful "
            "insights about company performance and market dynamics."
        ),
        max_loops=8,
        inference_mode=InferenceMode.XML,
    )

    competitive_analyst = ReActAgent(
        name="Competitive Intelligence Analyst",
        llm=llm,
        tools=[web_search_tool, web_scraper_tool],
        role=(
            "Specialist in competitive analysis, market positioning, and strategic "
            "intelligence. Analyzes competitor activities, market share, product "
            "portfolios, and competitive advantages."
        ),
        max_loops=6,
        inference_mode=InferenceMode.XML,
    )

    data_scientist = ReActAgent(
        name="Data Science and Visualization Expert",
        llm=llm,
        tools=[analytics_tool],
        role=(
            "Expert in data analysis, statistical modeling, predictive analytics, "
            "and data visualization. Creates charts, graphs, and statistical models "
            "to support business decisions and identify trends."
        ),
        max_loops=8,
        inference_mode=InferenceMode.XML,
    )

    sentiment_analyst = SimpleAgent(
        name="Market Sentiment Analyst",
        llm=llm,
        role=(
            "Specialist in analyzing market sentiment, customer feedback, and "
            "public perception. Interprets qualitative data, social media trends, "
            "and customer satisfaction metrics to gauge market dynamics."
        ),
    )

    business_strategist = SimpleAgent(
        name="Business Strategy Consultant",
        llm=llm,
        role=(
            "Expert in strategic analysis, business planning, and executive reporting. "
            "Synthesizes complex business data into actionable insights, strategic "
            "recommendations, and clear executive summaries."
        ),
    )

    manager = ConcurrentAgentManager(
        llm=llm,
        name="BI Dashboard Manager",
    )

    orchestrator = ConcurrentOrchestrator(
        name="Business Intelligence Orchestrator",
        manager=manager,
        agents=[
            market_data_agent,
            financial_analyst,
            competitive_analyst,
            data_scientist,
            sentiment_analyst,
            business_strategist,
        ],
        tools=[web_search_tool, data_search_tool, web_scraper_tool, analytics_tool],
        max_concurrency=4,
        task_timeout=1200,
        enable_complexity_analysis=True,
        enable_context_sharing=True,
    )

    return orchestrator


def run_bi_dashboard_example():
    """Run the main business intelligence dashboard example."""
    print("=== Business Intelligence Dashboard with ConcurrentOrchestrator ===\n")

    orchestrator = create_bi_orchestrator()

    print("Created BI orchestrator with:")
    print(f"- {len(orchestrator.agents)} specialized analysts")
    print(f"- {len(orchestrator.tools)} data gathering and analysis tools")
    print("- Parallel data streams for comprehensive analysis")
    print("- Advanced analytics and visualization capabilities")
    print()

    print("BI Dashboard Task:")
    print(f'"{INPUT_TASK[:200]}..."')
    print()

    print("Starting parallel business intelligence analysis...")
    print("=" * 60)

    try:
        result = orchestrator.run(
            input_data={"input": INPUT_TASK},
            config=None,
        )

        print("\n" + "=" * 60)
        print("BUSINESS INTELLIGENCE DASHBOARD RESULTS")
        print("=" * 60)

        if result and result.output:
            output_content = result.output.get("content")
            if output_content:
                print(output_content)
            else:
                print("No content in result output")
                print(f"Result output keys: {list(result.output.keys()) if result.output else 'None'}")
                print(f"Full result: {result}")
        else:
            print("No result returned from orchestrator")
            print(f"Result object: {result}")

    except Exception as e:
        print(f"Error during BI analysis: {e}")
        import traceback

        traceback.print_exc()


def run_financial_analysis_example():
    """Run a focused financial analysis example."""
    print("\n=== Financial Performance Analysis Example ===\n")

    financial_task = (
        "Analyze the financial performance of the top 5 semiconductor companies "
        "in 2024-2025. Compare revenue growth, profit margins, R&D spending, "
        "market capitalization trends, and stock performance. Include analysis "
        "of how AI demand has impacted their business and future projections."
    )

    orchestrator = create_bi_orchestrator()

    print(f"Financial Analysis Task: {financial_task[:100]}...")
    print("This demonstrates parallel financial data gathering and analysis.")
    print()

    try:
        result = orchestrator.run(
            input_data={"input": financial_task},
            config=None,
        )

        print("FINANCIAL ANALYSIS RESULTS:")
        print("-" * 40)
        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print(f"Full result: {result}")

    except Exception as e:
        print(f"Error: {e}")


def run_market_trends_example():
    """Run a market trends analysis example."""
    print("\n=== Market Trends Analysis Example ===\n")

    trends_task = (
        "Create a comprehensive analysis of emerging technology trends in 2025: "
        "identify top 10 trending technologies, analyze investment flows, "
        "assess market readiness, evaluate business impact potential, "
        "and provide recommendations for business strategy adaptation."
    )

    orchestrator = create_bi_orchestrator()

    print(f"Market Trends Task: {trends_task[:100]}...")
    print("This showcases parallel trend analysis across multiple domains.")
    print()

    try:
        result = orchestrator.run(
            input_data={"input": trends_task},
            config=None,
        )

        print("MARKET TRENDS ANALYSIS RESULTS:")
        print("-" * 40)
        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print(f"Full result: {result}")

    except Exception as e:
        print(f"Error: {e}")


def demonstrate_competitive_intelligence():
    """Demonstrate competitive intelligence capabilities."""
    print("\n=== Competitive Intelligence Demonstration ===\n")

    competitive_task = (
        "Conduct competitive intelligence analysis for the electric vehicle market: "
        "analyze Tesla, BYD, Volkswagen, and emerging EV companies. "
        "Compare market share, technology capabilities, production capacity, "
        "pricing strategies, and strategic partnerships. Identify competitive "
        "advantages and market opportunities."
    )

    orchestrator = create_bi_orchestrator()

    print("This task demonstrates parallel competitive analysis:")
    print(f'"{competitive_task}"')
    print()
    print("Expected analysis streams:")
    print("1. Company-specific analysis (parallel)")
    print("2. Market share and positioning (parallel)")
    print("3. Technology and innovation comparison (parallel)")
    print("4. Strategic synthesis and recommendations (sequential)")
    print()

    try:
        result = orchestrator.run(
            input_data={"input": competitive_task},
            config=None,
        )

        print("COMPETITIVE INTELLIGENCE RESULTS:")
        print("-" * 40)
        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print(f"Full result: {result}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":

    run_bi_dashboard_example()

    # Uncomment to run additional examples
    # run_financial_analysis_example()
    # run_market_trends_example()
    # demonstrate_competitive_intelligence()

    print("\n" + "=" * 60)
    print("Business Intelligence example completed!")
    print("The ConcurrentOrchestrator demonstrated:")
    print("✓ Parallel data collection from multiple sources")
    print("✓ Specialized financial and market analysis")
    print("✓ Competitive intelligence gathering")
    print("✓ Statistical analysis and visualization")
    print("✓ Strategic synthesis and business recommendations")
    print("✓ Comprehensive BI dashboard creation")
