from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.nodes.agents.orchestrators.intelligent_parallel import IntelligentParallelOrchestrator
from dynamiq.nodes.agents.orchestrators.intelligent_parallel_manager import IntelligentParallelAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

INPUT_TASK = (
    "I need help with three things: "
    "1) Find the current weather in New York City, "
    "2) Calculate the compound interest on $10,000 invested at 5% annually for 10 years, "
    "3) Search for the latest news about artificial intelligence breakthroughs. "
    "Please handle all three tasks efficiently and provide a summary."
)

TEST_TASKS = [
    (
        "Help me with: finding current Bitcoin price, calculating tax on $50k income, "
        "and searching for best Python learning resources."
    ),
    (
        "I need: today's stock market summary, a simple recipe for pasta, "
        "and information about renewable energy trends."
    ),
    ("Please: check Tesla's latest news, solve 15% tip on $87.50 bill, " "and find good books about machine learning."),
]


def create_simple_orchestrator():
    """Create a simple IntelligentParallelOrchestrator for demonstration."""

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0.2, max_tokens=2000)

    search_tool = TavilyTool(
        connection=TavilyConnection(),
        name="Web Search",
    )

    calculator_tool = E2BInterpreterTool(
        connection=E2BConnection(),
        name="Calculator",
    )

    search_agent = ReActAgent(
        name="Search Agent",
        llm=llm,
        tools=[search_tool],
        role="Expert at finding current information on the web.",
        max_loops=3,
        inference_mode=InferenceMode.DEFAULT,
    )

    calculation_agent = ReActAgent(
        name="Calculation Agent",
        llm=llm,
        tools=[calculator_tool],
        role="Expert at mathematical calculations and data analysis.",
        max_loops=3,
        inference_mode=InferenceMode.DEFAULT,
    )

    general_agent = SimpleAgent(
        name="General Assistant",
        llm=llm,
        role="Helpful assistant for general questions and information.",
    )

    manager = IntelligentParallelAgentManager(
        llm=llm,
        name="Simple Manager",
    )

    orchestrator = IntelligentParallelOrchestrator(
        name="Simple Parallel Orchestrator",
        manager=manager,
        agents=[search_agent, calculation_agent, general_agent],
        tools=[search_tool, calculator_tool],
        max_concurrency=2,
        task_timeout=300,
        enable_complexity_analysis=True,
        enable_context_sharing=True,
    )

    return orchestrator


def run_simple_demo():
    """Run the simple parallel execution demo."""
    print("=== Simple Parallel Execution Demo ===\n")

    orchestrator = create_simple_orchestrator()

    print("This demo shows basic parallel execution with the IntelligentParallelOrchestrator")
    print("Configuration:")
    print(f"- {len(orchestrator.agents)} agents (Search, Calculation, General)")
    print(f"- {len(orchestrator.tools)} tools (Web Search, Calculator)")
    print(f"- Max concurrency: {orchestrator.max_concurrency}")
    print()

    print("Task: Multi-part request that can be handled in parallel")
    print(f'"{INPUT_TASK}"')
    print()

    print("Expected behavior:")
    print("- The orchestrator should identify 3 independent sub-tasks")
    print("- Tasks 1 and 3 (weather, news) can use the Search Agent in parallel")
    print("- Task 2 (calculation) can use the Calculation Agent in parallel")
    print("- Results should be synthesized into a comprehensive answer")
    print()

    print("Starting execution...")
    print("-" * 50)

    try:
        result = orchestrator.run(
            input_data={"input": INPUT_TASK},
            config=None,
        )

        print("\n" + "-" * 50)
        print("RESULTS:")
        print("-" * 50)

        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print("No content in result")
            print(f"Full result structure: {result}")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()


def run_quick_tests():
    """Run quick tests with different task types."""
    print("\n=== Quick Parallel Tests ===\n")

    orchestrator = create_simple_orchestrator()

    for i, task in enumerate(TEST_TASKS, 1):
        print(f"Test {i}: {task[:80]}...")

        try:
            result = orchestrator.run(
                input_data={"input": task},
                config=None,
            )

            output_content = result.output.get("content")
            if output_content:
                print(f"✓ Success: {output_content[:150]}...")
            else:
                print("✗ No content returned")

        except Exception as e:
            print(f"✗ Error: {e}")

        print()


def demonstrate_task_analysis():
    """Demonstrate the task complexity analysis feature."""
    print("\n=== Task Complexity Analysis Demo ===\n")

    orchestrator = create_simple_orchestrator()

    # Test different complexity levels
    test_cases = [
        "What's 2+2?",
        "Find the weather in London",
        "Research quantum computing and write a detailed analysis",
        "Calculate mortgage payments for $300k at 6.5% for 30 years",
        "Compare the top 5 programming languages and provide recommendations",
    ]

    print("Testing task complexity analysis:")
    print()

    for task in test_cases:
        complexity = orchestrator.analyze_task_complexity(task)
        print(f"Task: '{task}'")
        print(f"Complexity: {complexity.value}")
        print()


def show_orchestrator_info():
    """Display information about the orchestrator configuration."""
    print("\n=== Orchestrator Configuration ===\n")

    orchestrator = create_simple_orchestrator()

    print("Agents:")
    for i, agent in enumerate(orchestrator.agents, 1):
        print(f"{i}. {agent.name}")
        if hasattr(agent, "role") and agent.role:
            print(f"   Role: {agent.role}")
        if hasattr(agent, "tools") and agent.tools:
            tools = [tool.name for tool in agent.tools]
            print(f"   Tools: {', '.join(tools)}")
        print()

    print("Tools:")
    for i, tool in enumerate(orchestrator.tools, 1):
        print(f"{i}. {tool.name}")
        if hasattr(tool, "description") and tool.description:
            print(f"   Description: {tool.description[:100]}...")
        print()

    print("Configuration:")
    print(f"- Max Concurrency: {orchestrator.max_concurrency}")
    print(f"- Task Timeout: {orchestrator.task_timeout}s")
    print(f"- Complexity Analysis: {orchestrator.enable_complexity_analysis}")
    print(f"- Context Sharing: {orchestrator.enable_context_sharing}")


if __name__ == "__main__":
    show_orchestrator_info()

    run_simple_demo()

    # run_quick_tests()
    # demonstrate_task_analysis()

    print("\n" + "=" * 50)
    print("Simple demo completed!")
    print("\nKey takeaways:")
    print("✓ The orchestrator automatically identifies parallel opportunities")
    print("✓ Independent tasks are executed concurrently for efficiency")
    print("✓ Different agent types handle different task categories")
    print("✓ Results are intelligently synthesized into coherent responses")
    print("✓ The system gracefully handles mixed complexity tasks")
    print("\nTry modifying the INPUT_TASK to test different scenarios!")
