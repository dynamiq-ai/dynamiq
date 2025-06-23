from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Exa
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.nodes.agents.orchestrators.intelligent_parallel import IntelligentParallelOrchestrator
from dynamiq.nodes.agents.orchestrators.intelligent_parallel_manager import IntelligentParallelAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

INPUT_TASK = (
    "Create a comprehensive learning package about 'Building RAG Applications with Python'. "
    "The package should include: "
    "1) A detailed technical tutorial with code examples, "
    "2) A beginner-friendly blog post explaining the concepts, "
    "3) Working Python code with proper documentation, "
    "4) A comparison of different RAG frameworks and tools, "
    "5) Best practices and common pitfalls guide, "
    "6) Interactive examples and exercises. "
    "Research the latest developments, ensure accuracy, and make it practical and engaging."
)

ALTERNATIVE_TASKS = [
    (
        "Create a complete marketing package for a new SaaS product: "
        "landing page copy, technical documentation, demo scripts, "
        "customer onboarding materials, and FAQ content. "
        "Research competitors and industry best practices."
    ),
    (
        "Develop educational content about machine learning fundamentals: "
        "interactive tutorial, code examples, visual explanations, "
        "practice exercises, and assessment materials. "
        "Include multiple learning styles and difficulty levels."
    ),
    (
        "Build a comprehensive API documentation package: "
        "technical reference, getting started guide, code samples "
        "in multiple languages, troubleshooting guide, and "
        "real-world use case examples."
    ),
]


def create_content_orchestrator():
    """Create the IntelligentParallelOrchestrator for content creation."""

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0.3, max_tokens=4000)

    research_tool = TavilyTool(
        connection=TavilyConnection(),
        name="Research Tool",
    )

    technical_search_tool = ExaTool(
        connection=Exa(),
        name="Technical Search",
    )

    code_executor = E2BInterpreterTool(
        connection=E2BConnection(),
        name="Code Executor",
    )

    research_agent = ReActAgent(
        name="Content Research Specialist",
        llm=llm,
        tools=[research_tool, technical_search_tool],
        role=(
            "Expert at researching topics for content creation. Gathers current "
            "information, trends, best practices, and examples from reliable sources. "
            "Focuses on accuracy and relevance for the target audience."
        ),
        max_loops=6,
        inference_mode=InferenceMode.XML,
    )

    technical_writer = SimpleAgent(
        name="Technical Writer",
        llm=llm,
        role=(
            "Specialist in creating clear, comprehensive technical documentation. "
            "Writes detailed tutorials, guides, and reference materials that are "
            "accurate, well-structured, and easy to follow. Excellent at explaining "
            "complex concepts with appropriate technical depth."
        ),
    )

    copywriter = SimpleAgent(
        name="Marketing Copywriter",
        llm=llm,
        role=(
            "Expert at creating engaging, accessible content for general audiences. "
            "Writes blog posts, marketing copy, and educational content that is "
            "compelling, clear, and tailored to the target audience. "
            "Focuses on benefits and practical value."
        ),
    )

    code_specialist = ReActAgent(
        name="Code Development Specialist",
        llm=llm,
        tools=[code_executor],
        role=(
            "Expert programmer who creates, tests, and documents code examples. "
            "Writes clean, well-commented code with proper error handling and "
            "follows best practices. Ensures code examples are working and practical."
        ),
        max_loops=8,
        inference_mode=InferenceMode.XML,
    )

    content_strategist = SimpleAgent(
        name="Content Strategy Analyst",
        llm=llm,
        role=(
            "Specialist in content strategy, user experience, and information architecture. "
            "Analyzes and compares different approaches, identifies best practices, "
            "and provides strategic recommendations for content structure and presentation."
        ),
    )

    editor_agent = SimpleAgent(
        name="Content Editor and QA",
        llm=llm,
        role=(
            "Expert editor who reviews, improves, and ensures quality of all content. "
            "Checks for accuracy, consistency, clarity, and completeness. "
            "Integrates different content pieces into cohesive packages and "
            "suggests improvements for better user experience."
        ),
    )

    manager = IntelligentParallelAgentManager(
        llm=llm,
        name="Content Creation Manager",
    )

    orchestrator = IntelligentParallelOrchestrator(
        name="Multi-Modal Content Creator",
        manager=manager,
        agents=[
            research_agent,
            technical_writer,
            copywriter,
            code_specialist,
            content_strategist,
            editor_agent,
        ],
        tools=[research_tool, technical_search_tool, code_executor],
        max_concurrency=3,
        task_timeout=900,
        enable_complexity_analysis=True,
        enable_context_sharing=True,
    )

    return orchestrator


def run_content_creation_example():
    """Run the multi-modal content creation example."""
    print("=== Multi-Modal Content Creation with IntelligentParallelOrchestrator ===\n")

    orchestrator = create_content_orchestrator()

    print("Created content creation orchestrator with:")
    print(f"- {len(orchestrator.agents)} specialized content agents")
    print(f"- {len(orchestrator.tools)} content creation tools")
    print("- Parallel content streams for efficiency")
    print("- Quality assurance and integration workflows")
    print()

    print("Content Creation Task:")
    print(f'"{INPUT_TASK[:200]}..."')
    print()

    print("Starting parallel content creation...")
    print("=" * 60)

    try:
        result = orchestrator.run(
            input_data={"input": INPUT_TASK},
            config=None,
        )

        print("\n" + "=" * 60)
        print("CONTENT CREATION RESULTS")
        print("=" * 60)

        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print("No content in result output")
            print(f"Full result: {result}")

    except Exception as e:
        print(f"Error during content creation: {e}")
        import traceback

        traceback.print_exc()


def run_blog_tutorial_example():
    """Run a focused blog and tutorial creation example."""
    print("\n=== Blog + Tutorial Creation Example ===\n")

    blog_task = (
        "Create both a beginner-friendly blog post and a technical tutorial about "
        "building a chatbot with OpenAI's API. The blog should be engaging and accessible, "
        "while the tutorial should be comprehensive with working code examples, "
        "error handling, and deployment instructions."
    )

    orchestrator = create_content_orchestrator()

    print(f"Dual Content Task: {blog_task[:100]}...")
    print("This demonstrates parallel creation of different content types.")
    print()

    try:
        result = orchestrator.run(
            input_data={"input": blog_task},
            config=None,
        )

        print("BLOG + TUTORIAL RESULTS:")
        print("-" * 40)
        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print(f"Full result: {result}")

    except Exception as e:
        print(f"Error: {e}")


def run_code_documentation_example():
    """Run a code and documentation creation example."""
    print("\n=== Code + Documentation Example ===\n")

    code_task = (
        "Build a complete Python package for text summarization using transformers. "
        "Include: working code with proper structure, comprehensive documentation, "
        "usage examples, test cases, and a README with installation instructions. "
        "Research current best practices and ensure the code is production-ready."
    )

    orchestrator = create_content_orchestrator()

    print(f"Code Development Task: {code_task[:100]}...")
    print("This showcases parallel code development and documentation creation.")
    print()

    try:
        result = orchestrator.run(
            input_data={"input": code_task},
            config=None,
        )

        print("CODE + DOCUMENTATION RESULTS:")
        print("-" * 40)
        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print(f"Full result: {result}")

    except Exception as e:
        print(f"Error: {e}")


def demonstrate_content_workflow():
    """Demonstrate the content creation workflow stages."""
    print("\n=== Content Workflow Demonstration ===\n")

    workflow_task = (
        "Create a complete onboarding package for new AI engineers: "
        "welcome guide, technical setup instructions, first-week learning path, "
        "coding standards document, and project templates. Research industry "
        "best practices and make it comprehensive yet approachable."
    )

    orchestrator = create_content_orchestrator()

    print("This task shows how the orchestrator handles complex, multi-component content:")
    print(f'"{workflow_task}"')
    print()
    print("Expected workflow:")
    print("1. Research phase (parallel)")
    print("2. Content creation (parallel streams)")
    print("3. Integration and QA (sequential)")
    print("4. Final packaging (sequential)")
    print()

    try:
        result = orchestrator.run(
            input_data={"input": workflow_task},
            config=None,
        )

        print("CONTENT WORKFLOW RESULTS:")
        print("-" * 40)
        output_content = result.output.get("content")
        if output_content:
            print(output_content)
        else:
            print(f"Full result: {result}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_content_creation_example()

    # Uncomment to run additional examples
    # run_blog_tutorial_example()
    # run_code_documentation_example()
    # demonstrate_content_workflow()

    print("\n" + "=" * 60)
    print("Content creation example completed!")
    print("The IntelligentParallelOrchestrator demonstrated:")
    print("✓ Parallel content research and creation")
    print("✓ Specialized agent coordination for different content types")
    print("✓ Code development with testing and documentation")
    print("✓ Quality assurance and content integration")
    print("✓ Efficient multi-modal content workflows")
