from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Exa
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.connections import ZenRows as ZenRowsConnection
from dynamiq.nodes.agents.orchestrators.intelligent_parallel import IntelligentParallelOrchestrator
from dynamiq.nodes.agents.orchestrators.intelligent_parallel_manager import IntelligentParallelAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm


def create_conversational_assistant():
    """Create an IntelligentParallelOrchestrator that handles both chat and complex tasks."""

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0.3, max_tokens=3000)

    search_tool = TavilyTool(
        connection=TavilyConnection(),
        name="Web Search",
    )

    research_tool = ExaTool(
        connection=Exa(),
        name="Deep Research",
    )

    scraper_tool = ZenRowsTool(
        connection=ZenRowsConnection(),
        name="Web Scraper",
    )

    code_tool = E2BInterpreterTool(
        connection=E2BConnection(),
        name="Code Executor",
    )

    research_agent = ReActAgent(
        name="Research Assistant",
        llm=llm,
        tools=[search_tool, research_tool, scraper_tool],
        role=(
            "Expert research assistant capable of gathering information from various sources, "
            "analyzing data, and providing comprehensive insights on any topic. "
            "Excellent at fact-checking and finding reliable sources."
        ),
        max_loops=5,
        inference_mode=InferenceMode.XML,
    )

    technical_agent = ReActAgent(
        name="Technical Specialist",
        llm=llm,
        tools=[code_tool, search_tool],
        role=(
            "Technical expert specializing in programming, data analysis, calculations, "
            "and problem-solving. Can write, execute, and debug code in various languages "
            "and help with technical challenges."
        ),
        max_loops=6,
        inference_mode=InferenceMode.XML,
    )

    creative_agent = SimpleAgent(
        name="Creative Assistant",
        llm=llm,
        role=(
            "Creative and communication specialist excellent at writing, editing, "
            "brainstorming, content creation, and helping with creative projects. "
            "Provides engaging and well-structured responses."
        ),
    )

    analytical_agent = SimpleAgent(
        name="Analysis Expert",
        llm=llm,
        role=(
            "Analytical specialist focused on problem-solving, strategic thinking, "
            "planning, and decision support. Excellent at breaking down complex "
            "problems and providing structured solutions."
        ),
    )

    manager = IntelligentParallelAgentManager(
        llm=llm,
        name="Conversational Assistant Manager",
    )

    orchestrator = IntelligentParallelOrchestrator(
        name="Conversational AI Assistant",
        manager=manager,
        agents=[
            research_agent,
            technical_agent,
            creative_agent,
            analytical_agent,
        ],
        tools=[search_tool, research_tool, scraper_tool, code_tool],
        max_concurrency=2,
        task_timeout=600,
        enable_complexity_analysis=True,
        enable_context_sharing=True,
        conversation_mode=True,
        orchestration_threshold=0.7,
    )

    return orchestrator


def demonstrate_conversation_flow():
    """Demonstrate the natural conversation flow with progressive complexity."""
    print("=== Conversational AI Assistant Demo ===\n")

    assistant = create_conversational_assistant()

    print("This demo shows how the assistant handles different types of interactions:")
    print("1. Simple conversation and questions")
    print("2. Requests needing clarification")
    print("3. Complex tasks requiring orchestration")
    print("4. Context building across multiple turns")
    print()

    conversation_scenarios = [
        {"input": "Hello! How are you doing today?", "expected": "Simple conversational response", "type": "greeting"},
        {
            "input": "What's the weather like in San Francisco?",
            "expected": "May escalate to using search tools",
            "type": "information",
        },
        {"input": "I need help with my project", "expected": "Should ask for clarification", "type": "clarification"},
        {
            "input": (
                "I'm planning a startup in the renewable energy space. "
                "Can you help me research the market, analyze competitors, "
                "create a business plan outline, and suggest potential funding sources?"
            ),
            "expected": "Should trigger parallel orchestration",
            "type": "complex_task",
        },
        {
            "input": "What about the regulatory landscape for my renewable energy startup?",
            "expected": "Should build on previous context",
            "type": "context_followup",
        },
    ]

    for i, scenario in enumerate(conversation_scenarios, 1):
        print(f"--- Scenario {i}: {scenario['type'].replace('_', ' ').title()} ---")
        print(f"User: {scenario['input']}")
        print(f"Expected: {scenario['expected']}")
        print()

        try:
            result = assistant.run(
                input_data={"input": scenario["input"]},
                config=None,
            )

            response = result.output.get("content", "No response generated")
            print(f"Assistant: {response[:300]}...")
            print()

        except Exception as e:
            print(f"Error: {e}")
            print()

        print("-" * 60)
        print()


def run_interactive_session():
    """Run an interactive session with the conversational assistant."""
    print("\n=== Interactive Session ===")
    print("Type 'quit' to exit the conversation")
    print()

    assistant = create_conversational_assistant()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Assistant: Goodbye! It was great chatting with you.")
                break

            if not user_input:
                continue

            print("Assistant: [Thinking...]")

            result = assistant.run(
                input_data={"input": user_input},
                config=None,
            )

            response = result.output.get("content", "I'm sorry, I couldn't generate a response.")
            print(f"Assistant: {response}")
            print()

        except KeyboardInterrupt:
            print("\nAssistant: Goodbye!")
            break
        except Exception as e:
            print(f"Assistant: I encountered an error: {e}")
            print()


def test_complexity_detection():
    """Test the adaptive complexity detection."""
    print("\n=== Complexity Detection Test ===\n")

    assistant = create_conversational_assistant()

    test_inputs = [
        "Hi there!",
        "What's 2+2?",
        "How are you?",
        "Tell me a joke",
        "What are the latest news about AI?",
        "Help me understand machine learning",
        "Calculate the ROI of this investment",
        "Research and analyze the electric vehicle market, "
        "create a competitive analysis, and suggest investment opportunities",
        "Plan a complete marketing strategy for "
        "a new SaaS product including market research, content creation, and campaign design",
        "Help me learn Python by creating a curriculum, finding resources, and building practice projects",
    ]

    for i, test_input in enumerate(test_inputs, 1):
        print(f"Test {i}: {test_input}")

        try:
            analysis = assistant.analyze_conversation(test_input)
            print(f"Decision: {analysis.decision.value}")
            print(f"Reasoning: {analysis.reasoning}")
            print(f"Confidence: {analysis.confidence}")

        except Exception as e:
            print(f"Error in analysis: {e}")

        print()


def demonstrate_context_building():
    """Demonstrate how context builds across conversation turns."""
    print("\n=== Context Building Demo ===\n")

    assistant = create_conversational_assistant()

    conversation_sequence = [
        "I'm working on a machine learning project",
        "It's focused on image classification",
        "Specifically, I want to classify medical images",
        "Can you help me find the right datasets and model architectures?",
        "What about data preprocessing techniques for medical images?",
        "And what are the ethical considerations I should be aware of?",
    ]

    print("This sequence shows how the assistant builds context over multiple turns:")
    print()

    for i, message in enumerate(conversation_sequence, 1):
        print(f"Turn {i}: {message}")

        try:
            result = assistant.run(
                input_data={"input": message},
                config=None,
            )

            response = result.output.get("content", "No response")
            print(f"Response: {response[:200]}...")
            print()

        except Exception as e:
            print(f"Error: {e}")
            print()


def test_conversation_mode_toggle():
    """Test toggling conversation mode on and off."""
    print("\n=== Conversation Mode Toggle Test ===\n")

    assistant = create_conversational_assistant()

    test_input = "Hello, how are you?"

    print("With conversation mode ENABLED:")
    assistant.conversation_mode = True
    try:
        result = assistant.run(input_data={"input": test_input}, config=None)
        print(f"Response: {result.output.get('content', 'No response')}")
    except Exception as e:
        print(f"Error: {e}")

    print()

    print("With conversation mode DISABLED:")
    assistant.conversation_mode = False
    try:
        result = assistant.run(input_data={"input": test_input}, config=None)
        print(f"Response: {result.output.get('content', 'No response')}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=== Unified Conversational Parallel Orchestrator Demo ===")
    print("Demonstrating adaptive conversation that scales to complex task orchestration\n")

    demonstrate_conversation_flow()

    # test_complexity_detection()
    # demonstrate_context_building()
    # test_conversation_mode_toggle()
    # run_interactive_session()

    print("\n" + "=" * 60)
    print("Demo completed! Key features demonstrated:")
    print("✓ Natural conversation handling")
    print("✓ Adaptive complexity detection")
    print("✓ Context building across turns")
    print("✓ Seamless escalation to task orchestration")
    print("✓ Progressive task refinement")
    print("✓ Unified orchestrator for all interaction types")
    print("\nThe IntelligentParallelOrchestrator seamlessly handles everything")
    print("from casual chat to complex multi-step task coordination!")
