from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import ErrorHandling
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.runnables import RunnableConfig

INPUT_PROMPT = "What is the capital of France and why is it important?"


def create_openai_llm_with_error_handling() -> OpenAI:
    """
    Create OpenAI LLM with error handling configuration.

    Returns:
        OpenAI: Configured OpenAI LLM instance with error handling
    """
    connection = OpenAIConnection()

    error_handling = ErrorHandling(
        timeout_seconds=30.0,
        retry_interval_seconds=2.0,
        max_retries=3,
        backoff_rate=2.0,
        behavior=Behavior.RAISE
    )

    return OpenAI(
        connection=connection,
        model="gpt-4o-mpini",
        max_tokens=1000,
        temperature=0.7,
        error_handling=error_handling
    )


def create_react_agent_no_tools() -> Agent:
    """
    Create a ReAct agent without tools, using only OpenAI LLM with error handling.

    Returns:
        Agent: Configured ReAct agent without tools
    """
    llm = create_openai_llm_with_error_handling()

    agent_error_handling = ErrorHandling(
        timeout_seconds=60.0,
        retry_interval_seconds=3.0,
        max_retries=2,
        backoff_rate=1.5,
        behavior=Behavior.RAISE
    )

    return Agent(
        name="React Agent No Tools",
        id="react_no_tools",
        llm=llm,
        tools=[],
        role="You are a helpful AI assistant that provides "
        "thoughtful and accurate responses to user questions. "
        "You think step by step and provide clear, concise answers.",
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=3,
        verbose=True,
        error_handling=agent_error_handling
    )


def run_agent_workflow(input_prompt: str = INPUT_PROMPT) -> tuple[str, bool]:
    """
    Execute a workflow using the ReAct agent without tools.

    Args:
        input_prompt (str): The input question/prompt for the agent

    Returns:
        tuple[str, bool]: The agent response and success status
    """
    agent = create_react_agent_no_tools()
    workflow = Workflow(flow=Flow(nodes=[agent]))

    try:
        print(f"Running agent with input: {input_prompt}")

        config = RunnableConfig(request_timeout=120)

        result = workflow.run(
            input_data={"input": input_prompt},
            config=config
        )

        if result and result.output and agent.id in result.output:
            agent_output = result.output[agent.id]["output"]["content"]
            print(f"Agent response: {agent_output}")
            return agent_output, True
        else:
            print("No valid output received from agent")
            return "No response generated", False

    except Exception as e:
        print(f"Error during agent execution: {type(e).__name__}: {e}")
        return f"Error: {str(e)}", False


def main():
    """
    Main function to demonstrate the React agent without tools.
    """
    print("=== React Agent Without Tools - Demo ===")
    print("This agent uses OpenAI LLM with comprehensive error handling.\n")

    test_queries = [
        "What is the capital of France and why is it important?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        response, success = run_agent_workflow(query)

        if success:
            print("✅ Success")
        else:
            print("❌ Failed")

        print("-" * 50)


if __name__ == "__main__":
    main()
