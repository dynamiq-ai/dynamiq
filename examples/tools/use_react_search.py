"""
This script sets up and runs a question-answering system using the Dynamiq framework.
It uses a ReActAgent with a Tavily search tool and an Anthropic LLM model.
"""

import time
from typing import Any

from dynamiq.connections import ScaleSerp
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from examples.llm_setup import setup_llm

llm = setup_llm()

# Constants
AGENT_ROLE = "Dynamiq AI, an AI model who is expert at searching the web and answering user's queries."
AGENT_GOAL = """
Generate a response that is informative and relevant to the user's query.
You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself.
Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.
Sourced links should be placed at the end of the response. You should not use any other sources other than the search results provided in the context.
"""  # noqa: E501


def setup_agent() -> ReActAgent:
    """
    Set up and configure the ReActAgent with necessary tools and LLM.

    Returns:
        ReActAgent: Configured agent ready to process queries.
    """

    serp_connection = ScaleSerp()
    tool_search = ScaleSerpTool(connection=serp_connection)

    llm = setup_llm()

    # Create and return the agent
    return ReActAgent(
        name="React Agent",
        llm=llm,
        tools=[tool_search],
        role=AGENT_ROLE,
        goal=AGENT_GOAL,
    )


def process_query(agent: ReActAgent, query: str) -> dict[str, Any]:
    """
    Process a single query using the given agent.

    Args:
        agent (ReActAgent): The agent to use for processing the query.
        query (str): The user's query.

    Returns:
        Dict[str, Any]: The agent's response and the time taken to process the query.
    """
    start = time.time()
    try:
        response = agent.run(input_data={"input": query})
        content = response.output.get("content")
    except Exception as e:
        content = f"An error occurred: {e}"
    finally:
        end = time.time()

    return {"content": content, "time_taken": end - start}


def main():
    """
    Main function to run the question-answering system.

    Continuously prompts the user for queries, processes them using the agent,
    and prints the response along with the time taken.
    """
    agent = setup_agent()
    print("Question-Answering System initialized. Type 'quit' to exit.")

    while True:
        query = input("Enter your query: ")
        if query.lower() == "quit":
            print("Exiting the system. Goodbye!")
            break

        result = process_query(agent, query)
        print(result["content"])
        print(f"Time taken: {result['time_taken']:.2f} seconds")


if __name__ == "__main__":
    main()
