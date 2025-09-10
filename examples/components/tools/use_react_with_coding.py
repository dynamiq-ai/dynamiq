from dynamiq.connections import E2B
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from examples.llm_setup import setup_llm

AGENT_ROLE = """
Senior Data Scientist and Programmer with ability to write a well written python code inside
<code> tags and you have access to python tool.
Goal is to provide well explained answer to requests
"""  # noqa: E501
if __name__ == "__main__":
    tool = E2BInterpreterTool(
        connection=E2B(),
    )

    llm = setup_llm()

    # Create the agent with tools and configuration
    agent = Agent(name="React Agent", llm=llm, tools=[tool], role=AGENT_ROLE)

    result_dice_game = agent.run(
        input_data={
            "input": "Add the first 10 numbers and tell if the result is prime, use functions",
        },
        config=None,
    )

    result_fibonacci = agent.run(
        input_data={
            "input": "generate the first 100 numbers in the Fibonacci sequence, and print me each odd number",
        },
        config=None,
    )
    result_scrape = agent.run(
        input_data={
            "input": "scrape the example com website and print information",
        },
        config=None,
    )
    result_data_analysis = agent.run(
        input_data={
            "input": "generate a sample for linear regression analysis and print the results, just textual representations",  # noqa: E501
        },
        config=None,
    )
    print("\nResult with dice game:")
    print(result_dice_game.output.get("content"))
    print("\nResult with Fibonacci sequence:")
    print(result_fibonacci.output.get("content"))
    print("\nResult with web scraping:")
    print(result_scrape.output.get("content"))
    print("\nResult with data analysis:")
    print(result_data_analysis.output.get("content"))
