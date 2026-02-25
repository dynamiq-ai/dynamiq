from dynamiq.connections import E2B, Exa
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = (
    "You are a versatile AI assistant:\n"
    "  1. Plan your approach in a clear reasoning step.\n"
    "  2. Search  for any documentation or data you need.\n"
    "  3. Code: write a snippet that solves the user’s request, use code interpreter\n"
    "  4. Execute & Validate:\n"
    "     - Run your code in the sandbox.\n"
    "     - Inspect stdout/stderr and the data structures returned.\n"
    "     - If there are errors or unexpected results, fix them.\n"
    "     - Optionally add simple assertions or checks to prove correctness.\n"
    "     - Repeat until the code runs cleanly and meets the spec.\n"
    "  5. Report: Return a concise Markdown‐formatted answer or code block,\n"
    "     including any tables or plots if relevant.\n\n"
    "Always think in terms of tool use: plan → search → code → test → refine → answer."
)


if __name__ == "__main__":
    conn_code = E2B()
    conn_search = Exa()

    tool_code = E2BInterpreterTool(connection=conn_code)
    tool_search = ExaTool(connection=conn_search)

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0.2)

    agent = Agent(
        name="MultiToolAgent",
        id="MultiToolAgent",
        llm=llm,
        tools=[tool_search, tool_code],
        role=AGENT_ROLE,
        parallel_tool_calls_enabled=True,
    )

    example_input = {
        "input": (
            "Find the official documentation for the OpenWeatherMap API."
            "Generate a Python snippet that fetches the 7-day forecast for Warsaw, Poland,"
            "parses the JSON, and prints a table of date vs. temperature."
        ),
        "files": None,
    }
    result = agent.run(input_data=example_input)
    logger.info("=== AGENT OUTPUT ===")
    logger.info(result.output.get("content"))
