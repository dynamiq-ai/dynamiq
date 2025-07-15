from dynamiq.connections import ScaleSerp
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.function_tool import function_tool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from examples.llm_setup import setup_llm

AGENT_ROLE = (
    "professional writer, goal is to produce"
    "a well written and informative response, that can be used for CHILDREN, "
    "with emojis and simple language."
)

if __name__ == "__main__":
    llm = setup_llm()

    @function_tool
    def calculate_age(input_age: int, current_year: int, **kwargs) -> int:
        """
        Calculate a person's age based on their birth year.

        Args:
            input_age (int): The year the person was born.
            current_year (int): The current year.

        Returns:
            age (int): The person's age.
        """
        age = int(current_year) - int(input_age)
        return age

    calculate_age_tool = calculate_age()
    serp_connection = ScaleSerp()
    tool_search = ScaleSerpTool(connection=serp_connection)

    agent = ReActAgent(
        name="Agent",
        id="agent",
        role=AGENT_ROLE,
        llm=llm,
        tools=[calculate_age_tool, tool_search],
    )

    result = agent.run(
        input_data={
            "input": "I was born in 2000, and now is 2024. Your task is to calculate my age and then find the top films, limiting results to number of my age"  # noqa: E501
        }
    )

    print(result.output.get("content"))
