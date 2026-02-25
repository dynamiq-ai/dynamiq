import agentops

from dynamiq.connections import Exa
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.env import get_env_var
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm


@agentops.track_agent(name="React Agent")
def get_react_agent():
    llm = setup_llm()
    connection_exa = Exa()
    tool_search = ExaTool(connection=connection_exa)
    agent = Agent(
        name="Agent",
        id="React Agent",
        llm=llm,
        tools=[tool_search],
        inference_mode=InferenceMode.FUNCTION_CALLING,
    )
    return agent


@agentops.track_agent(name="Simple Agent")
def get_simple_agent():
    llm = setup_llm()
    agent = Agent(
        name="Agent",
        id="Simple Agent",
        llm=llm,
        role="Agent, goal to provide information based on the user input",
    )
    return agent


if __name__ == "__main__":
    agentops.init(get_env_var("AGENTOPS_API_KEY"))

    agent = get_react_agent()
    result = agent.run(input_data={"input": "Who won Euro 2024?"})

    agent_simple = get_simple_agent()
    result_simple = agent_simple.run(input_data={"input": "What is the capital of France?"})

    output_content = result.output.get("content")
    output_content_simple = result_simple.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
    logger.info(output_content_simple)
    agentops.end_session("Success")
