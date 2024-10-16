from dynamiq.connections import E2B
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = "A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills."  # noqa: E501
AGENT_GOAL = """is to provide concise answer to user,
              also try to generate code for solve task, then run it accurately
              before answering try to create plan for solving task
              you can search any api, and then use any of free open-source APi that dont require authorization
              """  # noqa: E501

if __name__ == "__main__":
    connection_e2b = E2B()

    tool_code = E2BInterpreterTool(connection=connection_e2b)
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)

    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_code],
        role=AGENT_ROLE,
        goal=AGENT_GOAL,
        inference_mode=InferenceMode.XML,
    )

    result = agent.run(input_data={"input": "What is the current price of Bitcoin?"})

    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
