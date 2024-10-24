from dynamiq.connections import E2B
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = (
    "A helpful and general-purpose AI assistant with strong language, Python, "
    "and Linux command-line skills. The goal is to provide concise answers to the user. "
    "Additionally, try to generate code to solve tasks, then run it accurately. "
    "Before answering, create a plan for solving the task. You can search for any API, "
    "and use any free, open-source API that doesn't require authorization."
)

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
        inference_mode=InferenceMode.XML,
    )

    result = agent.run(input_data={"input": "What is the current price of Bitcoin?"})

    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
