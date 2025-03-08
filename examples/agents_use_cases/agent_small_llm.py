from dynamiq.connections import Exa
from dynamiq.connections import TogetherAI as TogetherAIConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms.togetherai import TogetherAI
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger

if __name__ == "__main__":
    connection_exa = Exa()
    tool_search = ExaTool(connection=connection_exa)
    llm = TogetherAI(
        connection=TogetherAIConnection(),
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0,
        max_tokens=4000,
    )
    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search],
        inference_mode=InferenceMode.XML,
    )
    result = agent.run(input_data={"input": "Provide me latest paper on LLM for last week."})
    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
