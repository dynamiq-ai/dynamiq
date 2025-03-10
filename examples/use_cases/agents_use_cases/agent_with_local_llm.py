from dynamiq.connections import Exa, Ollama
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms import Ollama as OllamaLLM
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger

if __name__ == "__main__":
    connection_exa = Exa()
    tool_search = ExaTool(connection=connection_exa)
    ollama_endpoint = "http://localhost:11434"

    llm = OllamaLLM(
        model="qwen2.5-coder:32b",
        connection=Ollama(url=ollama_endpoint),
        temperature=0.1,
        max_tokens=1000,
    )
    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search],
        inference_mode=InferenceMode.XML,
    )
    result = agent.run(input_data={"input": "Who won Euro 2024?"})
    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
