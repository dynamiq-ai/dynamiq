from dynamiq.connections import Exa
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

if __name__ == "__main__":
    connection_exa = Exa()
    tool_search = ExaTool(connection=connection_exa)
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=1)
    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search],
        inference_mode=InferenceMode.XML,
    )
    result = agent.run(input_data={"input": "Search for the best restaurants in New York"})
    output_content = result.output.get("content")
    print("Agent response:", output_content)
