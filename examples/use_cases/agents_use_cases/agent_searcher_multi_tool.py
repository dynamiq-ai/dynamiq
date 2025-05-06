from dynamiq.connections import Exa, Tavily
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

if __name__ == "__main__":
    connection_exa = Exa()
    connection_tavily = Tavily()
    tool_tavily = TavilyTool(connection=connection_tavily)
    tool_search = ExaTool(connection=connection_exa)
    llm = setup_llm(model_provider="gpt", model_name="o3-mini", temperature=1)
    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search, tool_tavily],
        inference_mode=InferenceMode.DEFAULT,
        role="",
    )
    result = agent.run(
        input_data={
            "input": "Please research Dynamiq AI and provide well-structured information"
            "along with a summary in markdown format. "
            "Include use cases, competitors, potential, future outlook, and recent news"
            # "USE parallel multi tool calling"
        }
    )
    output_content = result.output.get("content")
    print("Agent response:", output_content)
