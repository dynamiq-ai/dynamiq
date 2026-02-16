from dynamiq.connections import Exa
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools import ThinkingTool
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

if __name__ == "__main__":
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0.7)

    thinking_tool = ThinkingTool(llm=llm)
    connection_exa = Exa()
    tool_search = ExaTool(connection=connection_exa)

    agent = Agent(
        name="Thinking Agent",
        id="thinking_agent",
        llm=llm,
        tools=[thinking_tool, tool_search],
        inference_mode=InferenceMode.FUNCTION_CALLING,
    )

    result = agent.run(
        input_data={
            "input": "Please create a report on investment opportunities for 2025, "
            "including the best stocks, cryptocurrencies, and other assets."
        }
    )

    output_content = result.output.get("content")
    print("Agent response:", output_content)
