from dynamiq.connections import ScaleSerp
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

if __name__ == "__main__":
    connection_serp = ScaleSerp()
    tool_search = ScaleSerpTool(connection=connection_serp)
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=1)
    tools = [tool_search]
    agent_def = Agent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=tools,
        inference_mode=InferenceMode.DEFAULT,
    )
    agent_xml = Agent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=tools,
        inference_mode=InferenceMode.XML,
    )
    agent_fc = Agent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=tools,
        inference_mode=InferenceMode.FUNCTION_CALLING,
    )
    agent_so = Agent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=tools,
        inference_mode=InferenceMode.STRUCTURED_OUTPUT,
    )
    """
    result_def = agent_def.run(input_data={"input": "Search for the best restaurants in New York"})
    output_content_def = result_def.output.get("content")
    print("Agent response def :", output_content_def)
    result_xml = agent_xml.run(input_data={"input": "Search for the best restaurants in New York"})
    output_content_xml = result_xml.output.get("content")
    print("Agent response xml :", output_content_xml)
    """
    result_fc = agent_fc.run(input_data={"input": "Search for the best restaurants in New York"})
    output_content_fc = result_fc.output.get("content")
    print("Agent response fc :", output_content_fc)
    result_so = agent_so.run(input_data={"input": "Search for the best restaurants in New York"})
    output_content_so = result_so.output.get("content")
    print("Agent response so :", output_content_so)
