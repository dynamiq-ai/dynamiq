from dynamiq.connections import Exa
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

if __name__ == "__main__":
    connection_exa = Exa()
    tool_search = ExaTool(connection=connection_exa)
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=1)
    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search],
        inference_mode=InferenceMode.XML,
        role="You are an research assitant, "
        "you are good at deep research you can call multiple tool;with different query expansin and refining rpocess",
    )
    result = agent.run(input_data={"input": "research on dubai free zone company setup!Use multiple tool at once"})
    output_content = result.output.get("content")
    print("Agent response:", output_content)
