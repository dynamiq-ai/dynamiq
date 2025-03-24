from dynamiq.connections import Exa
from dynamiq.nodes.agents.agent import Agent
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

llm = setup_llm()
connection_exa = Exa()
tool_search = ExaTool(connection=connection_exa)
agent = Agent(
    name=" Agent",
    llm=llm,
    id="agent",
    role="use emojiss , explain as for children",
    verbose=True,
)
res = agent.run(input_data={"input": "What is the capital of France?"})

print(res)

agent_search = Agent(
    name="Agent",
    id="Agent",
    llm=llm,
    tools=[tool_search],
)
result = agent_search.run(input_data={"input": "Search for the best restaurants in New York"})


print(result)

agent_search_xml = Agent(
    name="Agent",
    id="Agent",
    llm=llm,
    tools=[tool_search],
    inference_mode=InferenceMode.XML,
)
result_xml = agent_search_xml.run(input_data={"input": "Search for the best restaurants in New York"})


print(result_xml)
