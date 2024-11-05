from dynamiq.connections import Tavily
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools import TavilyTool
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm


def run_workflow():

    tavily_connections = Tavily()
    tavily_tool = TavilyTool(connection=tavily_connections)
    llm = setup_llm()
    search_assistant = ReActAgent(
        name="Search assistant", llm=llm, role=("You are a helpful search assistant"), tools=[tavily_tool]
    )

    agent_manager = GraphAgentManager(llm=llm)

    graph_orchestrator = GraphOrchestrator(manager=agent_manager, final_summarizer=True)

    graph_orchestrator.add_node("search", [search_assistant])
    graph_orchestrator.add_edge(START, "search")
    graph_orchestrator.add_edge("search", END)

    return graph_orchestrator.run(input_data={"input": "Hello"}).output["content"]


if __name__ == "__main__":
    result = run_workflow()
    logger.info(result)
