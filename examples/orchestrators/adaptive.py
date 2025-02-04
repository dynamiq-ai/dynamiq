from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.connections import ZenRows as ZenRowsConnection
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

INPUT_TASK = "I'm in Warsaw. Can you suggest the best cinemas I can visit today?"

INPUT_TASK = (
    "Let's gather some data about NVIDIA stocks and use Python to analyze it."
    "We'll focus on the last month and build a predictive model to determine "
    "whether I should invest or not, considering the latest news as well."
)


if __name__ == "__main__":
    tavily_tool = TavilyTool(
        connection=TavilyConnection(),
    )
    zenrows_tool = ZenRowsTool(
        connection=ZenRowsConnection(),
    )

    python_tool = E2BInterpreterTool(
        connection=E2BConnection(),
    )

    llm = setup_llm()

    agent_coding = ReActAgent(
        name="Coding Agent",
        llm=llm,
        tools=[python_tool],
        max_loops=7,
        inference_mode=InferenceMode.XML,
    )

    agent_searcher = ReActAgent(
        name="Searcher Agent",
        llm=llm,
        tools=[tavily_tool, zenrows_tool],
        max_loops=7,
        inference_mode=InferenceMode.XML,
    )

    agent_manager = AdaptiveAgentManager(
        llm=llm,
    )

    orchestrator = AdaptiveOrchestrator(
        name="Adaptive Orchestrator",
        agents=[agent_coding, agent_searcher],
        manager=agent_manager,
        reflection_enabled=True,
    )

    result = orchestrator.run(
        input_data={
            "input": INPUT_TASK,
        },
        config=None,
    )

    output_content = result.output.get("content")
    print("RESULT")
    print(output_content)
