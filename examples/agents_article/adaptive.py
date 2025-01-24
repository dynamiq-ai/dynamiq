from dynamiq.connections import E2B
from dynamiq.connections import DeepSeek as DeepSeekConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Tavily
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.llms.deepseek import DeepSeek
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.tavily import TavilyTool

INPUT = """I would like to sell items on Etsy and need an analysis
to determine if this is a good time to start or if it's too late.
What can I focus on for summer 2025? What products should I consider selling and why?
Additionally, how can I maximize my profits? Please provide a brief report with sources and recommendations.
"""

if __name__ == "__main__":
    connection_e2b = E2B()
    tool_code = E2BInterpreterTool(connection=connection_e2b)
    connection_tavily = Tavily()
    tool_tavily = TavilyTool(connection=connection_tavily)
    connection_openai = OpenAIConnection()
    connection_deepseek = DeepSeekConnection()

    llm_deepseek = DeepSeek(
        connection=connection_deepseek, model="deepseek-chat", max_tokens=8000, stop=["Observation"]
    )

    llm_deepseek_reasoner = DeepSeek(
        connection=connection_deepseek, model="deepseek-reasoner", max_tokens=8000, stop=["Observation"]
    )

    llm_openai = OpenAI(connection=connection_openai, model="gpt-4o", max_tokens=8000, stop=["Observation"])

    llm_openai_reasoner = OpenAI(connection=connection_openai, model="o1-mini", max_tokens=8000, stop=["Observation"])

    agent = ReActAgent(
        name="Agent",
        llm=llm_openai,
        tools=[tool_code, tool_tavily],
        inference_mode=InferenceMode.DEFAULT,
    )

    agent_checker = ReActAgent(
        name="Checker and Validator Agent",
        llm=llm_openai_reasoner,
        tools=[tool_tavily],
        inference_mode=InferenceMode.DEFAULT,
    )

    agent_manager = AdaptiveAgentManager(
        llm=llm_openai_reasoner,
    )

    orchestrator = AdaptiveOrchestrator(
        name="Adaptive Orchestrator",
        agents=[agent, agent_checker],
        manager=agent_manager,
    )

    result = orchestrator.run(
        input_data={
            "input": INPUT,
        },
        config=None,
    )

    output_content = result.output.get("content")
    print("RESULT")
    print(output_content)
