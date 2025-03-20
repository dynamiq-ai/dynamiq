from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Exa, ZenRows
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

INPUT_TASK = (
    "Let's find data on optimizing "
    "SEO campaigns in 2025, analyze it, "
    "and provide predictions with calculations "
    "on how to improve and implement these strategies."
)


if __name__ == "__main__":
    python_tool = E2BInterpreterTool(
        name="Code Executor",
        connection=E2BConnection(),
    )

    zenrows_tool = ZenRowsTool(
        connection=ZenRows(),
        name="Web Scraper",
    )

    exa_tool = ExaTool(
        connection=Exa(),
        name="Search Engine",
    )

    llm = setup_llm(model_provider="gpt", model_name="o3-mini", max_tokens=100000)

    agent_coding = ReActAgent(
        name="Coding Agent",
        llm=llm,
        tools=[python_tool],
        max_loops=13,
        inference_mode=InferenceMode.XML,
    )

    agent_web = ReActAgent(
        name="Web Agent",
        llm=llm,
        tools=[zenrows_tool, exa_tool],
        max_loops=13,
        inference_mode=InferenceMode.XML,
    )

    agent_reflection = SimpleAgent(
        name="Reflection Agent (Reviewer, Critic)",
        llm=llm,
        role=(
            "Analyze and review the accuracy of any results, "
            "including tasks, code, or data. "
            "Offer feedback and suggestions for improvement."
        ),
    )

    agent_manager = AdaptiveAgentManager(
        llm=llm,
    )

    orchestrator = AdaptiveOrchestrator(
        name="Adaptive Orchestrator",
        agents=[agent_coding, agent_web, agent_reflection],
        manager=agent_manager,
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
