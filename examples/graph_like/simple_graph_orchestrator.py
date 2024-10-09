from typing import Literal

from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

INPUT_TASK = "Create and execute tasks"  # noqa: E501

if __name__ == "__main__":

    llm = setup_llm()

    create_task_agent = ReActAgent(
        name="Task creation Agent",
        llm=llm,
        role="Create simple coding task.",
        goal="Create simple coding task taylored to request.",
        max_loops=15,
        inference_mode=InferenceMode.XML,
    )

    solve_task_agent = ReActAgent(
        name="Task solving Agent",
        llm=llm,
        role="Expert Agent with high programming skills, he can solve any problem using coding skills",
        goal="provide the best solution for request, using all his algorithmic knowledge and coding skills",
        max_loops=15,
        inference_mode=InferenceMode.XML,
    )

    agent_manager = GraphAgentManager(
        llm=llm,
    )

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=agent_manager,
    )

    orchestrator.add_node("create_task", [create_task_agent])
    orchestrator.add_node("solve_task", [solve_task_agent])

    orchestrator.add_edge(START, "create_task")
    orchestrator.add_edge("create_task", "solve_task")

    def switch_logic(context) -> Literal["create_task", END]:
        if "Error" in context:
            return "create_task"

        return END

    orchestrator.add_conditional_edge("solve_task", [END, "create_task"], switch_logic)

    result = orchestrator.run(
        input_data={
            "input": INPUT_TASK,
        },
        config=None,
    )

    print("Result:")
    print(result.output.get("content"))
