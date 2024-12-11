from dynamiq.connections import E2B as E2BConnection
from dynamiq.nodes.agents.orchestrators.linear import LinearOrchestrator
from dynamiq.nodes.agents.orchestrators.linear_manager import LinearAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

AGENT_ROLE = (
    "An Expert Agent with high programming skills, he can solve any problem using coding skills."
    "Goal is to provide the best solution for request,"
    "using all his algorithmic knowledge and coding skills"
)

INPUT_TASK = (
    "Use code skills to gather data about NVIDIA and INTEL stocks prices for last 10 years"
    ",calculate average per year for each company and create a table per me. "
    "Then craft a report and ad conclusion,"
    " what would be better if I could invest 100$ 10 yeasr ago. Use yahoo finance."
)

if __name__ == "__main__":
    python_tool = E2BInterpreterTool(
        connection=E2BConnection(),
    )
    llm = setup_llm()

    agent_coding = ReActAgent(
        name="Coding Agent",
        llm=llm,
        tools=[python_tool],
        role=AGENT_ROLE,
        max_loops=5,
        inference_mode=InferenceMode.DEFAULT,
    )

    agent_manager = LinearAgentManager(
        llm=llm,
    )

    orchestrator = LinearOrchestrator(
        name="Linear Orchestrator",
        agents=[agent_coding],
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
