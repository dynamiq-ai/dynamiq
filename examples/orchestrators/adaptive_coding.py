from dynamiq.connections import E2B as E2BConnection
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
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
    "Write the report about the weather in Warsaw for September 2024 and compare it with the latest three years."
    "Provide the results in a clear table."
    "Also compare it with San Francisco."
    "Firstly, try to search available free APIs."
)

INPUT_TASK = (
    "Write code in Python that fits linear regression model between 4 features"
    "(number of rooms, size of a house, etc) and price of a house from the data."
    "Count loss function and optimize it using gradient descent just for 3 iterations."
    "Simulate data for 100 houses."
    "Write a report in markdown with code and results."
    "In results provide initial and optimized loss of model."
    "also include the equation of the model."
)

INPUT_TASK = (
    "Use code skills to gather data about NVIDIA and INTEL stocks prices for last 10 years"
    ",calculate average per year for each company and createa atable per me. "
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
        inference_mode=InferenceMode.XML,
    )

    agent_manager = AdaptiveAgentManager(
        llm=llm,
    )

    orchestrator = AdaptiveOrchestrator(
        name="Adaptive Orchestrator",
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
