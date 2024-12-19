import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

AGENT_ROLE = (
    "An Expert Agent with advanced programming skills who can solve any problem using coding expertise. "
    "The goal is to provide the best solution for requests, utilizing all algorithmic knowledge and coding skills."
)

INPUT_TASK = (
    "Write a report about the weather in "
    "Warsaw for September 2024 and compare "
    "it with the data from the last three years. "
    "Provide the results in a clear table "
    "and also compare them with the weather in San Francisco. "
    "Start by searching for available free APIs."
)

INPUT_TASK = (
    "Write Python code to fit a linear regression model between four features "
    "(e.g., number of rooms, size of a house) and the price of a house from the data. "
    "Calculate the loss function and optimize it using gradient descent for just three iterations. "
    "Simulate data for 100 houses. "
    "Write a markdown report that includes the code and results. "
    "In the results, provide the initial and optimized loss of the model, as well as the equation of the model."
)

INPUT_TASK = (
    "Use programming skills to gather data on "
    "NVIDIA and INTEL stock prices over the last 10 years. "
    "Calculate the average price per year for each company and create a table. "
    "Then, craft a report and add a conclusion "
    "about what would have been the better investment "
    "if $100 had been invested 10 years ago. "
    "Use Yahoo Finance for the data."
)


def run_coding_task():
    tool_python_connection = E2B()
    tool_python = E2BInterpreterTool(connection=tool_python_connection)
    llm = setup_llm()

    agent_coding = ReActAgent(
        name="Coding Agent",
        llm=llm,
        tools=[tool_python],
        role=AGENT_ROLE,
        max_loops=15,
    )

    agent_manager = AdaptiveAgentManager(
        llm=llm,
    )

    orchestrator = AdaptiveOrchestrator(
        name="Adaptive Orchestrator",
        agents=[agent_coding],
        manager=agent_manager,
    )

    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[orchestrator]))

    try:
        result = wf.run(
            input_data={"input": INPUT_TASK},
            config=RunnableConfig(callbacks=[tracing]),
        )
        # Verify that traces can be serialized to JSON
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )
        content = result.output[orchestrator.id]["output"]["content"]
        print("Result with coding task:")
        print(content)
        return content, tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    run_coding_task()
