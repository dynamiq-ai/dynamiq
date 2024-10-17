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

AGENT_ROLE = """
Expert Agent with high programming skills, he can solve any problem using coding skills.
Goal is to provide the best solution for request, using all his algorithmic knowledge and coding skills
"""  # noqa: E501
# simple coding tasks
INPUT_TASK = """
Write code in Python that fits linear regression model between 4 features (number of rooms, size of a house, etc) and price of a house from the data.
                Count loss function.
                Simulate data for 100 houses.
                Provide report in markdown
                In results provide initial and optimized loss of model.
                also include the equation of the model.
"""  # noqa: E501
INPUT_TASK = "Add the first 10 numbers and tell if the result is prime"


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
