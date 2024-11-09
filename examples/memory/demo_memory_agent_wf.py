import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.memory import Memory
from dynamiq.memory.backend.in_memory import InMemory
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

# Constants
AGENT_ROLE = "professional writer, goal is to produce a well-written and informative response"
INPUT_QUESTION = "What is the capital of France?"


def run_simple_workflow() -> tuple[str, dict]:
    """
    Execute a workflow using the OpenAI agent to process a predefined question.

    Returns:
        tuple[str, dict]: The generated content by the agent and the trace logs.

    Raises:
        Exception: Captures and prints any errors during workflow execution.
    """
    llm = setup_llm()
    memory = Memory(backend=InMemory())
    memory.add(
        MessageRole.USER, "Hey! I'm Oleksii, machine learning engineer from Dynamiq.", metadata={"user_id": "01"}
    )
    agent = SimpleAgent(
        name=" Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
        memory=memory,
    )
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": INPUT_QUESTION},
            config=RunnableConfig(callbacks=[tracing]),
        )

        # Ensure trace logs can be serialized to JSON
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    output, traces = run_simple_workflow()
    print(output)
