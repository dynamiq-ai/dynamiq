import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

# Constants
AGENT_ROLE = "professional writer,goal is to produce a well-written and informative response"
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
    agent = SimpleAgent(
        name=" Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
        verbose=True,
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


def run_simple_custom_workflow() -> tuple[str, dict]:
    """
    Execute a workflow using the OpenAI agent to process a predefined question.

    Returns:
        tuple[str, dict]: The generated content by the agent and the trace logs.

    Raises:
        Exception: Captures and prints any errors during workflow execution.
    """
    llm = setup_llm()
    agent_custom = Agent(
        name="Agent - Custom",
        role=AGENT_ROLE,
        id="agent_custom",
        llm=llm,
    )
    agent_custom.add_block(
        "instructions",
        """
                           Use markdown as a main engine.
                            Use the following structure:
                            - Title
                            - Introduction
                            - Chapter 1
                            ...
                            - Conclusion
                            - References
                           """,
    )
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent_custom]))

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

        return result.output[agent_custom.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":

    output, traces = run_simple_workflow()
    print(output)

    output, traces = run_simple_custom_workflow()
    print(output)
