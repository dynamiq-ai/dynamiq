import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.reflection import ReflectionAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.prompts import (
    MessageRole,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

# Constants
AGENT_NAME = "Art Agent"
AGENT_ROLE = "Professional writer with the goal of producing well-written and informative responses about art"
INPUT_QUESTION = "Describe main idea of this piece of art."
IMAGE_URL = "IMAGE_URL"


def run_reflection_agent_workflow() -> tuple[str, dict]:
    """
    Set up and run a workflow using a ReflexionAgent with OpenAI's language model.

    Returns:
        str: The output content generated by the agent, or an empty string if an error occurs.

    Raises:
        Exception: Any exception that occurs during the workflow execution is caught and printed.
    """
    # Set up OpenAI connection and language model
    llm = setup_llm()
    agent = ReflectionAgent(
        name=AGENT_NAME,
        llm=llm,
        id="agent",
        verbose=True,
        context="You are helpful assistant that answers on question about art."
        "Take into account style of response: {{context}}",
        input_message=VisionMessage(
            content=[
                VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{ url }}")),
                VisionMessageTextContent(text="{{ request }}"),
            ],
            role=MessageRole.USER,
        ),
    )

    # Set up tracing and create the workflow
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    # Run the workflow and handle the result
    try:
        result = wf.run(
            input_data={
                "request": INPUT_QUESTION,
                "url": IMAGE_URL,
                "context": "Keep answer short and simple",
            },
            config=RunnableConfig(callbacks=[tracing]),
        )

        # Verify that traces can be serialized to JSON
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


def run_react_agent_workflow() -> tuple[str, dict]:
    """
    Set up and run a workflow using a ReActAgent with OpenAI's language model.

    Returns:
        str: The output content generated by the agent, or an empty string if an error occurs.

    Raises:
        Exception: Any exception that occurs during the workflow execution is caught and printed.
    """
    # Set up OpenAI connection and language model
    llm = setup_llm()
    agent = ReActAgent(
        name=AGENT_NAME,
        llm=llm,
        id="agent",
        verbose=True,
        context="You are helpful assistant that answers on question about art."
        "Take into account style of response: {{context}}",
        input_message=VisionMessage(
            content=[
                VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{ url }}")),
                VisionMessageTextContent(text="{{ request }}"),
            ],
            role=MessageRole.USER,
        ),
    )

    # Set up tracing and create the workflow
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    # Run the workflow and handle the result
    try:
        result = wf.run(
            input_data={
                "request": INPUT_QUESTION,
                "url": IMAGE_URL,
                "context": "Keep answer short and simple",
            },
            config=RunnableConfig(callbacks=[tracing]),
        )

        # Verify that traces can be serialized to JSON
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


def run_simple_agent_workflow() -> tuple[str, dict]:
    """
    Set up and run a workflow using a SimpleAgent with OpenAI's language model.

    Returns:
        str: The output content generated by the agent, or an empty string if an error occurs.

    Raises:
        Exception: Any exception that occurs during the workflow execution is caught and printed.
    """
    # Set up OpenAI connection and language model
    llm = setup_llm()
    agent = SimpleAgent(
        name=AGENT_NAME,
        llm=llm,
        id="agent",
        verbose=True,
        context="You are helpful assistant that answers on question about art."
        "Take into account style of response: {{context}}",
        input_message=VisionMessage(
            content=[
                VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{ url }}")),
                VisionMessageTextContent(text="{{ request }}"),
            ],
            role=MessageRole.USER,
        ),
    )

    # Set up tracing and create the workflow
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    # Run the workflow and handle the result
    try:
        result = wf.run(
            input_data={
                "request": INPUT_QUESTION,
                "url": IMAGE_URL,
                "context": "Keep answer short and simple",
            },
            config=RunnableConfig(callbacks=[tracing]),
        )

        # Verify that traces can be serialized to JSON
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    output, _ = run_reflection_agent_workflow()
    print(output)

    output, _ = run_react_agent_workflow()
    print(output)

    output, _ = run_simple_agent_workflow()
    print(output)
