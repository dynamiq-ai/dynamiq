import inspect
import json
import textwrap

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import connections
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.callbacks import NodeCallbackHandler
from typing import Any
from dynamiq.prompts import Message, Prompt, MessageRole


class HumanFeedbackHandler(NodeCallbackHandler):
    def on_node_end(self, serialized, output_data, **kwargs):
        context = output_data["context"]
        iteration_num = context.get("iteration", 1)
        if iteration_num < 1:
            context = output_data["context"]
            context |= {"feedback": "Feedback", "iteration": iteration_num + 1}
            return context
        
        context |= {"feedback": None, "iteration": iteration_num}


def create_orchestrator(model: str, connection: connections.OpenAI, context_input: dict) -> GraphOrchestrator:
    """
    Creates orchestrator

    Returns:
        GraphOrchestrator: The configured orchestrator.
    """
    llm = OpenAI(
        name="OpenAI",
        model=model,
        connection=connection,
        temperature=0.1,
    )


    def generate_sketch(context: dict[str, Any]):
        "Generate sketch"
        messages = context.get("messages")

        response = llm.run(
            input_data={},
            prompt=Prompt(
                messages=messages,
            ),
        ).output["content"]

        context["messages"] += [
            {
                "role": MessageRole.ASSISTANT,
                "content": f"mocked_response",
                'metadata': None
            }
        ]

        return {"result": response, **context}

    def accept_sketch(context: dict[str, Any]):
        if context.get("feedback"):
            return "generate_sketch"

        return END

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=GraphAgentManager(llm=llm),
        context = context_input
    )

    orchestrator.add_state_by_tasks("generate_sketch", [generate_sketch], callbacks=[HumanFeedbackHandler()])

    orchestrator.add_edge(START, "generate_sketch")
    orchestrator.add_conditional_edge("generate_sketch", ["generate_sketch", END], accept_sketch)


    wf_orchestrator = Workflow(
        flow=Flow(
            nodes=[orchestrator],
        ),
    )

    return wf_orchestrator


@pytest.mark.parametrize(
    ("get_orchestrator_workflow", "context_input", "outputs", "context_output"),
    [
        (
            create_orchestrator,
            {"iteration": 0, "messages": [Message(role="user", content=f"Answer on question")]},
            {"content": "mocked_response"},
            {"iteration": 1, 'feedback': None, 'messages': [
                {'content': 'Answer on question', 'metadata': None, 'role': MessageRole.USER},
                {'content': 'mocked_response',  'role': MessageRole.ASSISTANT, 'metadata': None},
                {'content': 'mocked_response',  'role': MessageRole.ASSISTANT, 'metadata': None}
                ]},
            ),
        (
            create_orchestrator,
            {"iteration": 1, "messages": [Message(role="user", content=f"Answer on question")]},
            {"content": "mocked_response"},
            {"iteration": 1, 'feedback': None, 'messages': [
                {'content': 'Answer on question', 'role': MessageRole.USER, 'metadata': None,},
                {'content': 'mocked_response',  'role': MessageRole.ASSISTANT, 'metadata': None},
                ]},
            ),
        
    ],
)

def test_workflow_with_map_node(get_orchestrator_workflow, context_input, outputs, context_output):
    model = "gpt-3.5-turbo"
    connection = connections.OpenAI(
        api_key="api_key",
    )

    wf_orchestrator = get_orchestrator_workflow(model, connection, context_input)
    input_data = {"input": ""}
    tracing = TracingCallbackHandler()
    response = wf_orchestrator.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=outputs,
    ).to_dict()

    expected_output = {wf_orchestrator.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
    assert json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)

    assert wf_orchestrator.flow.nodes[0].context == context_output
