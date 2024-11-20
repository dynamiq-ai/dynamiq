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


def get_orchestrator_workflow(model: str, connection: connections.OpenAI, context_input: dict):
    llm = OpenAI(
        name="OpenAI",
        model=model,
        connection=connection,
        temperature=0.1,
    )

    agent_manager = GraphAgentManager(llm=llm)
    graph_orchestrator = GraphOrchestrator(manager=agent_manager, final_summarizer=True, context=context_input)

    # Task 1
    def task1(context: dict):
        return {"result": "task 1 completed", "task1": "task 1 result"}

    # Task 2
    def run(context: dict):
        return {"result": "task 2 completed", "task2": "task 2 result"}

    task2 = Python(code=textwrap.dedent(inspect.getsource(run)))

    # Task 3
    def task3(context: dict):
        return {"result": "task 3 completed", "task3": "task 3 result"}

    # Condition
    def orchestrate(context: dict):
        return "task3" if context.get("task3", False) else END

    graph_orchestrator.add_node("task1", [task1])
    graph_orchestrator.add_node("task2", [task2])
    graph_orchestrator.add_edge(START, "task1")
    graph_orchestrator.add_edge("task1", "task2")

    graph_orchestrator.add_node("task3", [task3])
    graph_orchestrator.add_conditional_edge("task2", ["task3", END], orchestrate)

    graph_orchestrator.add_edge("task3", END)

    wf_orchestrator = Workflow(
        flow=Flow(
            nodes=[graph_orchestrator],
        ),
    )

    return wf_orchestrator


@pytest.mark.parametrize(
    ("context_input", "outputs", "context_output"),
    [
        (
            {},
            {"content": "mocked_response"},
            {"task1": "task 1 result", "task2": "task 2 result"},
        ),
        (
            {"task3": True},
            {"content": "mocked_response"},
            {"task1": "task 1 result", "task2": "task 2 result", "task3": "task 3 result"},
        ),
    ],
)
def test_workflow_with_map_node(context_input, outputs, context_output):
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
