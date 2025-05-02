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


def get_orchestrator_workflow1(model: str, connection: connections.OpenAI, context_input: dict):
    llm = OpenAI(
        name="OpenAI",
        model=model,
        connection=connection,
        temperature=0.1,
    )

    agent_manager = GraphAgentManager(llm=llm)
    graph_orchestrator = GraphOrchestrator(
        manager=agent_manager, final_summarizer=True, context=context_input, enable_handle_input=False
    )

    # Task 1
    def task1(context: dict, **kwargs):
        return {"result": "task 1 completed", "task1": "task 1 result"}

    # Task 2
    def run(context: dict, **kwargs):
        return {"result": "task 2 completed", "task2": "task 2 result"}

    task2 = Python(code=textwrap.dedent(inspect.getsource(run)))

    # Task 3
    def task3(context: dict, **kwargs):
        return {"result": "task 3 completed", "task3": "task 3 result"}

    # Condition
    def orchestrate(context: dict, **kwargs):
        return "task3" if context.get("task3", False) else END

    graph_orchestrator.add_state_by_tasks("task1", [task1])
    graph_orchestrator.add_state_by_tasks("task2", [task2])
    graph_orchestrator.add_edge(START, "task1")
    graph_orchestrator.add_edge("task1", "task2")

    graph_orchestrator.add_state_by_tasks("task3", [task3])
    graph_orchestrator.add_conditional_edge("task2", ["task3", END], orchestrate)

    graph_orchestrator.add_edge("task3", END)

    wf_orchestrator = Workflow(
        flow=Flow(
            nodes=[graph_orchestrator],
        ),
    )

    return wf_orchestrator


def get_orchestrator_workflow2(model: str, connection: connections.OpenAI, context_input: dict):
    llm = OpenAI(
        name="OpenAI",
        model=model,
        connection=connection,
        temperature=0.1,
    )

    agent_manager = GraphAgentManager(llm=llm)
    graph_orchestrator = GraphOrchestrator(
        manager=agent_manager,
        initial_state="task1_task2",
        final_summarizer=True,
        context=context_input,
        enable_handle_input=False,
    )

    # Task 1
    def task1(context: dict, **kwargs):
        return {"result": "task 1 completed", "task1": "task 1 result"}

    # Task 2
    def run(context: dict, **kwargs):
        return {"result": "task 2 completed", "task2": "task 2 result"}

    task2 = Python(code=textwrap.dedent(inspect.getsource(run)))

    graph_orchestrator.add_state_by_tasks("task1_task2", [task1, task2])
    graph_orchestrator.add_edge("task1_task2", END)

    wf_orchestrator = Workflow(
        flow=Flow(
            nodes=[graph_orchestrator],
        ),
    )

    return wf_orchestrator


@pytest.mark.parametrize(
    ("get_orchestrator_workflow", "context_input", "outputs"),
    [
        (
            get_orchestrator_workflow1,
            {},
            {
                "content": "task 2 completed",
                "context": {
                    "task1": "task 1 result",
                    "task2": "task 2 result",
                    "history": [
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": "task 1 completed"},
                        {"role": "assistant", "content": "task 2 completed"},
                    ],
                },
            },
        ),
        (
            get_orchestrator_workflow1,
            {"task3": True},
            {
                "content": "task 3 completed",
                "context": {
                    "task1": "task 1 result",
                    "task2": "task 2 result",
                    "task3": "task 3 result",
                    "history": [
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": "task 1 completed"},
                        {"role": "assistant", "content": "task 2 completed"},
                        {"role": "assistant", "content": "task 3 completed"},
                    ],
                },
            },
        ),
        (
            get_orchestrator_workflow2,
            {},
            {
                "content": "task 2 completed",
                "context": {
                    "task1": "task 1 result",
                    "task2": "task 2 result",
                    "history": [
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": "task 1 completed"},
                        {"role": "assistant", "content": "task 2 completed"},
                    ],
                },
            },
        ),
    ],
)
def test_workflow_with_map_node(
    mock_llm_executor, mock_llm_response_text, get_orchestrator_workflow, context_input, outputs
):
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
