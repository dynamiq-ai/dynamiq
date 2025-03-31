import uuid
from unittest.mock import ANY

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus, RunType
from dynamiq.flows import Flow
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.operators import operators
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


@pytest.fixture()
def choice_condition_a_str_eq():
    return operators.ChoiceCondition(
        operator=operators.ConditionOperator.NUMERIC_EQUALS,
        variable="$.a",
        value=4,
        operands=[],
    )


@pytest.fixture()
def choice_condition_b_str_eq():
    return operators.ChoiceCondition(
        operator=operators.ConditionOperator.STRING_EQUALS,
        variable="$.b",
        value="test",
        operands=[],
    )


@pytest.fixture()
def choice_condition_a_bool_eq():
    return operators.ChoiceCondition(
        operator=operators.ConditionOperator.BOOLEAN_EQUALS,
        variable="$.a",
        value=True,
        operands=[],
    )


@pytest.fixture()
def choice_condition_a_and_b_str_eq(
    choice_condition_a_str_eq, choice_condition_b_str_eq
):
    return operators.ChoiceCondition(
        operator=operators.ConditionOperator.AND,
        operands=[
            choice_condition_a_str_eq,
            choice_condition_b_str_eq,
        ],
    )


@pytest.fixture()
def choice_option_a_and_b_str_eq(choice_condition_a_and_b_str_eq):
    return operators.ChoiceOption(
        condition=choice_condition_a_and_b_str_eq,
    )


@pytest.fixture()
def choice_option_a_bool_eq(choice_condition_a_bool_eq):
    return operators.ChoiceOption(
        condition=choice_condition_a_bool_eq,
    )


@pytest.fixture()
def choice_option_default():
    return operators.ChoiceOption()


@pytest.fixture()
def choice_node(
    choice_option_a_and_b_str_eq, choice_option_a_bool_eq, choice_option_default
):
    return operators.Choice(
        name="Choice",
        options=[
            choice_option_a_and_b_str_eq,
            choice_option_a_bool_eq,
            choice_option_default,
        ],
        error_handling=ErrorHandling(max_retries=3, backoff_rate=0.2),
    )


@pytest.fixture()
def wf_choice_operator(choice_node):
    return Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[choice_node],
        ),
    )


@pytest.mark.parametrize(
    ("input_data", "choice_options_results"),
    [
        (
            {"a": 4, "b": "test"},
            [
                RunnableResult(status=RunnableStatus.SUCCESS, output=True),
                RunnableResult(status=RunnableStatus.SKIP, output=None),
                RunnableResult(status=RunnableStatus.SKIP, output=None),
            ],
        ),
        (
            {"a": True, "b": "test"},
            [
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.SUCCESS, output=True),
                RunnableResult(status=RunnableStatus.SKIP, output=None),
            ],
        ),
        (
            {"a": 4, "b": 4},
            [
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.SUCCESS, output=True),
            ],
        ),
    ],
)
def test_workflow_with_choice_operator(
    wf_choice_operator,
    choice_node,
    choice_option_a_and_b_str_eq,
    choice_option_a_bool_eq,
    choice_option_default,
    input_data,
    choice_options_results,
    mock_tracing_client,
):
    tracing = TracingCallbackHandler(client=mock_tracing_client())
    choice_node = wf_choice_operator.flow.nodes[0]

    response = wf_choice_operator.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    choice_option_a_and_b_str_eq_result = choice_options_results[0]
    choice_option_a_and_b_str_eq_result.input = input_data
    expected_result_choice_option_a_and_b_str_eq = {
        choice_option_a_and_b_str_eq.id: choice_option_a_and_b_str_eq_result.to_dict(),
    }
    choice_option_a_bool_eq_result = choice_options_results[1]
    choice_option_a_bool_eq_result.input = input_data
    expected_result_choice_option_a_bool_eq = {
        choice_option_a_bool_eq.id: choice_option_a_bool_eq_result.to_dict(),
    }
    choice_option_default_result = choice_options_results[2]
    choice_option_default_result.input = input_data
    expected_result_choice_option_default = {
        choice_option_default.id: choice_option_default_result.to_dict(),
    }

    expected_output_choice_node = (
        expected_result_choice_option_a_and_b_str_eq
        | expected_result_choice_option_a_bool_eq
        | expected_result_choice_option_default
    )
    expected_result_choice_node = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output_choice_node,
    ).to_dict()
    expected_output = {choice_node.id: expected_result_choice_node}

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS, input=input_data, output=expected_output
    )
    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 3
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf_choice_operator.id
    assert wf_run.output == expected_output
    assert wf_run.status == RunStatus.SUCCEEDED
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf_choice_operator.flow.id
    assert flow_run.output == expected_output
    assert flow_run.status == RunStatus.SUCCEEDED
    choice_trace = tracing_runs[2]
    assert choice_trace.to_dict() == {
        "id": ANY,
        "name": choice_node.name,
        "type": RunType.NODE,
        "trace_id": ANY,
        "source_id": ANY,
        "session_id": ANY,
        "start_time": ANY,
        "end_time": ANY,
        "parent_run_id": ANY,
        "status": RunStatus.SUCCEEDED,
        "input": input_data,
        "output": expected_output_choice_node,
        "error": None,
        "metadata": ANY,
        "executions": [
            {
                "id": ANY,
                "start_time": ANY,
                "end_time": ANY,
                "status": RunStatus.SUCCEEDED,
                "input": input_data,
                "output": expected_output_choice_node,
                "error": None,
                "metadata": ANY,
            }
        ],
        "tags": [],
    }
    mock_tracing_client.trace.assert_called_once_with(
        [run for run in tracing.runs.values()]
    )


@pytest.mark.parametrize(
    ("input_data", "choice_options_results"),
    [
        ({"a": 4, "b": "test"}, [True, False]),
        ({"a": True, "b": "test"}, [False, True]),
    ],
)
def test_workflow_with_choice_operator_with_errors_and_retries(
    mocker,
    wf_choice_operator,
    choice_node,
    choice_option_a_and_b_str_eq,
    choice_option_a_bool_eq,
    input_data,
    choice_options_results,
    mock_tracing_client,
):
    tracing = TracingCallbackHandler(client=mock_tracing_client())

    choice_option_a_and_b_str_eq_result = choice_options_results[0]
    expected_result_choice_option_a_and_b_str_eq = {
        choice_option_a_and_b_str_eq.id: RunnableResult(
            status=(
                RunnableStatus.SUCCESS
                if choice_option_a_and_b_str_eq_result
                else RunnableStatus.FAILURE
            ),
            input=input_data,
            output=choice_option_a_and_b_str_eq_result,
        ).to_dict(),
    }
    choice_option_a_bool_eq_result = choice_options_results[1]
    expected_result_choice_option_a_bool_eq = {
        choice_option_a_bool_eq.id: RunnableResult(
            status=(
                RunnableStatus.SUCCESS
                if choice_option_a_bool_eq_result
                else RunnableStatus.FAILURE
            ),
            input=input_data,
            output=choice_option_a_bool_eq_result,
        ).to_dict(),
    }
    expected_output_choice_node = (
        expected_result_choice_option_a_and_b_str_eq
        | expected_result_choice_option_a_bool_eq
    )
    expected_result_choice_node = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output_choice_node,
    ).to_dict()
    expected_output = {choice_node.id: expected_result_choice_node}

    # Handle errors and retries
    max_retries = choice_node.error_handling.max_retries
    error = ValueError("Error")
    error_executions = [error for _ in range(max_retries - 1)]
    success_executions = [expected_output_choice_node]
    executions = error_executions + success_executions
    mocker.patch(
        "dynamiq.nodes.operators.operators.Choice.execute",
        side_effect=executions,
    )

    response = wf_choice_operator.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS, input=input_data, output=expected_output
    )
    assert len(tracing.runs) == 3

    choice_trace = list(tracing.runs.values())[2]
    assert choice_trace.to_dict() == {
        "id": ANY,
        "name": choice_node.name,
        "type": RunType.NODE,
        "trace_id": ANY,
        "source_id": ANY,
        "session_id": ANY,
        "start_time": ANY,
        "end_time": ANY,
        "parent_run_id": ANY,
        "status": RunStatus.SUCCEEDED,
        "input": input_data,
        "output": expected_output_choice_node,
        "error": None,
        "metadata": ANY,
        "executions": [
            {
                "id": ANY,
                "start_time": ANY,
                "end_time": ANY,
                "status": RunStatus.FAILED,
                "input": input_data,
                "output": None,
                "error": {
                    "message": str(error),
                    "traceback": ANY,
                },
                "metadata": ANY,
            },
            {
                "id": ANY,
                "start_time": ANY,
                "end_time": ANY,
                "status": RunStatus.FAILED,
                "input": input_data,
                "output": None,
                "error": {
                    "message": str(error),
                    "traceback": ANY,
                },
                "metadata": ANY,
            },
            {
                "id": ANY,
                "start_time": ANY,
                "end_time": ANY,
                "status": RunStatus.SUCCEEDED,
                "input": input_data,
                "output": expected_output_choice_node,
                "error": None,
                "metadata": ANY,
            },
        ],
        "tags": [],
    }
    mock_tracing_client.trace.assert_called_once_with(
        [run for run in tracing.runs.values()]
    )
