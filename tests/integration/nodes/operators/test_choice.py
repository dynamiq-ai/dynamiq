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
def choice_condition_email_starts_with():
    return operators.ChoiceCondition(
        operator=operators.ConditionOperator.STRING_STARTS_WITH,
        variable="$.email",
        value="user",
        operands=[],
    )


@pytest.fixture()
def choice_condition_message_contains():
    return operators.ChoiceCondition(
        operator=operators.ConditionOperator.STRING_CONTAINS,
        variable="$.message",
        value="test",
        operands=[],
    )


@pytest.fixture()
def choice_condition_filename_regexp():
    return operators.ChoiceCondition(
        operator=operators.ConditionOperator.STRING_REGEXP,
        variable="$.filename",
        value=r"\.pdf$",
        operands=[],
    )


@pytest.fixture()
def choice_condition_filename_ends_with():
    return operators.ChoiceCondition(
        operator=operators.ConditionOperator.STRING_ENDS_WITH,
        variable="$.filename",
        value=".docx",
        operands=[],
    )


@pytest.fixture()
def choice_condition_email_starts_with_negated():
    return operators.ChoiceCondition(
        operator=operators.ConditionOperator.STRING_STARTS_WITH,
        variable="$.email",
        value="admin",
        is_not=True,
        operands=[],
    )


@pytest.fixture()
def choice_condition_filename_ends_with_negated():
    return operators.ChoiceCondition(
        operator=operators.ConditionOperator.STRING_ENDS_WITH,
        variable="$.filename",
        value=".txt",
        is_not=True,
        operands=[],
    )


@pytest.fixture()
def choice_option_email_starts_with(choice_condition_email_starts_with):
    return operators.ChoiceOption(
        condition=choice_condition_email_starts_with,
    )


@pytest.fixture()
def choice_option_message_contains(choice_condition_message_contains):
    return operators.ChoiceOption(
        condition=choice_condition_message_contains,
    )


@pytest.fixture()
def choice_option_filename_regexp(choice_condition_filename_regexp):
    return operators.ChoiceOption(
        condition=choice_condition_filename_regexp,
    )


@pytest.fixture()
def choice_option_filename_ends_with(choice_condition_filename_ends_with):
    return operators.ChoiceOption(
        condition=choice_condition_filename_ends_with,
    )


@pytest.fixture()
def choice_option_email_starts_with_negated(choice_condition_email_starts_with_negated):
    return operators.ChoiceOption(
        condition=choice_condition_email_starts_with_negated,
    )


@pytest.fixture()
def choice_option_filename_ends_with_negated(choice_condition_filename_ends_with_negated):
    return operators.ChoiceOption(
        condition=choice_condition_filename_ends_with_negated,
    )


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


@pytest.mark.parametrize(
    ("input_data", "choice_options_results"),
    [
        (
            {"email": "user@example.com", "message": "Hello test", "filename": "document.pdf"},
            [
                RunnableResult(status=RunnableStatus.SUCCESS, output=True),
                RunnableResult(status=RunnableStatus.SKIP, output=None),
                RunnableResult(status=RunnableStatus.SKIP, output=None),
                RunnableResult(status=RunnableStatus.SKIP, output=None),
            ],
        ),
        (
            {"email": "admin@example.com", "message": "Hello test", "filename": "document.pdf"},
            [
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.SUCCESS, output=True),
                RunnableResult(status=RunnableStatus.SKIP, output=None),
                RunnableResult(status=RunnableStatus.SKIP, output=None),
            ],
        ),
        (
            {"email": "admin@example.com", "message": "Hello world", "filename": "document.pdf"},
            [
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.SUCCESS, output=True),
                RunnableResult(status=RunnableStatus.SKIP, output=None),
            ],
        ),
        (
            {"email": "admin@example.com", "message": "Hello world", "filename": "report.pdf"},
            [
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.SUCCESS, output=True),
                RunnableResult(status=RunnableStatus.SKIP, output=None),
            ],
        ),
        (
            {"email": "admin@example.com", "message": "Hello world", "filename": "document.docx"},
            [
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
                RunnableResult(status=RunnableStatus.SUCCESS, output=True),
            ],
        ),
    ],
)
def test_workflow_with_string_operators(
    choice_option_email_starts_with,
    choice_option_message_contains,
    choice_option_filename_regexp,
    choice_option_filename_ends_with,
    input_data,
    choice_options_results,
    mock_tracing_client,
):
    """Test workflow with string operators: STRING_STARTS_WITH, STRING_CONTAINS, STRING_REGEXP, STRING_ENDS_WITH."""
    tracing = TracingCallbackHandler(client=mock_tracing_client())

    choice_node = operators.Choice(
        name="StringChoice",
        options=[
            choice_option_email_starts_with,
            choice_option_message_contains,
            choice_option_filename_regexp,
            choice_option_filename_ends_with,
        ],
        error_handling=ErrorHandling(max_retries=3, backoff_rate=0.2),
    )

    wf_string_operators = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[choice_node]),
    )

    response = wf_string_operators.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    email_starts_with_result = choice_options_results[0]
    email_starts_with_result.input = input_data
    expected_result_email_starts_with = {
        choice_option_email_starts_with.id: email_starts_with_result.to_dict(),
    }

    message_contains_result = choice_options_results[1]
    message_contains_result.input = input_data
    expected_result_message_contains = {
        choice_option_message_contains.id: message_contains_result.to_dict(),
    }

    filename_regexp_result = choice_options_results[2]
    filename_regexp_result.input = input_data
    expected_result_filename_regexp = {
        choice_option_filename_regexp.id: filename_regexp_result.to_dict(),
    }

    filename_ends_with_result = choice_options_results[3]
    filename_ends_with_result.input = input_data
    expected_result_filename_ends_with = {
        choice_option_filename_ends_with.id: filename_ends_with_result.to_dict(),
    }

    expected_output_choice_node = (
        expected_result_email_starts_with
        | expected_result_message_contains
        | expected_result_filename_regexp
        | expected_result_filename_ends_with
    )
    expected_result_choice_node = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output_choice_node,
    ).to_dict()
    expected_output = {choice_node.id: expected_result_choice_node}

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)


@pytest.mark.parametrize(
    ("input_data", "expected_result"),
    [
        (
            {"filename": "document.pdf"},
            RunnableResult(status=RunnableStatus.SUCCESS, output=True),
        ),
        (
            {"filename": "document.txt"},
            RunnableResult(status=RunnableStatus.FAILURE, output=False),
        ),
    ],
)
def test_workflow_with_string_ends_with_operator_negative(
    choice_option_filename_ends_with_negated,
    input_data,
    expected_result,
    mock_tracing_client,
):
    """Test workflow with STRING_ENDS_WITH operator (is_not=True)."""
    tracing = TracingCallbackHandler(client=mock_tracing_client())

    choice_node = operators.Choice(
        name="EndsWithNegationChoice",
        options=[choice_option_filename_ends_with_negated],
        error_handling=ErrorHandling(max_retries=3, backoff_rate=0.2),
    )

    wf_negation = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[choice_node]),
    )

    response = wf_negation.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    expected_result.input = input_data
    expected_output_choice_node = {
        choice_option_filename_ends_with_negated.id: expected_result.to_dict(),
    }
    expected_result_choice_node = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output_choice_node,
    ).to_dict()
    expected_output = {choice_node.id: expected_result_choice_node}

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)


@pytest.mark.parametrize(
    ("input_data", "expected_result"),
    [
        (
            {"email": "user@example.com"},
            RunnableResult(status=RunnableStatus.SUCCESS, output=True),
        ),
        (
            {"email": "admin@example.com"},
            RunnableResult(status=RunnableStatus.FAILURE, output=False),
        ),
    ],
)
def test_workflow_with_string_operator_negative(
    choice_option_email_starts_with_negated,
    input_data,
    expected_result,
    mock_tracing_client,
):
    """Test workflow with string operator (is_not=True)."""
    tracing = TracingCallbackHandler(client=mock_tracing_client())

    choice_node = operators.Choice(
        name="NegationChoice",
        options=[choice_option_email_starts_with_negated],
        error_handling=ErrorHandling(max_retries=3, backoff_rate=0.2),
    )

    wf_negation = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[choice_node]),
    )

    response = wf_negation.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    # Build expected results
    expected_result.input = input_data
    expected_output_choice_node = {
        choice_option_email_starts_with_negated.id: expected_result.to_dict(),
    }
    expected_result_choice_node = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output_choice_node,
    ).to_dict()
    expected_output = {choice_node.id: expected_result_choice_node}

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)


@pytest.mark.parametrize(
    ("input_data", "choice_options_results"),
    [
        (
            {"message": "Hello TEST message"},
            [
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
            ],
        ),
        (
            {"status_code": "200"},
            [
                RunnableResult(status=RunnableStatus.SUCCESS, output=True),
            ],
        ),
        (
            {"filename": "report.DOCX"},
            [
                RunnableResult(status=RunnableStatus.FAILURE, output=False),
            ],
        ),
        (
            {"filename": "document.docx"},
            [
                RunnableResult(status=RunnableStatus.SUCCESS, output=True),
            ],
        ),
    ],
)
def test_workflow_with_string_operator_edge_cases(
    input_data,
    choice_options_results,
    mock_tracing_client,
):
    """Test edge cases for string operators."""
    tracing = TracingCallbackHandler(client=mock_tracing_client())

    if "message" in input_data:
        condition = operators.ChoiceCondition(
            operator=operators.ConditionOperator.STRING_CONTAINS,
            variable="$.message",
            value="test",
            operands=[],
        )
    elif "status_code" in input_data:
        condition = operators.ChoiceCondition(
            operator=operators.ConditionOperator.STRING_STARTS_WITH,
            variable="$.status_code",
            value="2",
            operands=[],
        )
    else:
        condition = operators.ChoiceCondition(
            operator=operators.ConditionOperator.STRING_ENDS_WITH,
            variable="$.filename",
            value=".docx",
            operands=[],
        )

    choice_option = operators.ChoiceOption(condition=condition)

    choice_node = operators.Choice(
        name="EdgeCaseChoice",
        options=[choice_option],
        error_handling=ErrorHandling(max_retries=3, backoff_rate=0.2),
    )

    wf_edge_case = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[choice_node]),
    )

    response = wf_edge_case.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    expected_result = choice_options_results[0]
    expected_result.input = input_data
    expected_output_choice_node = {
        choice_option.id: expected_result.to_dict(),
    }
    expected_result_choice_node = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output_choice_node,
    ).to_dict()
    expected_output = {choice_node.id: expected_result_choice_node}

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
