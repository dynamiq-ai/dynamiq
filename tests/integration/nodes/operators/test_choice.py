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
                "input": None,
                "output": None,
                "error": None,
                "status": RunStatus.SUCCEEDED,
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
        "metadata": ANY,
        "error": None,
        "executions": [
            {
                "id": ANY,
                "start_time": ANY,
                "end_time": ANY,
                "status": RunStatus.FAILED,
                "error": {
                    "message": str(error),
                    "traceback": ANY,
                },
                "input": None,
                "output": None,
                "metadata": ANY,
            },
            {
                "id": ANY,
                "start_time": ANY,
                "end_time": ANY,
                "status": RunStatus.FAILED,
                "error": {
                    "message": str(error),
                    "traceback": ANY,
                },
                "input": None,
                "output": None,
                "metadata": ANY,
            },
            {
                "id": ANY,
                "start_time": ANY,
                "end_time": ANY,
                "status": RunStatus.SUCCEEDED,
                "input": None,
                "output": None,
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


@pytest.mark.parametrize(
    ("operator", "input_data", "expected"),
    [
        (operators.ConditionOperator.BOOLEAN_EQUALS_PATH, {"a": True, "b": True}, True),
        (operators.ConditionOperator.BOOLEAN_EQUALS_PATH, {"a": True, "b": False}, False),
        (operators.ConditionOperator.NUMERIC_EQUALS_PATH, {"a": 4, "b": 4}, True),
        (operators.ConditionOperator.NUMERIC_EQUALS_PATH, {"a": 4, "b": 5}, False),
        (operators.ConditionOperator.NUMERIC_GREATER_THAN_PATH, {"a": 5, "b": 4}, True),
        (operators.ConditionOperator.NUMERIC_GREATER_THAN_PATH, {"a": 4, "b": 4}, False),
        (operators.ConditionOperator.NUMERIC_GREATER_THAN_OR_EQUALS_PATH, {"a": 4, "b": 4}, True),
        (operators.ConditionOperator.NUMERIC_GREATER_THAN_OR_EQUALS_PATH, {"a": 3, "b": 4}, False),
        (operators.ConditionOperator.NUMERIC_LESS_THAN_PATH, {"a": 3, "b": 4}, True),
        (operators.ConditionOperator.NUMERIC_LESS_THAN_PATH, {"a": 4, "b": 4}, False),
        (operators.ConditionOperator.NUMERIC_LESS_THAN_OR_EQUALS_PATH, {"a": 4, "b": 4}, True),
        (operators.ConditionOperator.NUMERIC_LESS_THAN_OR_EQUALS_PATH, {"a": 5, "b": 4}, False),
        (operators.ConditionOperator.STRING_EQUALS_PATH, {"a": "x", "b": "x"}, True),
        (operators.ConditionOperator.STRING_EQUALS_PATH, {"a": "x", "b": "y"}, False),
        (operators.ConditionOperator.STRING_GREATER_THAN_PATH, {"a": "b", "b": "a"}, True),
        (operators.ConditionOperator.STRING_GREATER_THAN_PATH, {"a": "a", "b": "a"}, False),
        (operators.ConditionOperator.STRING_GREATER_THAN_OR_EQUALS_PATH, {"a": "a", "b": "a"}, True),
        (operators.ConditionOperator.STRING_GREATER_THAN_OR_EQUALS_PATH, {"a": "a", "b": "b"}, False),
        (operators.ConditionOperator.STRING_LESS_THAN_PATH, {"a": "a", "b": "b"}, True),
        (operators.ConditionOperator.STRING_LESS_THAN_PATH, {"a": "a", "b": "a"}, False),
        (operators.ConditionOperator.STRING_LESS_THAN_OR_EQUALS_PATH, {"a": "a", "b": "a"}, True),
        (operators.ConditionOperator.STRING_LESS_THAN_OR_EQUALS_PATH, {"a": "b", "b": "a"}, False),
    ],
)
def test_evaluate_path_operators(operator, input_data, expected):
    """Path operators compare the variable against the value another JSONPath resolves to."""
    condition = operators.ChoiceCondition(operator=operator, variable="$.a", value="$.b")
    assert operators.Choice.evaluate(condition, input_data) is expected

    negated = operators.ChoiceCondition(operator=operator, variable="$.a", value="$.b", is_not=True)
    assert operators.Choice.evaluate(negated, input_data) is (not expected)


def test_every_path_operator_is_evaluable():
    path_operators = {operator for operator in operators.ConditionOperator if operator.value.endswith("-path")}
    assert path_operators == set(operators.PATH_OPERATORS)


@pytest.mark.parametrize(
    ("operator", "value", "input_data"),
    [
        (operators.ConditionOperator.NUMERIC_EQUALS_PATH, 4, {"a": 4}),
        (operators.ConditionOperator.NUMERIC_EQUALS_PATH, "", {"a": 4}),
        (operators.ConditionOperator.NUMERIC_EQUALS_PATH, "5", {"a": 5}),
        (operators.ConditionOperator.STRING_EQUALS_PATH, "USD", {"a": "USD", "USD": "USD"}),
    ],
)
def test_evaluate_path_operator_requires_jsonpath_value(operator, value, input_data):
    """A value that is not a rooted path is refused: a bare word parses as a path to the field of that name."""
    condition = operators.ChoiceCondition(operator=operator, variable="$.a", value=value)

    with pytest.raises(ValueError, match="requires a JSONPath"):
        operators.Choice.evaluate(condition, input_data)


@pytest.mark.parametrize(
    ("operator", "value", "input_data", "selected"),
    [
        (operators.ConditionOperator.NUMERIC_EQUALS_PATH, "$.b", {"a": 4}, 0),
        (operators.ConditionOperator.NUMERIC_EQUALS_PATH, "$.b", {"c": 4}, 0),
        (operators.ConditionOperator.NUMERIC_GREATER_THAN_PATH, "$.b", {"a": 4}, 0),
        (operators.ConditionOperator.STRING_EQUALS_PATH, "$.b", {"a": "x"}, 0),
        (operators.ConditionOperator.NUMERIC_EQUALS_PATH, "$.items[*].n", {"a": 4, "items": [{"n": 4}, {"n": 5}]}, 2),
    ],
)
def test_evaluate_path_operator_needs_exactly_one_value(operator, value, input_data, selected):
    """A value path that selects nothing, or several values, fails the condition rather than comparing `None`."""
    condition = operators.ChoiceCondition(operator=operator, variable="$.a", value=value)

    with pytest.raises(ValueError, match=f"selected {selected} values"):
        operators.Choice.evaluate(condition, input_data)


def test_evaluate_path_operator_reads_a_list_valued_field_as_one_value():
    """One match whose value is a list is that list, which is what tells it apart from several matches."""
    condition = operators.ChoiceCondition(
        operator=operators.ConditionOperator.STRING_EQUALS_PATH, variable="$.a", value="$.b"
    )

    assert operators.Choice.evaluate(condition, {"a": [1, 2], "b": [1, 2]}) is True
    assert operators.Choice.evaluate(condition, {"a": [1, 2], "b": [1]}) is False


def test_evaluate_path_operator_inside_group_uses_group_scope():
    """Operands of a group resolve both paths against the data its variable selects."""
    condition = operators.ChoiceCondition(
        operator=operators.ConditionOperator.AND,
        variable="$.order",
        operands=[
            operators.ChoiceCondition(
                operator=operators.ConditionOperator.NUMERIC_GREATER_THAN_PATH,
                variable="$.total",
                value="$.limit",
            ),
            operators.ChoiceCondition(
                operator=operators.ConditionOperator.STRING_EQUALS_PATH,
                variable="$.currency",
                value="$.account_currency",
            ),
        ],
    )

    within_limit = {"order": {"total": 10, "limit": 5, "currency": "USD", "account_currency": "USD"}}
    assert operators.Choice.evaluate(condition, within_limit) is True

    over_limit = {"order": {"total": 10, "limit": 50, "currency": "USD", "account_currency": "USD"}}
    assert operators.Choice.evaluate(condition, over_limit) is False


@pytest.mark.parametrize(
    ("operator", "input_data", "expected"),
    [
        (operators.ConditionOperator.OR, {"a": 1, "b": 0}, True),
        (operators.ConditionOperator.OR, {"a": 0, "b": 0}, False),
        (operators.ConditionOperator.AND, {"a": 1, "b": 2}, True),
        (operators.ConditionOperator.AND, {"a": 1, "b": 0}, False),
    ],
)
def test_evaluate_negated_group(operator, input_data, expected):
    """is_not on an AND/OR group negates the group instead of forcing it to False."""
    operands = [
        operators.ChoiceCondition(operator=operators.ConditionOperator.NUMERIC_EQUALS, variable="$.a", value=1),
        operators.ChoiceCondition(operator=operators.ConditionOperator.NUMERIC_EQUALS, variable="$.b", value=2),
    ]
    group = operators.ChoiceCondition(operator=operator, operands=operands)
    assert operators.Choice.evaluate(group, input_data) is expected

    negated_group = operators.ChoiceCondition(operator=operator, operands=operands, is_not=True)
    assert operators.Choice.evaluate(negated_group, input_data) is (not expected)


@pytest.mark.parametrize(
    ("input_data", "expected_statuses"),
    [
        ({"a": 4, "b": 4}, [RunnableStatus.SUCCESS, RunnableStatus.SKIP]),
        ({"a": 4, "b": 5}, [RunnableStatus.FAILURE, RunnableStatus.SUCCESS]),
    ],
)
def test_workflow_with_path_operator(input_data, expected_statuses):
    """A variable-to-variable branch, in the shape the platform UI serializes it."""
    choice_node = operators.Choice(
        name="PathChoice",
        options=[
            {
                "id": "equal",
                "name": "equal",
                "condition": {
                    "operator": "numeric-equals-path",
                    "variable": "$.a",
                    "value": "$.b",
                    "is_not": False,
                },
            },
            {"id": "default", "name": "default", "condition": None},
        ],
    )
    wf_path_operator = Workflow(id=str(uuid.uuid4()), flow=Flow(nodes=[choice_node]))

    response = wf_path_operator.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS
    option_results = response.output[choice_node.id]["output"]
    assert [option_results[option.id]["status"] for option in choice_node.options] == expected_statuses


def test_workflow_with_path_operator_over_a_missing_value_path():
    """An ordering operator whose value path selects nothing fails the node, naming the path."""
    choice_node = operators.Choice(
        name="PathChoice",
        options=[
            {
                "id": "over",
                "name": "over",
                "condition": {
                    "operator": "numeric-greater-than-path",
                    "variable": "$.a",
                    "value": "$.limit",
                    "is_not": False,
                },
            },
            {"id": "default", "name": "default", "condition": None},
        ],
    )
    wf_missing_path = Workflow(id=str(uuid.uuid4()), flow=Flow(nodes=[choice_node]))

    response = wf_missing_path.run(input_data={"a": 4})

    assert response.status == RunnableStatus.FAILURE
    node_result = response.output[choice_node.id]
    assert node_result["status"] == RunnableStatus.FAILURE.value
    assert "$.limit" in node_result["error"]["message"]
    assert "selected 0 values" in node_result["error"]["message"]


@pytest.mark.parametrize(
    ("input_data", "statuses"),
    [
        ({"a": 4, "b": "test"}, [RunnableStatus.SUCCESS, RunnableStatus.SUCCESS, RunnableStatus.SKIP]),
        ({"a": 4, "b": "other"}, [RunnableStatus.SUCCESS, RunnableStatus.FAILURE, RunnableStatus.SKIP]),
        ({"a": 1, "b": "other"}, [RunnableStatus.FAILURE, RunnableStatus.FAILURE, RunnableStatus.SUCCESS]),
    ],
)
def test_all_hit_policy_runs_every_matching_branch_and_the_fallback_only_when_none_did(
    choice_condition_a_str_eq, choice_condition_b_str_eq, input_data, statuses
):
    choice_node = operators.Choice(
        hit_policy="all",
        options=[
            operators.ChoiceOption(id="a", condition=choice_condition_a_str_eq),
            operators.ChoiceOption(id="b", condition=choice_condition_b_str_eq),
            operators.ChoiceOption(id="fallback"),
        ],
    )

    result = choice_node.run(input_data=input_data, config=RunnableConfig(callbacks=[]))

    assert [result.output[option].status for option in ("a", "b", "fallback")] == statuses


@pytest.mark.parametrize("fallback_position", ["first", "last"])
def test_all_hit_policy_decides_the_fallback_over_the_whole_list_wherever_it_sits(fallback_position):
    high = operators.ChoiceOption(
        id="high",
        condition=operators.ChoiceCondition(
            operator=operators.ConditionOperator.NUMERIC_GREATER_THAN, variable="$.score", value=5
        ),
    )
    fallback = operators.ChoiceOption(id="fallback")
    options = [fallback, high] if fallback_position == "first" else [high, fallback]
    node = operators.Choice(hit_policy="all", options=options)

    routed = node.run(input_data={"score": 10}, config=RunnableConfig(callbacks=[]))
    unrouted = node.run(input_data={"score": 1}, config=RunnableConfig(callbacks=[]))

    assert {option: result.status for option, result in routed.output.items()} == {
        "fallback": RunnableStatus.SKIP,
        "high": RunnableStatus.SUCCESS,
    }
    assert {option: result.status for option, result in unrouted.output.items()} == {
        "fallback": RunnableStatus.SUCCESS,
        "high": RunnableStatus.FAILURE,
    }


def test_first_hit_policy_still_takes_a_fallback_listed_first():
    high = operators.ChoiceOption(
        id="high",
        condition=operators.ChoiceCondition(
            operator=operators.ConditionOperator.NUMERIC_GREATER_THAN, variable="$.score", value=5
        ),
    )
    node = operators.Choice(options=[operators.ChoiceOption(id="fallback"), high])

    result = node.run(input_data={"score": 10}, config=RunnableConfig(callbacks=[]))

    assert {option: item.status for option, item in result.output.items()} == {
        "fallback": RunnableStatus.SUCCESS,
        "high": RunnableStatus.SKIP,
    }
