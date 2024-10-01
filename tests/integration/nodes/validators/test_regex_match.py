import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.validators import RegexMatch
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.mark.parametrize(
    ("content", "regex"),
    [
        ("abc123", r"^[a-z]+\d+$"),
        ("2021-08-15", r"^\d{4}-\d{2}-\d{2}$"),
        ("hello_world", r"^\w+$"),
        ("user@example.com", r"^[\w\.-]+@[\w\.-]+\.\w+$"),
        ("A1B2C3", r"^[A-Z0-9]+$"),
    ],
)
def test_workflow_with_regex_match(content, regex):
    wf_regex_match = Workflow(
        flow=Flow(nodes=[RegexMatch(regex=regex)]),
    )
    input_data = {"content": content}
    response = wf_regex_match.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"valid": True, **input_data},
    ).to_dict()

    expected_output = {wf_regex_match.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
