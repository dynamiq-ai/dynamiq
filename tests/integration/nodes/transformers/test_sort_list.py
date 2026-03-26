import pytest
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document

from dynamiq.nodes.transformers import SortList
from dynamiq.nodes.transformers.sort_list import SortListInputSchema


@pytest.mark.parametrize(
    "input_list, order, field, expected_result",
    [
        ([3, 5, 6, 7], "DESC", None, [7, 6, 5, 3]),
        ([3, 5, 6, 7], "ASC", None, [3, 5, 6, 7]),
        ([3.2, 5.1, 6.8, 7.0], "DESC", None, [7.0, 6.8, 5.1, 3.2]),
        ([3.2, 5.1, 6.8, 7.0], "ASC", None, [3.2, 5.1, 6.8, 7.0]),
        (["banana", "apple", "cherry", "date"], "DESC", None, ["date", "cherry", "banana", "apple"]),
        (["banana", "apple", "cherry", "date"], "ASC", None, ["apple", "banana", "cherry", "date"]),
        (
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}],
            "ASC",
            "age",
            [{"name": "Bob", "age": 25}, {"name": "Alice", "age": 30}, {"name": "Charlie", "age": 35}],
        ),
        (
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}],
            "DESC",
            "age",
            [{"name": "Charlie", "age": 35}, {"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
        ),
        (
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}],
            "ASC",
            "name",
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}],
        ),
        (
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}],
            "DESC",
            "name",
            [{"name": "Charlie", "age": 35}, {"name": "Bob", "age": 25}, {"name": "Alice", "age": 30}],
        ),
        (
            [
                Document(id="1", content="Document 1", embedding=[0.1, 0.1, 0.2], score=0.4),
                Document(id="2", content="Document 2", embedding=[0.1, 0.1, 0.2], score=0.8),
            ],
            "DESC",
            "score",
            [
                Document(id="2", content="Document 2", embedding=[0.1, 0.1, 0.2], score=0.8),
                Document(id="1", content="Document 1", embedding=[0.1, 0.1, 0.2], score=0.4),
            ],
        ),
    ],
)
def test_workflow_with_sorting_list(input_list, order, field, expected_result):
    wf_sorting_list = Workflow(flow=Flow(nodes=[SortList(sort_by=order)]))

    field = {"field": field} if field else {}
    input_data = {"input": input_list, **field}
    output = {"output": expected_result}
    response = wf_sorting_list.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=dict(SortListInputSchema(**input_data)),
        output=output,
    ).to_dict(skip_format_types={bytes})

    expected_output = {wf_sorting_list.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
