import uuid
from unittest.mock import patch

import pytest
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types import Document
from pydantic import ValidationError

from dynamiq.nodes.transformers import TextToDynamiqDocument
from dynamiq.nodes.transformers.dynamiq_document import TextToDynamiqDocumentInputSchema


@patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678"))
@pytest.mark.parametrize(
    "texts, metadata, result",
    [
        (
            ["Sample text 1", "Sample text 2"],
            [{"filename": "sample1.txt"}, {"filename": "sample2.txt"}],
            [
                Document(
                    id="12345678123456781234567812345678",
                    content="Sample text 1",
                    metadata={"filename": "sample1.txt"},
                    embedding=None,
                    score=None,
                ),
                Document(
                    id="12345678123456781234567812345678",
                    content="Sample text 2",
                    metadata={"filename": "sample2.txt"},
                    embedding=None,
                    score=None,
                ),
            ],
        ),
        (
            ["Sample text 1", "Sample text 2"],
            [{"filename": "sample1.txt"}, {}],
            [
                Document(
                    id="12345678123456781234567812345678",
                    content="Sample text 1",
                    metadata={"filename": "sample1.txt"},
                    embedding=None,
                    score=None,
                ),
                Document(
                    id="12345678123456781234567812345678",
                    content="Sample text 2",
                    metadata={},
                    embedding=None,
                    score=None,
                ),
            ],
        ),
    ],
)
def test_workflow_with_text_to_dynamiq_document(mock_uuid, texts, metadata, result):
    wf_json_to_string = Workflow(flow=Flow(nodes=[TextToDynamiqDocument()]))

    input_data = {"texts": texts, "metadata": metadata}
    output = {"documents": result}
    response = wf_json_to_string.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=output,
    ).to_dict(skip_format_types={bytes})

    expected_output = {wf_json_to_string.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )


@pytest.mark.parametrize(
    "texts, metadata, expected_error",
    [
        (
            ["Sample text 1", "Sample text 2"],
            [{"filename": "sample1.txt"}],
            "Metadata list length 1 does not match sources count 2",
        ),
        (
            ["Sample text 1"],
            [{"filename": "sample1.txt"}, {"filename": "sample2.txt"}],
            "Metadata list length 2 does not match sources count 1",
        ),
        (
            ["Sample text 1"],
            [],
            "Metadata list length 0 does not match sources count 1",
        ),
    ],
)
def test_text_to_dynamiq_document_error_cases(texts, metadata, expected_error):
    """Test that appropriate ValidationError is raised for invalid metadata inputs."""
    with pytest.raises(ValidationError) as exc_info:
        TextToDynamiqDocumentInputSchema(texts=texts, metadata=metadata)

    assert expected_error in str(exc_info.value)
