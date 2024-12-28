import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Replicate
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_replicate_workflow(
    model: str,
    connection: connections.Replicate,
):
    wf_deepinfra = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                Replicate(
                    name="Replicate",
                    model=model,
                    connection=connection,
                    prompt=Prompt(
                        messages=[
                            Message(
                                role="user",
                                content="What is LLM?",
                            ),
                        ],
                    ),
                    temperature=0.1,
                ),
            ],
        ),
    )

    return wf_deepinfra


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [
        ("replicate/meta/meta-llama-3-8b-instruct", "replicate/meta/meta-llama-3-8b-instruct"),
        ("meta/meta-llama-3-8b-instruct", "replicate/meta/meta-llama-3-8b-instruct"),
    ],
)
def test_workflow_with_replicate_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    model = model
    connection = connections.Replicate(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_replicate_ai = get_replicate_workflow(model=model, connection=connection)

    response = wf_replicate_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text},
    ).to_dict()
    expected_output = {wf_replicate_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    assert response.output == expected_output
    mock_llm_executor.assert_called_once_with(
        tools=None,
        tool_choice=None,
        model=expected_model,
        messages=wf_replicate_ai.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=0.1,
        max_tokens=1000,
        stop=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        api_key=connection.api_key,
        response_format=None,
        drop_params=True,
    )
