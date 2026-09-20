import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import AtlasCloud
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_atlascloud_workflow(
    model: str,
    connection: connections.AtlasCloud,
):
    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                AtlasCloud(
                    name="AtlasCloud",
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
    return wf


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4.1-mini",
        "anthropic/claude-sonnet-4.6",
    ],
)
def test_workflow_with_atlascloud_llm(mock_llm_response_text, mock_llm_executor, model):
    connection = connections.AtlasCloud(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_atlascloud_ai = get_atlascloud_workflow(model=model, connection=connection)

    response = wf_atlascloud_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text},
    ).to_dict()
    expected_output = {wf_atlascloud_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    assert response.output == expected_output
    mock_llm_executor.assert_called_once_with(
        tools=None,
        tool_choice=None,
        model=model,
        messages=wf_atlascloud_ai.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=0.1,
        max_tokens=None,
        stop=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        api_key=connection.api_key,
        api_base=connection.url,
        custom_llm_provider="openai",
        response_format=None,
        drop_params=True,
    )
