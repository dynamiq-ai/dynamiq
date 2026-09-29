import uuid

import pytest
from litellm import get_llm_provider

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
    ("model", "expected_model"),
    [
        ("openai/gpt-4.1-mini", "openai/gpt-4.1-mini"),
        ("anthropic/claude-sonnet-4.6", "openai/anthropic/claude-sonnet-4.6"),
        ("deepseek-ai/deepseek-v3.2", "openai/deepseek-ai/deepseek-v3.2"),
    ],
)
def test_workflow_with_atlascloud_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
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
        model=expected_model,
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
        response_format=None,
        drop_params=True,
    )
    assert wf_atlascloud_ai.flow.nodes[0].model == expected_model


@pytest.mark.parametrize(
    ("model", "sent_model"),
    [
        ("openai/gpt-4.1-mini", "gpt-4.1-mini"),
        ("anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-4.6"),
        ("deepseek-ai/deepseek-v3.2", "deepseek-ai/deepseek-v3.2"),
    ],
)
def test_atlascloud_routes_through_openai_compatible_client(model, sent_model):
    connection = connections.AtlasCloud(api_key="api_key")
    llm = AtlasCloud(model=model, connection=connection)

    routed_model, provider, _, api_base = get_llm_provider(model=llm.model, api_base=connection.url)

    assert provider == "openai"
    assert routed_model == sent_model
    assert api_base == "https://api.atlascloud.ai/v1"
