import uuid

import pytest
from litellm import get_llm_provider

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Runcrate
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_runcrate_workflow(
    model: str,
    connection: connections.Runcrate,
):
    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                Runcrate(
                    name="Runcrate",
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
    ("model", "expected_model", "sent_model"),
    [
        (
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "openai/deepseek-ai/DeepSeek-V4.1-Flash",
            "openai/deepseek-ai/DeepSeek-V4.1-Flash",
        ),
        ("zai-org/GLM-5.3", "openai/zai-org/GLM-5.3", "openai/zai-org/GLM-5.3"),
        ("openai/gpt-oss-120b", "openai/gpt-oss-120b", "openai/openai/gpt-oss-120b"),
    ],
)
def test_workflow_with_runcrate_llm(mock_llm_response_text, mock_llm_executor, model, expected_model, sent_model):
    connection = connections.Runcrate(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_runcrate_ai = get_runcrate_workflow(model=model, connection=connection)

    response = wf_runcrate_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text},
    ).to_dict()
    expected_output = {wf_runcrate_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    assert response.output == expected_output
    mock_llm_executor.assert_called_once_with(
        tools=None,
        tool_choice=None,
        model=sent_model,
        messages=wf_runcrate_ai.flow.nodes[0].prompt.format_messages(),
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
    assert wf_runcrate_ai.flow.nodes[0].model == expected_model


@pytest.mark.parametrize(
    ("model", "sent_model"),
    [
        ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4.1-Flash"),
        ("zai-org/GLM-5.3", "zai-org/GLM-5.3"),
        ("openai/gpt-oss-120b", "openai/gpt-oss-120b"),
    ],
)
def test_runcrate_routes_through_openai_compatible_client(model, sent_model):
    connection = connections.Runcrate(api_key="api_key")
    llm = Runcrate(model=model, connection=connection)
    params = llm.update_completion_params({"model": llm.model})

    routed_model, provider, _, api_base = get_llm_provider(model=params["model"], api_base=connection.url)

    assert provider == "openai"
    assert routed_model == sent_model
    assert api_base == "https://api.runcrate.ai/v1"
