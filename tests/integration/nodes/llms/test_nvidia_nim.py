import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import NvidiaNIM
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_nvidia_workflow(
    model: str,
    connection: connections.NvidiaNIM,
):
    wf_nvidia = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                NvidiaNIM(
                    name="NvidiaNIM",
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

    return wf_nvidia


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [("nvidia_nim/google/gemma-7b", "nvidia_nim/google/gemma-7b"), ("google/gemma-7b", "nvidia_nim/google/gemma-7b")],
)
def test_workflow_with_nvidia_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    connection = connections.NvidiaNIM(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_nvidia_nim = get_nvidia_workflow(model=model, connection=connection)

    response = wf_nvidia_nim.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text},
    ).to_dict()
    expected_output = {wf_nvidia_nim.flow.nodes[0].id: expected_result}
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
        messages=wf_nvidia_nim.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=0.1,
        max_tokens=None,
        stop=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        api_key=connection.api_key,
        api_base=None,
        response_format=None,
        drop_params=True,
    )
