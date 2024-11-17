import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Bedrock
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_bedrock_workflow(
    model: str,
    connection: connections.AWS,
):
    wf_bedrock = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                Bedrock(
                    name="bedrock",
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

    return wf_bedrock


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [
        ("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"),
        ("anthropic.claude-3-5-sonnet-20240620-v1:0", "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"),
    ],
)
def test_workflow_with_bedrock_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    model = model
    connection = connections.AWS(
        id=str(uuid.uuid4()),
        profile="default",
        region="us-east-1",
    )
    wf_bedrock_ai = get_bedrock_workflow(model=model, connection=connection)

    response = wf_bedrock_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text, "tool_calls": None},
    ).to_dict()
    expected_output = {wf_bedrock_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    assert response.output == expected_output
    mock_llm_executor.assert_called_once_with(
        model=expected_model,
        tool_choice=None,
        messages=wf_bedrock_ai.flow.nodes[0].prompt.format_messages(),
        stream=False,
        tools=None,
        temperature=0.1,
        max_tokens=1000,
        stop=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        aws_profile_name='default',
        aws_region_name='us-east-1',
        response_format=None,
        drop_params=True,
    )
