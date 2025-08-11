import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Anthropic
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_anthropic_workflow(
    model: str,
    connection: connections.Anthropic,
):
    wf_anthropic = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                Anthropic(
                    name="Anthropic",
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

    return wf_anthropic


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [
        ("anthropic/claude-opus-4-20250514", "anthropic/claude-opus-4-20250514"),
        ("claude-opus-4-20250514", "anthropic/claude-opus-4-20250514"),
    ],
)
def test_workflow_with_anthropic_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    model = model
    connection = connections.Anthropic(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_anthropic = get_anthropic_workflow(model=model, connection=connection)

    response = wf_anthropic.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text},
    ).to_dict()
    expected_output = {wf_anthropic.flow.nodes[0].id: expected_result}
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
        messages=wf_anthropic.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=wf_anthropic.flow.nodes[0].temperature,
        max_tokens=None,
        stop=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        api_key=connection.api_key,
        response_format=None,
        drop_params=True,
    )
