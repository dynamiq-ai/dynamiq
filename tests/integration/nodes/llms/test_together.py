import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import TogetherAI
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_together_workflow(
    model: str,
    connection: connections.TogetherAI,
):
    wf_together = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                TogetherAI(
                    name="Together",
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

    return wf_together


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [
        ("together_ai/togethercomputer/llama-2-70b-chat", "together_ai/togethercomputer/llama-2-70b-chat"),
        ("togethercomputer/llama-2-70b-chat", "together_ai/togethercomputer/llama-2-70b-chat"),
    ],
)
def test_workflow_with_together_ai(mock_llm_response_text, mock_llm_executor, model, expected_model):
    model = model
    connection = connections.TogetherAI(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_together_ai = get_together_workflow(model=model, connection=connection)

    response = wf_together_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text, "tool_calls": None},
    ).to_dict()
    expected_output = {wf_together_ai.flow.nodes[0].id: expected_result}
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
        messages=wf_together_ai.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=0.1,
        max_tokens=1000,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        top_p=None,
        api_key=connection.api_key,
        response_format=None,
        drop_params=True,
    )
