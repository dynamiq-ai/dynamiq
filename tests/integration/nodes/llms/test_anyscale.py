import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Anyscale
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_anyscale_workflow(
    model: str,
    connection: connections.Anyscale,
):
    wf_groq = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                Anyscale(
                    name="Anyscale",
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

    return wf_groq


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [
        ("anyscale/meta-llama/Llama-2-70b-chat-hf", "anyscale/meta-llama/Llama-2-70b-chat-hf"),
        ("meta-llama/Llama-2-70b-chat-hf", "anyscale/meta-llama/Llama-2-70b-chat-hf"),
    ],
)
def test_workflow_with_anyscale_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    model = model
    connection = connections.Anyscale(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_anyscale_ai = get_anyscale_workflow(model=model, connection=connection)

    response = wf_anyscale_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text, "tool_calls": None},
    ).to_dict()
    expected_output = {wf_anyscale_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    assert response.output == expected_output
    mock_llm_executor.assert_called_once_with(
        model=expected_model,
        tool_choice=None,
        messages=wf_anyscale_ai.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=0.1,
        max_tokens=1000,
        tools=None,
        stop=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        api_key=connection.api_key,
        response_format=None,
        drop_params=True,
    )
