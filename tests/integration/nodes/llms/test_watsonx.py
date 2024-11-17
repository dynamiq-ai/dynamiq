import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import WatsonX
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_watsonx_workflow(
    model: str,
    connection: connections.WatsonX,
):
    wf_groq = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                WatsonX(
                    name="watsonx",
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
        ("watsonx/ibm/granite-13b-chat-v2", "watsonx/ibm/granite-13b-chat-v2"),
        ("ibm/granite-13b-chat-v2", "watsonx/ibm/granite-13b-chat-v2"),
    ],
)
def test_workflow_with_watsonx_ai(mock_llm_response_text, mock_llm_executor, model, expected_model):
    model = model
    connection = connections.WatsonX(
        id=str(uuid.uuid4()),
        api_key="api_key",
        project_id="project_id",
        url="https://your-url",
    )
    wf_watsonx_ai = get_watsonx_workflow(model=model, connection=connection)

    response = wf_watsonx_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text, "tool_calls": None},
    ).to_dict()
    expected_output = {wf_watsonx_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    assert response.output == expected_output
    mock_llm_executor.assert_called_once_with(
        tools=None,
        tool_choice=None,
        url=connection.url,
        project_id=connection.project_id,
        model=expected_model,
        messages=wf_watsonx_ai.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=0.1,
        max_tokens=1000,
        stop=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        apikey=connection.api_key,
        response_format=None,
        drop_params=True,
    )
