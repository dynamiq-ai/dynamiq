import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Groq
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_groq_workflow(
    model: str,
    connection: connections.Groq,
):
    wf_groq = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                Groq(
                    name="Groq",
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
    [("groq/llama3-8b-8192", "groq/llama3-8b-8192"), ("llama3-8b-8192", "groq/llama3-8b-8192")],
)
def test_workflow_with_groq_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    model = model
    connection = connections.Groq(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_groq_ai = get_groq_workflow(model=model, connection=connection)

    response = wf_groq_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text, "tool_calls": None},
    ).to_dict()
    expected_output = {wf_groq_ai.flow.nodes[0].id: expected_result}
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
        messages=wf_groq_ai.flow.nodes[0].prompt.format_messages(),
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
