import uuid
from unittest.mock import ANY

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.llms.openai import ReasoningEffort
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_openai_workflow(
    model: str,
    connection: connections.OpenAI,
):
    wf_openai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                OpenAI(
                    name="OpenAI",
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
                    temperature=1.0,
                ),
            ],
        ),
    )

    return wf_openai


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [
        ("openai/o4-mini", "openai/o4-mini"),
        ("o4-mini", "openai/o4-mini"),
    ],
)
def test_workflow_with_openai_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    model = model
    connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_openai = get_openai_workflow(model=model, connection=connection)

    response = wf_openai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text},
    ).to_dict()
    expected_output = {wf_openai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    assert response.output == expected_output
    mock_llm_executor.assert_called_once_with(
        api_base="https://api.openai.com/v1",
        api_key=connection.api_key,
        client=ANY,
        tools=None,
        tool_choice=None,
        model=expected_model,
        messages=wf_openai.flow.nodes[0].prompt.format_messages(),
        stream=False,
        max_completion_tokens=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        reasoning_effort=ReasoningEffort.MEDIUM,
        response_format=None,
        drop_params=True,
    )
