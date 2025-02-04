import uuid
from unittest.mock import ANY

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Ollama
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_ollama_workflow(
    model: str,
    connection: connections.Ollama,
):
    wf_ollama = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                Ollama(
                    name="ollama",
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

    return wf_ollama


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [
        ("ollama/llama32", "ollama/llama32"),
        ("llama32", "ollama/llama32"),
    ],
)
def test_workflow_with_ollama_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    model = model
    connection = connections.Ollama(
        id=str(uuid.uuid4()),

    )
    wf_ollama_ai = get_ollama_workflow(model=model, connection=connection)

    response = wf_ollama_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text},
    ).to_dict()
    expected_output = {wf_ollama_ai.flow.nodes[0].id: expected_result}
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
        messages=wf_ollama_ai.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=0.1,
        max_tokens=1000,
        stop=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        response_format=None,
        drop_params=True,
        client=ANY,
        api_base='http://localhost:11434',
    )
