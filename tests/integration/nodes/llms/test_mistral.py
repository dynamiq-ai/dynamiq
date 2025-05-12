import uuid
from unittest.mock import ANY

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Mistral
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_mistral_workflow(
    model: str,
    connection: connections.Mistral,
    prompt: Prompt,
):
    return Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                Mistral(
                    name="Mistral",
                    model=model,
                    connection=connection,
                    prompt=prompt,
                    temperature=0.1,
                ),
            ],
        ),
    )


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [
        ("mistral/mistral-tiny", "mistral/mistral-tiny"),
        ("mistral-tiny", "mistral/mistral-tiny"),
    ],
)
def test_workflow_with_mistral_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    prompt = Prompt(messages=[Message(role="user", content="What is LLM?")])
    connection = connections.Mistral(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf = get_mistral_workflow(model=model, connection=connection, prompt=prompt)

    response = wf.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text},
    ).to_dict()
    expected_output = {wf.flow.nodes[0].id: expected_result}

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    mock_llm_executor.assert_called_once_with(
        tools=None,
        tool_choice=None,
        model=expected_model,
        messages=prompt.format_messages(),
        stream=False,
        temperature=0.1,
        max_tokens=None,
        stop=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        api_key=ANY,
        response_format=None,
        drop_params=True,
    )


def test_workflow_prefix_added_when_last_assistant(
    mock_llm_executor,
):
    prompt = Prompt(
        messages=[
            Message(role=MessageRole.SYSTEM, content="Initialize system"),
            Message(role=MessageRole.USER, content="Hello!"),
            Message(role=MessageRole.ASSISTANT, content="How can I help?"),
        ]
    )
    connection = connections.Mistral(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf = get_mistral_workflow(
        model="mistral/mistral-test",
        connection=connection,
        prompt=prompt,
    )

    wf.run(input_data={}, config=RunnableConfig(callbacks=[]))

    called_kwargs = mock_llm_executor.call_args.kwargs
    sent_messages = called_kwargs["messages"]

    last = sent_messages[-1]
    assert last["role"] == MessageRole.ASSISTANT.value
    assert last.get("prefix") is True

    assert all("prefix" not in m for m in sent_messages[:-1])
