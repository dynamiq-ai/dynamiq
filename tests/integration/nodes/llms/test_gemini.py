import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Gemini
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_gemini_workflow(
    model: str,
    connection: connections.Gemini | connections.GeminiVertexAI,
):
    wf_gemini = Workflow(
        flow=Flow(
            nodes=[
                Gemini(
                    name="Gemini",
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

    return wf_gemini


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [("gemini/gemini-pro", "gemini/gemini-pro"), ("gemini-pro", "gemini/gemini-pro")],
)
def test_workflow_with_gemini_llm_and_gemini_ai_studio_conn(
    mock_llm_response_text, mock_llm_executor, model, expected_model
):
    model = model
    connection = connections.Gemini(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_gemini_ai_studio = get_gemini_workflow(model=model, connection=connection)

    response = wf_gemini_ai_studio.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text, "tool_calls": None},
    ).to_dict()
    expected_output = {wf_gemini_ai_studio.flow.nodes[0].id: expected_result}
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
        messages=wf_gemini_ai_studio.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=0.1,
        api_key=connection.api_key,
        max_tokens=1000,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        stop=None,
        response_format=None,
        drop_params=True,
    )


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [("vertex_ai/gemini-pro", "vertex_ai/gemini-pro"), ("gemini-pro", "vertex_ai/gemini-pro")],
)
def test_workflow_with_gemini_llm_and_gemini_vertex_ai_conn(
    mock_llm_response_text, mock_llm_executor, model, expected_model
):
    model = model
    connection = connections.GeminiVertexAI(
        id=str(uuid.uuid4()), project_id="test-project-id", project_location="us-west1"
    )
    wf_gemini_vertex_ai = get_gemini_workflow(model=model, connection=connection)

    response = wf_gemini_vertex_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text, "tool_calls": None},
    ).to_dict()
    expected_output = {wf_gemini_vertex_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    mock_llm_executor.assert_called_once_with(
        tools=None,
        tool_choice=None,
        model=expected_model,
        messages=wf_gemini_vertex_ai.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=0.1,
        max_tokens=1000,
        stop=None,
        seed=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        vertex_project=connection.project_id,
        vertex_location=connection.project_location,
        response_format=None,
        drop_params=True,
    )
