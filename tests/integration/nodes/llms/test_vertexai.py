import json
import uuid

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import connections
from dynamiq.flows import Flow
from dynamiq.nodes.llms import VertexAI
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_vertexai_workflow(
    model: str,
    connection: connections.VertexAI,
):
    wf_gemini = Workflow(
        flow=Flow(
            nodes=[
                VertexAI(
                    name="VertexAI",
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
    [("vertex_ai/gemini-pro", "vertex_ai/gemini-pro"), ("gemini-pro", "vertex_ai/gemini-pro")],
)
def test_workflow_with_vertex_ai_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    google_params = {
        "project_id": "your_project_id",
        "private_key_id": "your_private_key_id",
        "private_key": "your_private_key",
        "client_email": "your_client_email",
        "client_id": "your_client_id",
        "client_x509_cert_url": "your_client_x509_cert_url",
        "auth_uri": "your_auth_uri",
        "token_uri": "your_token_uri",
        "auth_provider_x509_cert_url": "your_auth_provider_x509_cert_url",
        "universe_domain": "your_universe_domain",
    }
    connection = connections.VertexAI(
        id=str(uuid.uuid4()), vertex_project_id="test-project-id", vertex_project_location="us-west1", **google_params
    )
    wf_gemini_vertex_ai = get_vertexai_workflow(model=model, connection=connection)

    response = wf_gemini_vertex_ai.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text},
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
        temperature=wf_gemini_vertex_ai.flow.nodes[0].temperature,
        max_tokens=None,
        stop=None,
        seed=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        vertex_project=connection.vertex_project_id,
        vertex_location=connection.vertex_project_location,
        vertex_credentials=json.dumps(google_params),
        response_format=None,
        drop_params=True,
    )
