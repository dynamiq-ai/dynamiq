import uuid


from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.llms import CustomLLM
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


def get_customllm_workflow(
    model: str,
    connection: connections.Http,
):
    wf_customllm = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                CustomLLM(
                    name="CustomLLM",
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

    return wf_customllm


def test_workflow_with_custom_llm(mock_llm_response_text, mock_llm_executor):
    model = "groq/llama3-8b-8192"
    connection = connections.HttpApiKey(
        id=str(uuid.uuid4()),
        api_key="api_key",
        url="http://localhost:8000",
    )
    wf_customllm_studio = get_customllm_workflow(model=model, connection=connection)

    response = wf_customllm_studio.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text, "tool_calls": None},
    ).to_dict()
    expected_output = {wf_customllm_studio.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    assert response.output == expected_output
    mock_llm_executor.assert_called_once_with(
        tool_choice=None,
        api_base=connection.url,
        model=model,
        messages=wf_customllm_studio.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=0.1,
        api_key=connection.api_key,
        max_tokens=1000,
        seed=None,
        frequency_penalty=None,
        tools=None,
        presence_penalty=None,
        top_p=None,
        stop=None,
        response_format=None,
        drop_params=True,
    )
