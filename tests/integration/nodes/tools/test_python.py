from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.tools import Python
from dynamiq.runnables import RunnableResult, RunnableStatus


def test_workflow_with_python(openai_node, anthropic_node, mock_llm_executor, mock_llm_response_text):
    input_data = {"question": "What is LLM?"}
    python_node_extra_output = {"test_python": "test_python"}

    python_node = (
        Python(
            code=f"def run(inputs): return inputs | {python_node_extra_output}",
        )
        .inputs(question_lowercase=lambda inputs, outputs: inputs["question"].lower())
        .depends_on([openai_node, anthropic_node])
    )
    wf = Workflow(flow=Flow(nodes=[openai_node, anthropic_node, python_node]))

    response = wf.run(input_data=input_data)

    expected_output_openai_anthropic = {"content": mock_llm_response_text, "tool_calls": None}
    expected_result_openai_anthropic = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output_openai_anthropic,
    )
    expected_input_python = input_data | {
        openai_node.id: expected_result_openai_anthropic.to_tracing_depend_dict(),
        anthropic_node.id: expected_result_openai_anthropic.to_tracing_depend_dict(),
        "question_lowercase": input_data["question"].lower(),
    }
    expected_output_python = {"content": expected_input_python | python_node_extra_output}
    expected_result_python = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_python,
        output=expected_output_python,
    )

    expected_output = {
        openai_node.id: expected_result_openai_anthropic.to_dict(),
        anthropic_node.id: expected_result_openai_anthropic.to_dict(),
        python_node.id: expected_result_python.to_dict(),
    }

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
    assert mock_llm_executor.call_count == 2
