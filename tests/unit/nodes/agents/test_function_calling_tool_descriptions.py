import json

from litellm import ModelResponse
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.function_tool import function_tool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus


@function_tool
def lookup(query: str) -> str:
    """Look something up."""
    return query


def _final_answer(**_kwargs):
    call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function=Function(name="provide_final_answer", arguments=json.dumps({"thought": "Done.", "answer": "ok"})),
    )
    message = Message(role="assistant", content=None, tool_calls=[call])
    return ModelResponse(choices=[Choices(index=0, finish_reason="tool_calls", message=message)])


def test_a_tool_description_longer_than_a_function_schema_allows_still_reaches_the_model(mocker):
    rules = " ".join(f"Rule {n}: pass the query in its original language." for n in range(40))
    description = f"Searches the company wiki.\n\n{rules}\n\nNever search for credentials."
    agent = Agent(
        name="assistant",
        llm=OpenAI(model="gpt-5.4-mini", connection=OpenAIConnection(api_key="sk-test")),
        tools=[lookup(description=description)],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=2,
    )
    completion = mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=_final_answer)

    result = agent.run(input_data={"input": "Find the travel policy."})

    assert result.status == RunnableStatus.SUCCESS, result.error
    request = completion.call_args.kwargs
    sent = next(t["function"]["description"] for t in request["tools"] if t["function"]["name"] == "lookup")
    system = request["messages"][0]["content"]
    assert len(sent) <= 1024
    assert sent.startswith("Searches the company wiki.")
    assert "Rule 39: pass the query in its original language." in system
    assert "Never search for credentials." in system
