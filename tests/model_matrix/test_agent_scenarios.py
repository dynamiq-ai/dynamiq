"""Agent scenarios run against live models (see conftest.py for how to run them and what they cost).

Every tool returns a code made up for the test, so a correct answer can only come from the tool's result.
The assertions read the answer, the tool calls and the requests sent to the provider, never a model's opinion.
"""

import uuid
from typing import Any, ClassVar, Literal

import pytest
from pydantic import BaseModel, Field

from dynamiq.connections import OpenRouter as OpenRouterConnection
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.llms import OpenRouter
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus

ALL_MODES = [InferenceMode.DEFAULT, InferenceMode.XML, InferenceMode.STRUCTURED_OUTPUT, InferenceMode.FUNCTION_CALLING]


def made_up_code() -> str:
    return f"Q{uuid.uuid4().hex[:8].upper()}"


def build_agent(model: dict, tools: list[Node], mode: InferenceMode, **kwargs) -> Agent:
    llm = OpenRouter(
        model=model["id"],
        connection=OpenRouterConnection(),
        max_tokens=4096 if model.get("reasoning") else 1024,
    )
    return Agent(
        name="matrix-agent",
        llm=llm,
        tools=tools,
        role="You answer the user's question with the tools you are given. Quote codes exactly as tools return them.",
        inference_mode=mode,
        max_loops=kwargs.pop("max_loops", 6),
        **kwargs,
    )


def run(agent: Agent, question: str, meter) -> Any:
    return agent.run(input_data={"input": question}, config=RunnableConfig(callbacks=[meter]))


class LookupInput(BaseModel):
    key: str = Field(..., description="What to look up.")


class Lookup(Node):
    """Answers each key with the code in ``answers``; a key it does not know gets an error the model can fix."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "lookup"
    description: str = "Looks up a key and returns its code."
    input_schema: ClassVar[type[LookupInput]] = LookupInput
    answers: dict[str, str] = Field(default_factory=dict)
    calls: list[str] = Field(default_factory=list)
    filler: str = ""

    def execute(self, input_data: LookupInput, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        self.calls.append(input_data.key)
        for key, code in self.answers.items():
            if key.lower() in input_data.key.lower():
                return {"content": f"{self.filler}The code for {key} is {code}."}
        raise ToolExecutionException(
            f"Unknown key '{input_data.key}'. Known keys: {', '.join(self.answers)}.", recoverable=True
        )


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda mode: mode.value)
def test_one_tool_call(model, mode, meter, contract):
    code = made_up_code()
    tool = Lookup(name="order_status", description="Returns an order's status code by its number.", answers={"7": code})
    agent = build_agent(model, [tool], mode)

    result = run(agent, "What is the status code of order 7? Use the order_status tool.", meter)

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert code in str(result.output["content"])
    assert 1 <= len(tool.calls) <= 2
    assert meter.calls <= 4


@pytest.mark.parametrize("mode", [InferenceMode.XML, InferenceMode.FUNCTION_CALLING], ids=lambda mode: mode.value)
def test_two_dependent_calls(model, mode, meter, contract):
    # The second call needs the first one's result, so the model must read a tool result and act on it.
    customer_id, balance = made_up_code(), made_up_code()
    find = Lookup(name="find_customer", description="Returns a customer's id by name.", answers={"ada": customer_id})
    balance_tool = Lookup(
        name="customer_balance",
        description="Returns the balance code for a customer id (not a name).",
        answers={customer_id: balance},
    )
    agent = build_agent(model, [find, balance_tool], mode)

    result = run(agent, "What is the balance code of the customer named Ada?", meter)

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert balance in str(result.output["content"])
    assert find.calls and any(customer_id.lower() in key.lower() for key in balance_tool.calls)


def test_two_independent_calls_in_one_turn(model, meter, contract):
    if not model.get("parallel"):
        pytest.skip("the provider does not list parallel tool calls for this model")
    paris, tokyo = made_up_code(), made_up_code()
    tool = Lookup(name="city_code", description="Returns a city's code.", answers={"paris": paris, "tokyo": tokyo})
    agent = build_agent(model, [tool], InferenceMode.FUNCTION_CALLING, parallel_tool_calls_enabled=True)

    result = run(agent, "Get the codes of Paris and Tokyo. Request both at once, in one turn.", meter)

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert paris in str(result.output["content"]) and tokyo in str(result.output["content"])
    assert meter.calls <= 3


@pytest.mark.parametrize("mode", [InferenceMode.XML, InferenceMode.FUNCTION_CALLING], ids=lambda mode: mode.value)
def test_recovers_from_a_tool_error(model, mode, meter, contract):
    # The first call is expected to miss: the error names the keys that exist, and the model must retry with one.
    code = made_up_code()
    tool = Lookup(name="warehouse", description="Returns a warehouse's code by its key.", answers={"WH-NORTH": code})
    agent = build_agent(model, [tool], mode)

    result = run(agent, "What is the code of the north warehouse? Try the key 'north' first.", meter)

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert code in str(result.output["content"])
    assert meter.calls <= 5


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "code": {"type": "string"}},
    "required": ["city", "code"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "mode", [InferenceMode.STRUCTURED_OUTPUT, InferenceMode.FUNCTION_CALLING], ids=lambda mode: mode.value
)
def test_the_final_answer_follows_the_response_format(model, mode, meter, contract):
    code = made_up_code()
    tool = Lookup(name="city_code", description="Returns a city's code.", answers={"lisbon": code})
    agent = build_agent(model, [tool], mode, response_format=ANSWER_SCHEMA)

    result = run(agent, "Look up the code of Lisbon and answer with the city and its code.", meter)

    assert result.status == RunnableStatus.SUCCESS, result.error
    content = result.output["content"]
    assert isinstance(content, dict) and set(content) == {"city", "code"}
    assert content["code"] == code


def test_compaction_mid_run_keeps_every_request_valid(model, meter, contract):
    # Each page is long, so the history is summarized between calls. The contract fixture fails the cell if a
    # request after compaction carries a tool reply without its call, which providers reject.
    codes = {str(number): made_up_code() for number in (1, 2, 3)}
    tool = Lookup(
        name="read_page",
        description="Returns one page of the report.",
        answers=codes,
        filler="Background notes on the quarterly report. " * 80,
    )
    agent = build_agent(
        model,
        [tool],
        InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        summarization_config=SummarizationConfig(
            enabled=True, max_token_context_length=2500, max_preserved_tokens=1200
        ),
    )

    result = run(agent, "Read page 1, then page 2, then page 3, one at a time. Tell me the code on page 3.", meter)

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert codes["3"] in str(result.output["content"])
