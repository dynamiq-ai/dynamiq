"""The new operators next to the nodes a workflow built in the editor already has."""

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators import Choice, ChoiceOption, DecisionTable, Expression, Map
from dynamiq.nodes.tools import Python
from dynamiq.nodes.types import ChoiceCondition, ConditionOperator, DecisionRule, ExpressionItem, NamedField
from dynamiq.nodes.utils import Input, Output
from dynamiq.nodes.validators import ValidChoices
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus

FEATURES = """
def run(input_data):
    ltv = round(input_data["loan_amount"] / input_data["property_value"] * 100)
    return {"fico": input_data["fico"], "ltv": ltv, "program": input_data["program"].upper()}
"""

FEATURE_SELECTOR = {
    "fico": "$.features.output.content.fico",
    "ltv": "$.features.output.content.ltv",
    "program": "$.features.output.content.program",
}


def eligibility_table(**overrides) -> DecisionTable:
    fields = {
        "id": "eligibility",
        "name": "eligibility",
        "input_columns": [
            NamedField(name="fico", type="int"),
            NamedField(name="ltv", type="int"),
            NamedField(name="program", type="string"),
        ],
        "output_columns": [NamedField(name="decision", type="string")],
        "rules": [
            DecisionRule(name="thin file", when=["< 580", "", ""], then=["decline"]),
            DecisionRule(name="fha floor", when=["[580..619]", "", "FHA"], then=["review"]),
            DecisionRule(name="conforming", when=[">= 620", "<= 97", ""], then=["approve"]),
            DecisionRule(name="otherwise", when=["", "", ""], then=["review"]),
        ],
    }
    return DecisionTable(**(fields | overrides))


def letter_workflow() -> Workflow:
    """Input → Python (features) → table → Choice → LLM, only when approved → Output."""
    start = Input(id="start", name="start")
    features = Python(id="features", name="features", code=FEATURES, depends=[NodeDependency(node=start)])
    table = eligibility_table(
        depends=[NodeDependency(node=features)], input_transformer=InputTransformer(selector=FEATURE_SELECTOR)
    )
    route = Choice(
        id="route",
        name="route",
        options=[
            ChoiceOption(
                id="approved",
                condition=ChoiceCondition(
                    operator=ConditionOperator.STRING_EQUALS, variable="$.decision", value="approve"
                ),
            ),
            ChoiceOption(id="manual"),
        ],
        depends=[NodeDependency(node=table)],
        input_transformer=InputTransformer(selector={"decision": "$.eligibility.output.decision"}),
    )
    letter = OpenAI(
        id="letter",
        name="letter",
        model="gpt-4o-mini",
        connection=connections.OpenAI(api_key="test"),
        prompt=Prompt(
            messages=[
                Message(role="user", content="Draft an approval letter for a {{ program }} loan at {{ ltv }}% LTV.")
            ]
        ),
        depends=[NodeDependency(node=route, option="approved"), NodeDependency(node=features)],
        input_transformer=InputTransformer(
            selector={"program": "$.features.output.content.program", "ltv": "$.features.output.content.ltv"}
        ),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=letter)],
        input_transformer=InputTransformer(selector={"letter": "$.letter.output.content"}),
    )
    return Workflow(flow=Flow(nodes=[start, features, table, route, letter, end]))


def test_a_python_node_feeds_the_table_and_the_choice_gates_the_llm(mock_llm_executor):
    workflow = letter_workflow()

    approved = workflow.run(
        input_data={"fico": 720, "loan_amount": 380_000, "property_value": 400_000, "program": "conventional"}
    )

    assert approved.status == RunnableStatus.SUCCESS
    assert approved.output["eligibility"]["output"]["decision"] == "approve"
    assert approved.output["end"]["output"] == {"letter": "mocked_response"}
    assert mock_llm_executor.call_count == 1

    declined = workflow.run(
        input_data={"fico": 550, "loan_amount": 380_000, "property_value": 400_000, "program": "fha"}
    )

    assert declined.status == RunnableStatus.SUCCESS
    assert declined.output["eligibility"]["output"]["decision"] == "decline"
    assert declined.output["letter"]["status"] == RunnableStatus.SKIP
    assert declined.output["end"]["status"] == RunnableStatus.SKIP
    assert mock_llm_executor.call_count == 1


def test_a_validator_checks_the_decision_and_an_expression_reads_the_verdict():
    start = Input(id="start", name="start")
    table = eligibility_table(
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(
            selector={"fico": "$.start.output.fico", "ltv": "$.start.output.ltv", "program": "$.start.output.program"}
        ),
    )
    guard = ValidChoices(
        id="guard",
        name="guard",
        choices=["approve", "decline"],
        depends=[NodeDependency(node=table)],
        input_transformer=InputTransformer(selector={"content": "$.eligibility.output.decision"}),
    )
    verdict = Expression(
        id="verdict",
        name="verdict",
        expressions=[ExpressionItem(key="label", expression="content | upper if valid else 'needs a human'")],
        depends=[NodeDependency(node=guard)],
        input_transformer=InputTransformer(
            selector={"valid": "$.guard.output.valid", "content": "$.guard.output.content"}
        ),
    )
    workflow = Workflow(flow=Flow(nodes=[start, table, guard, verdict]))

    approved = workflow.run(input_data={"fico": 720, "ltv": 80, "program": "VA"})
    review = workflow.run(input_data={"fico": 600, "ltv": 80, "program": "FHA"})

    assert approved.output["verdict"]["output"] == {"label": "APPROVE"}
    assert review.output["guard"]["output"] == {"valid": False, "content": "review"}
    assert review.output["verdict"]["output"] == {"label": "needs a human"}


def test_a_map_runs_the_table_over_a_batch_in_parallel():
    batch = Map(id="batch", name="batch", node=eligibility_table(), max_workers=4)
    applications = [
        {"fico": 720, "ltv": 80, "program": "VA"},
        {"fico": 550, "ltv": 80, "program": "VA"},
        {"fico": 600, "ltv": 90, "program": "FHA"},
        # A value the column cannot read matches only empty cells, so it lands on the catch-all rule.
        {"fico": "seven hundred", "ltv": 80, "program": "VA"},
    ] * 3

    result = batch.run(input_data={"input": applications}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert [item["decision"] for item in result.output["output"]] == ["approve", "decline", "review", "review"] * 3
