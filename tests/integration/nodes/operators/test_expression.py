import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.operators import Expression
from dynamiq.nodes.types import ExpressionItem, NamedField
from dynamiq.runnables import RunnableConfig, RunnableStatus


def expression_node(**overrides) -> Expression:
    fields = {
        "id": "pricing",
        "name": "pricing",
        "input_fields": [NamedField(id="f1", name="price"), NamedField(id="f2", name="quantity")],
        "expressions": [
            ExpressionItem(id="x1", key="total", expression="(price * quantity) | round(2)"),
            ExpressionItem(id="x2", key="tier", expression="'bulk' if quantity >= 10 else 'single'"),
            ExpressionItem(id="x3", key="items", expression="[price, quantity] | map('float') | list"),
        ],
    }
    return Expression(**(fields | overrides))


def run_node(node: Expression, input_data: dict):
    return node.run(input_data=input_data, config=RunnableConfig(callbacks=[]))


def test_expressions_return_typed_values():
    result = run_node(expression_node(), {"price": 2.5, "quantity": 12})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"total": 30.0, "tier": "bulk", "items": [2.5, 12.0]}


def test_pass_through_adds_the_inputs_and_an_expression_wins_on_a_clash():
    node = expression_node(
        pass_through=True,
        expressions=[ExpressionItem(key="quantity", expression="quantity + 1")],
    )

    assert run_node(node, {"price": 2.5, "quantity": 1}).output == {"price": 2.5, "quantity": 2}


def test_a_missing_input_on_its_own_is_none_but_fails_inside_arithmetic():
    bare = expression_node(expressions=[ExpressionItem(key="discount", expression="discount")])
    assert run_node(bare, {"price": 2.5}).output == {"discount": None}

    arithmetic = expression_node(expressions=[ExpressionItem(key="total", expression="price * quantity")])
    result = run_node(arithmetic, {"price": 2.5})
    assert result.status == RunnableStatus.FAILURE
    assert "'quantity' is undefined" in result.error.message


@pytest.mark.parametrize(
    "expression",
    ["price.__class__.__mro__", "price.__class__.__base__.__subclasses__()", "range(10 ** 9) | length"],
)
def test_the_sandbox_refuses_to_reach_into_the_process(expression):
    node = expression_node(expressions=[ExpressionItem(key="v", expression=expression)])

    result = run_node(node, {"price": 1})

    assert result.status == RunnableStatus.FAILURE
    assert result.output is None


@pytest.mark.parametrize(
    ("expressions", "message"),
    [
        ([ExpressionItem(key="1st", expression="1")], "key '1st' is not a valid identifier"),
        ([ExpressionItem(key="a", expression="1 +")], "'a' is not a valid expression"),
        ([ExpressionItem(key="a", expression="1"), ExpressionItem(key="a", expression="2")], "used twice"),
    ],
)
def test_malformed_expressions_fail_when_the_node_is_built(expressions, message):
    with pytest.raises(ValueError, match=message):
        expression_node(expressions=expressions)


def test_yaml_round_trip(tmp_path):
    workflow = Workflow(id="workflow", flow=Flow(id="flow", nodes=[expression_node(pass_through=True)]))
    path = tmp_path / "expression.yaml"
    workflow.to_yaml_file(path)

    loaded = Workflow.from_yaml_file(str(path), init_components=True)
    node = loaded.flow.nodes[0]

    assert isinstance(node, Expression)
    assert node.pass_through is True
    assert [item.key for item in node.expressions] == ["total", "tier", "items"]
    assert [field.name for field in node.input_fields] == ["price", "quantity"]
    assert loaded.run(input_data={"price": 2, "quantity": 3}).output["pricing"]["output"]["total"] == 6
