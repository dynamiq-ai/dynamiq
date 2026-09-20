import json
import re

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


def test_an_expression_left_lazy_returns_a_list():
    """`map`, `select` and `selectattr` return generators, which no encoder can record."""
    node = Expression(
        id="calc",
        input_fields=[NamedField(name="items")],
        expressions=[
            ExpressionItem(key="prices", expression="items | map(attribute='price')"),
            ExpressionItem(key="cheap", expression="items | selectattr('price', 'lt', 100)"),
            ExpressionItem(key="keys", expression="items[0].keys()"),
        ],
    )

    result = node.run(input_data={"items": [{"price": 60}, {"price": 120}]}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"prices": [60, 120], "cheap": [{"price": 60}], "keys": ["price"]}
    assert json.dumps(result.output)


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


def test_the_rule_helpers_serve_an_expression_too():
    node = Expression(
        id="due",
        name="due",
        input_fields=[NamedField(name="opened_at"), NamedField(name="closed_at"), NamedField(name="sla_hours")],
        expressions=[
            ExpressionItem(key="age_days", expression="days_between(opened_at, closed_at)"),
            ExpressionItem(key="opened", expression="date(opened_at) | string"),
            ExpressionItem(key="has_sla", expression="has(sla_hours)"),
            ExpressionItem(key="hours", expression="max(sla_hours or 0, 1)"),
            ExpressionItem(key="is_today", expression="date(opened_at) == today()"),
        ],
    )

    result = node.run(
        input_data={"opened_at": "09/01/2026", "closed_at": "2026-09-19", "sla_hours": None},
        config=RunnableConfig(callbacks=[]),
    )

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output == {"age_days": 18, "opened": "2026-09-01", "has_sla": False, "hours": 1, "is_today": False}


def test_an_input_named_self_does_not_stop_the_evaluation():
    """A REST payload's top-level `self` link lands in the input dict and must be an ordinary key there."""
    node = Expression(
        name="pricing",
        input_fields=[NamedField(name="amount")],
        expressions=[ExpressionItem(key="doubled", expression="amount * 2")],
        pass_through=True,
    )

    result = node.run(input_data={"self": "https://api/x/1", "amount": 3}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"self": "https://api/x/1", "amount": 3, "doubled": 6}


def test_an_input_named_like_a_helper_is_the_input_where_it_is_read_and_the_helper_where_it_is_called():
    node = Expression(
        name="dates",
        input_fields=[NamedField(name="date"), NamedField(name="opened")],
        expressions=[
            ExpressionItem(key="dated", expression="date"),
            ExpressionItem(key="opened_year", expression="date(opened).year"),
        ],
    )

    result = node.run(input_data={"date": "2026-09-01", "opened": "2026-08-20"}, config=RunnableConfig(callbacks=[]))
    absent = node.run(input_data={"opened": "2026-08-20"}, config=RunnableConfig(callbacks=[]))

    assert result.output == {"dated": "2026-09-01", "opened_year": 2026}
    # A missing input named like a helper is missing, not the helper.
    assert absent.output == {"dated": None, "opened_year": 2026}
    with pytest.raises(ValueError, match="'clash' reads 'date' as a value and calls it as a helper"):
        Expression(expressions=[ExpressionItem(key="clash", expression="date(date)")])


def test_a_missing_input_inside_a_list_or_a_dict_the_expression_builds_is_none_there_too():
    node = Expression(
        name="pairs",
        input_fields=[NamedField(name="price"), NamedField(name="discount")],
        expressions=[
            ExpressionItem(key="pair", expression="{'a': price, 'b': discount}"),
            ExpressionItem(key="alone", expression="discount"),
            ExpressionItem(key="items", expression="[price, discount]"),
            ExpressionItem(key="nested", expression="[[price, discount], {'d': [discount]}]"),
        ],
    )

    result = node.run(input_data={"price": 2.5}, config=RunnableConfig(callbacks=[]))

    # Jinja turns only a whole undefined result into None; one inside a container has to be replaced the
    # same way, or the output cannot be serialized and raises on its first use downstream.
    assert result.output == {
        "pair": {"a": 2.5, "b": None},
        "alone": None,
        "items": [2.5, None],
        "nested": [[2.5, None], {"d": [None]}],
    }
    assert json.loads(json.dumps(result.output)) == result.output


def test_an_expression_reading_self_is_refused_at_build_while_a_nested_self_reads():
    """Jinja binds `self` to its template reference inside every compiled expression, so an input of that name
    is never the value handed over: `self.href` would read None and `self` an unserializable reference."""
    for expression, where in [("self.href", "'link' reads 'self.href'"), ("self", "'link' reads 'self'")]:
        with pytest.raises(ValueError, match=re.escape(where)):
            Expression(
                name="links",
                input_fields=[NamedField(name="self")],
                expressions=[ExpressionItem(key="link", expression=expression)],
            )

    node = Expression(
        name="links",
        input_fields=[NamedField(name="payload")],
        expressions=[ExpressionItem(key="link", expression="payload.self.href")],
    )

    result = node.run(
        input_data={"payload": {"self": {"href": "https://api/x/1"}}}, config=RunnableConfig(callbacks=[])
    )

    assert result.output == {"link": "https://api/x/1"}


def test_a_record_key_named_like_a_dict_method_reads_as_the_data():
    node = expression_node(
        input_fields=[NamedField(name="invoice")],
        expressions=[
            ExpressionItem(key="count", expression="invoice.items | length"),
            ExpressionItem(key="skus", expression="invoice.items | map(attribute='sku') | list"),
            ExpressionItem(key="vat", expression="invoice.get('vat_rate', 0)"),
        ],
    )

    result = node.run(
        input_data={"invoice": {"items": [{"sku": "a"}, {"sku": "b"}]}}, config=RunnableConfig(callbacks=[])
    )

    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"count": 2, "skus": ["a", "b"], "vat": 0}
