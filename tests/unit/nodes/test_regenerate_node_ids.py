from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.cloning import regenerate_node_ids
from dynamiq.nodes.node import NodeDependency, NodeOutputReference
from dynamiq.nodes.operators import Choice, ChoiceOption, DecisionTable, Pass, SubWorkflow
from dynamiq.nodes.types import ChoiceCondition, ConditionOperator, DecisionRule, NamedField


def test_a_node_reachable_twice_gets_one_new_id_and_id_paths_follow_it():
    first = Pass(id="first", name="first")
    second = Pass(
        id="second",
        name="second",
        depends=[NodeDependency(node=first)],
        input_transformer=InputTransformer(
            path="$.first.output", selector={"x": "$.first.output.x || $.first.output.y"}
        ),
    )
    node = SubWorkflow(id="sub", name="sub", flow=Flow(id="flow", nodes=[first, second]))

    id_map: dict[str, set[str]] = {}
    clone = regenerate_node_ids(node.clone(), id_map)

    cloned_first, cloned_second = clone.flow.nodes
    assert cloned_second.depends[0].node is cloned_first
    assert id_map["first"] == {cloned_first.id}
    assert cloned_second.input_transformer.path == f'$."{cloned_first.id}".output'
    assert cloned_second.input_transformer.selector == {
        "x": f'$."{cloned_first.id}".output.x || $."{cloned_first.id}".output.y'
    }
    # The original is untouched.
    assert first.id == "first" and second.input_transformer.path == "$.first.output"

    # A second pass, as under a Map inside a Map, moves the paths on again.
    again = regenerate_node_ids(clone.clone(), {})
    again_first, again_second = again.flow.nodes
    assert again_first.id != cloned_first.id
    assert again_second.input_transformer.path == f'$."{again_first.id}".output'
    assert (
        again_second.input_transformer.selector["x"]
        == f'$."{again_first.id}".output.x || $."{again_first.id}".output.y'
    )


def test_a_dependency_gated_on_a_choice_option_follows_the_option_id():
    route = Choice(
        id="route",
        name="route",
        options=[
            ChoiceOption(
                id="opt-hi",
                condition=ChoiceCondition(
                    operator=ConditionOperator.NUMERIC_GREATER_THAN, variable="$.score", value=50
                ),
            ),
            ChoiceOption(id="opt-lo"),
        ],
    )
    hi = Pass(id="hi", name="hi", depends=[NodeDependency(node=route, option="opt-hi")])
    node = SubWorkflow(id="sub", name="sub", flow=Flow(id="flow", nodes=[route, hi]))

    clone = regenerate_node_ids(node.clone(), {})

    cloned_route, cloned_hi = clone.flow.nodes
    assert cloned_route.options[0].id != "opt-hi"
    assert cloned_hi.depends[0].option == cloned_route.options[0].id
    assert hi.depends[0].option == "opt-hi"


def test_a_flow_copy_relinks_output_references_to_the_copied_nodes():
    start = Pass(id="start", name="start")
    calc = Pass(
        id="calc",
        name="calc",
        depends=[NodeDependency(node=start)],
        input_mapping={"score": NodeOutputReference(node=start, output_key="score"), "scale": 2},
    )
    flow = Flow(id="flow", nodes=[start, calc])

    copied = flow.clone()

    copied_start, copied_calc = copied.nodes
    assert copied_calc.input_mapping["score"].node is copied_start
    assert copied_calc.input_mapping["scale"] == 2
    assert calc.input_mapping["score"].node is start


def test_paths_naming_a_column_or_an_option_rather_than_a_node_are_left_alone():
    table = DecisionTable(
        id="table",
        name="table",
        input_columns=[NamedField(id="fico", name="fico", type="int")],
        output_columns=[NamedField(id="decision", name="decision", type="string")],
        rules=[DecisionRule(id="r1", when=[">= 700"], then=["approve"])],
        input_transformer=InputTransformer(selector={"fico": "$.fico"}),
    )
    route = Choice(
        id="route",
        name="route",
        options=[ChoiceOption(id="query")],
        input_transformer=InputTransformer(selector={"q": "$.query"}),
    )
    node = SubWorkflow(id="sub", name="sub", flow=Flow(id="flow", nodes=[table, route]))

    clone = regenerate_node_ids(node.clone(), {})

    # The column and the option carry new ids, but `$.fico` and `$.query` name input keys, not nodes.
    cloned_table, cloned_route = clone.flow.nodes
    assert cloned_table.input_columns[0].id != "fico"
    assert cloned_table.input_transformer.selector == {"fico": "$.fico"}
    assert cloned_route.input_transformer.selector == {"q": "$.query"}
