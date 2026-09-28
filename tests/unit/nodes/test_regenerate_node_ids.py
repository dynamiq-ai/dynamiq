from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer, OutputTransformer
from dynamiq.nodes.cloning import regenerate_node_ids
from dynamiq.nodes.node import NodeDependency, NodeOutputReference
from dynamiq.nodes.operators import Choice, ChoiceOption, DecisionTable, Expression, Pass, Rules, SubWorkflow
from dynamiq.nodes.types import (
    ChoiceCondition,
    ConditionOperator,
    DecisionRule,
    DerivedValue,
    ExpressionItem,
    NamedField,
    Rule,
)
from dynamiq.nodes.utils import Input, Output


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


def test_two_choices_sharing_an_option_id_keep_their_own_gates():
    start = Input(id="start", name="start")

    def route(node_id: str, key: str) -> Choice:
        return Choice(
            id=node_id,
            name=node_id,
            options=[
                ChoiceOption(
                    id="go",
                    condition=ChoiceCondition(
                        operator=ConditionOperator.NUMERIC_GREATER_THAN, variable=f"$.start.output.{key}", value=0
                    ),
                ),
                ChoiceOption(id="default"),
            ],
            depends=[NodeDependency(node=start)],
        )

    route_a, route_b = route("route_a", "a"), route("route_b", "b")
    a_go = Pass(id="a_go", name="a_go", depends=[NodeDependency(node=route_a, option="go")])
    b_default = Pass(id="b_default", name="b_default", depends=[NodeDependency(node=route_b, option="default")])
    node = SubWorkflow(id="sub", name="sub", flow=Flow(id="flow", nodes=[start, route_a, route_b, a_go, b_default]))

    clone = regenerate_node_ids(node.clone(), {})

    # A gate follows the option of the Choice it depends on, not the last option of that name the walk met.
    _, cloned_a, cloned_b, cloned_a_go, cloned_b_default = clone.flow.nodes
    assert cloned_a.options[0].id != cloned_b.options[0].id
    assert cloned_a_go.depends[0].option == cloned_a.options[0].id
    assert cloned_b_default.depends[0].option == cloned_b.options[1].id
    assert a_go.depends[0].option == "go" and b_default.depends[0].option == "default"


def test_nested_flows_spelling_a_node_id_alike_keep_their_own_paths():
    inner_start = Input(id="start", name="start")
    inner_calc = Pass(
        id="calc",
        name="calc",
        depends=[NodeDependency(node=inner_start)],
        input_transformer=InputTransformer(selector={"x": "$.start.output.x"}),
    )
    inner_end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=inner_calc)],
        input_transformer=InputTransformer(selector={"x": "$.calc.output.x"}),
    )
    outer_start = Input(id="start", name="start")
    inner = SubWorkflow(
        id="inner",
        name="inner",
        flow=Flow(id="inner-flow", nodes=[inner_start, inner_calc, inner_end]),
        depends=[NodeDependency(node=outer_start)],
        input_transformer=InputTransformer(selector={"x": "$.start.output.x"}),
    )
    outer_end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=inner)],
        input_transformer=InputTransformer(selector={"x": "$.start.output.x", "inner": "$.inner.output.x"}),
    )
    # The copied node's own selector reads the flow around it, which the copy does not carry.
    node = SubWorkflow(
        id="outer",
        name="outer",
        flow=Flow(id="outer-flow", nodes=[outer_start, inner, outer_end]),
        input_transformer=InputTransformer(selector={"x": "$.start.output.x"}),
    )

    clone = regenerate_node_ids(node.clone(), {})

    cloned_outer_start, cloned_inner, cloned_outer_end = clone.flow.nodes
    cloned_inner_start, cloned_inner_calc, _ = cloned_inner.flow.nodes
    assert cloned_inner_start.id != cloned_outer_start.id
    assert cloned_inner_calc.input_transformer.selector == {"x": f'$."{cloned_inner_start.id}".output.x'}
    assert cloned_inner.input_transformer.selector == {"x": f'$."{cloned_outer_start.id}".output.x'}
    assert cloned_outer_end.input_transformer.selector == {
        "x": f'$."{cloned_outer_start.id}".output.x',
        "inner": f'$."{cloned_inner.id}".output.x',
    }
    assert clone.input_transformer.selector == {"x": "$.start.output.x"}


def test_a_choice_condition_naming_a_node_by_id_follows_the_new_id():
    start = Input(id="start", name="start")
    route = Choice(
        id="route",
        name="route",
        options=[
            ChoiceOption(
                id="opt-hi",
                condition=ChoiceCondition(
                    operands=[
                        ChoiceCondition(
                            operator=ConditionOperator.NUMERIC_GREATER_THAN, variable="$.start.output.score", value=50
                        )
                    ],
                    operator=ConditionOperator.AND,
                ),
            ),
            ChoiceOption(id="opt-lo"),
        ],
        depends=[NodeDependency(node=start)],
    )
    node = SubWorkflow(id="sub", name="sub", flow=Flow(id="flow", nodes=[start, route]))

    clone = regenerate_node_ids(node.clone(), {})

    cloned_start, cloned_route = clone.flow.nodes
    assert cloned_start.id != "start"
    assert cloned_route.options[0].condition.operands[0].variable == f'$."{cloned_start.id}".output.score'
    assert route.options[0].condition.operands[0].variable == "$.start.output.score"


def test_a_dependency_condition_and_an_output_transformer_read_results_not_nodes():
    """A node named `output` or `content` must not pull along the paths that read a result's keys."""
    start = Input(id="input", name="input")
    content = Pass(id="content", name="content", depends=[NodeDependency(node=start)])
    end = Pass(
        id="output",
        name="output",
        depends=[
            NodeDependency(
                node=content,
                condition=ChoiceCondition(
                    operator=ConditionOperator.NUMERIC_GREATER_THAN, variable="$.output.score", value=5
                ),
            )
        ],
        input_transformer=InputTransformer(selector={"score": "$.content.output.score"}),
        output_transformer=OutputTransformer(selector={"answer": "$.content"}),
    )
    node = SubWorkflow(id="sub", name="sub", flow=Flow(id="flow", nodes=[start, content, end]))

    clone = regenerate_node_ids(node.clone(), {})

    _, cloned_content, cloned_end = clone.flow.nodes
    assert cloned_content.id != "content"
    assert cloned_end.depends[0].condition.variable == "$.output.score"
    assert cloned_end.output_transformer.selector == {"answer": "$.content"}
    assert cloned_end.input_transformer.selector == {"score": f'$."{cloned_content.id}".output.score'}


def test_a_flow_copy_gives_each_dependency_its_own_condition():
    start = Pass(id="start", name="start")
    end = Pass(
        id="end",
        name="end",
        depends=[
            NodeDependency(
                node=start,
                condition=ChoiceCondition(
                    operator=ConditionOperator.NUMERIC_GREATER_THAN, variable="$.output.score", value=5
                ),
            )
        ],
    )
    flow = Flow(id="flow", nodes=[start, end])

    copied = flow.clone()

    copied_end = copied.nodes[1]
    assert copied_end.depends[0].node is copied.nodes[0]
    assert copied_end.depends[0].condition == end.depends[0].condition
    assert copied_end.depends[0].condition is not end.depends[0].condition
    copied_end.depends[0].condition.variable = "$.output.other"
    assert end.depends[0].condition.variable == "$.output.score"


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

    # The column keeps its id and the option carries a new one, but `$.fico` and `$.query` name input
    # keys, not nodes.
    cloned_table, cloned_route = clone.flow.nodes
    assert cloned_table.input_columns[0].id == "fico"
    assert cloned_route.options[0].id != "query"
    assert cloned_table.input_transformer.selector == {"fico": "$.fico"}
    assert cloned_route.input_transformer.selector == {"q": "$.query"}


def test_a_rule_a_row_and_a_field_keep_the_ids_the_user_wrote_where_the_node_gets_a_new_one():
    table = DecisionTable(
        id="pricing",
        input_columns=[NamedField(id="col-1", name="tier")],
        output_columns=[NamedField(id="col-2", name="rate")],
        rules=[DecisionRule(id="r-1", when=["gold"], then=[0.1])],
    )
    checks = Rules(
        id="review",
        input_fields=[NamedField(id="f-1", name="claim")],
        derived_values=[DerivedValue(id="d-1", name="total", expression="claim.amount")],
        rules=[Rule(id="POL-01", check="claim.amount < 1000")],
    )

    id_map: dict[str, set[str]] = {}
    table_copy = regenerate_node_ids(table.clone(), id_map)
    checks_copy = regenerate_node_ids(checks.clone(), id_map)

    assert table_copy.id != "pricing" and checks_copy.id != "review"
    assert set(id_map) == {"pricing", "review"}
    assert [rule.id for rule in table_copy.rules] == ["r-1"]
    assert [column.id for column in table_copy.input_columns + table_copy.output_columns] == ["col-1", "col-2"]
    assert [rule.id for rule in checks_copy.rules] == ["POL-01"]
    assert [field.id for field in checks_copy.input_fields] == ["f-1"]
    assert [value.id for value in checks_copy.derived_values] == ["d-1"]
    assert table_copy.to_dict(for_tracing=True)["rules"][0]["id"] == "r-1"


def test_the_bracket_path_form_follows_the_new_id_on_every_pass():
    start = Input(id="start", name="start")
    calc = Pass(
        id="calc",
        name="calc",
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(
            selector={
                "single": "$['start'].output.score",
                "double": '$["start"].output.score',
                "bare": "$[start].output.score",
                "longer": "$['starter'].output.score",
            }
        ),
    )
    node = SubWorkflow(id="sub", name="sub", flow=Flow(id="flow", nodes=[start, calc]))

    clone = regenerate_node_ids(node.clone(), {})

    cloned_start, cloned_calc = clone.flow.nodes
    renamed = f"$['{cloned_start.id}'].output.score"
    assert cloned_calc.input_transformer.selector == {
        "single": renamed,
        "double": renamed,
        "bare": renamed,
        "longer": "$['starter'].output.score",
    }
    assert calc.input_transformer.selector["single"] == "$['start'].output.score"

    second = regenerate_node_ids(clone.clone(), {})
    second_start, second_calc = second.flow.nodes
    assert second_calc.input_transformer.selector["single"] == f"$['{second_start.id}'].output.score"


def test_a_dotted_path_follows_the_new_id_wherever_the_id_ends():
    start = Input(id="start", name="start")
    calc = Pass(
        id="calc",
        name="calc",
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(
            selector={
                "braces": "{{$.start.output.score}}",
                "bare_braces": "{{$.start}}",
                "bracket_after": "$.start['output'].score",
                "quoted": '"$.start"',
                "underscore": "$.start_2.output.score",
                "dash": "$.start-2.output.score",
            }
        ),
    )
    node = SubWorkflow(id="sub", name="sub", flow=Flow(id="flow", nodes=[start, calc]))

    clone = regenerate_node_ids(node.clone(), {})

    cloned_start, cloned_calc = clone.flow.nodes
    new = f'$."{cloned_start.id}"'
    assert cloned_calc.input_transformer.selector == {
        "braces": "{{" + new + ".output.score}}",
        "bare_braces": "{{" + new + "}}",
        "bracket_after": new + "['output'].score",
        "quoted": '"' + new + '"',
        "underscore": "$.start_2.output.score",
        "dash": "$.start-2.output.score",
    }


def test_a_sub_workflow_returning_every_inner_output_keeps_its_own_selector_wired():
    inner_start = Input(id="inner_start", name="inner_start")
    inner_calc = Expression(
        id="inner_calc",
        name="inner_calc",
        depends=[NodeDependency(node=inner_start)],
        input_fields=[NamedField(name="x")],
        expressions=[ExpressionItem(key="doubled", expression="x * 2")],
        input_transformer=InputTransformer(selector={"x": "$.inner_start.output.x"}),
    )
    # No Output node: the sub-workflow returns every inner node's output keyed by id, which its own
    # output transformer then selects from.
    keyed = SubWorkflow(
        id="keyed",
        name="keyed",
        flow=Flow(id="keyed_flow", nodes=[inner_start, inner_calc]),
        output_transformer=OutputTransformer(selector={"result": "$.inner_calc.doubled"}),
    )
    content = Pass(id="content", name="content")
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=content)],
        input_transformer=InputTransformer(selector={"content": "$.content.output.text"}),
    )
    # One Output node: the sub-workflow returns its output, whose keys are fields, not node ids.
    single = SubWorkflow(
        id="single",
        name="single",
        flow=Flow(id="single_flow", nodes=[content, end]),
        output_transformer=OutputTransformer(selector={"answer": "$.content"}),
    )

    keyed_clone = regenerate_node_ids(keyed.clone(), {})
    single_clone = regenerate_node_ids(single.clone(), {})

    _, cloned_calc = keyed_clone.flow.nodes
    assert keyed_clone.output_transformer.selector == {"result": f'$."{cloned_calc.id}".doubled'}
    cloned_content, cloned_end = single_clone.flow.nodes
    assert single_clone.output_transformer.selector == {"answer": "$.content"}
    assert cloned_end.input_transformer.selector == {"content": f'$."{cloned_content.id}".output.text'}
