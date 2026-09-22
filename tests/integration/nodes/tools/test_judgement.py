import json
from unittest.mock import MagicMock

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import TypeSafe
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.detectors import SystemOne
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators import DecisionTable
from dynamiq.nodes.tools import Judgement, JudgementOption, JudgementQuestion
from dynamiq.nodes.types import DecisionRule, NamedField
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableConfig, RunnableStatus


def questions() -> list[JudgementQuestion]:
    return [
        JudgementQuestion(
            id="q1",
            name="is_urgent",
            type="noul",
            instructions="The customer needs an answer today",
            yes_when="A deadline or a blocked payment",
        ),
        JudgementQuestion(
            id="q2",
            name="team",
            type="choice",
            instructions="Which team should handle this?",
            options=[
                JudgementOption(id="o1", name="billing", description="Charges and refunds"),
                JudgementOption(id="o2", name="technical", description="Errors and outages"),
                JudgementOption(id="o3", name="sales"),
            ],
        ),
        JudgementQuestion(
            id="q3",
            name="anger",
            type="score",
            instructions="How upset is the customer?",
            options=[
                JudgementOption(id="l1", name="calm"),
                JudgementOption(id="l2", name="frustrated"),
                JudgementOption(id="l3", name="furious"),
            ],
        ),
    ]


def system_one_answers(**overrides) -> dict:
    answers = {
        "is_urgent": {"type": "noul", "noul": 0.92},
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.84, "technical": 0.159, "sales": 0.001},
            "confidence": 0.76,
        },
        "anger": {
            "type": "score",
            "score": 1.035,
            "probabilities": {"0": 0.1, "1": 0.765, "2": 0.135},
            "legend": {"0": "calm", "1": "frustrated", "2": "furious"},
            "confidence": 0.6475,
        },
    }
    answers.update(overrides)
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 1200, "output_tokens": 0}}


def http(body: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.text = json.dumps(body)
    response.headers = {}
    return response


@pytest.fixture
def post(mocker):
    return mocker.patch("requests.post", return_value=http(system_one_answers()))


def triage_workflow(judge: dict) -> Workflow:
    """A ticket is judged, the verdict is routed by a decision table, and an uncertain one goes to a person."""
    start = Input(id="start", name="start")
    triage = Judgement(
        id="triage",
        name="triage",
        input_fields=[NamedField(name="ticket")],
        questions=questions(),
        min_confidence=0.6,
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"ticket": "$.start.output.ticket"}),
        **judge,
    )
    route = DecisionTable(
        id="route",
        name="route",
        hit_policy="first",
        input_columns=[
            NamedField(id="review", name="review", type="bool"),
            NamedField(id="urgent", name="urgent", type="bool"),
            NamedField(id="team", name="team", type="string"),
            NamedField(id="anger", name="anger", type="string"),
        ],
        output_columns=[
            NamedField(id="queue", name="queue", type="string"),
            NamedField(id="priority", name="priority", type="string"),
        ],
        rules=[
            DecisionRule(id="r1", name="uncertain verdict", when=["true", "", "", ""], then=["human-review", "high"]),
            DecisionRule(id="r2", name="furious customer", when=["", "", "", "furious"], then=["escalations", "high"]),
            DecisionRule(id="r3", name="urgent billing", when=["", "true", "billing", ""], then=["billing", "high"]),
            DecisionRule(id="r4", name="billing", when=["", "", "billing", ""], then=["billing", "normal"]),
            DecisionRule(id="r5", name="everything else", when=["", "", "", ""], then=["general", "low"]),
        ],
        depends=[NodeDependency(node=triage)],
        input_transformer=InputTransformer(
            selector={
                "review": "$.triage.output.needs_review",
                "urgent": "$.triage.output.decisions.is_urgent",
                "team": "$.triage.output.decisions.team",
                "anger": "$.triage.output.decisions.anger",
            }
        ),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=route)],
        input_transformer=InputTransformer(
            selector={"queue": "$.route.output.queue", "priority": "$.route.output.priority"}
        ),
    )
    return Workflow(id="triage-workflow", flow=Flow(id="triage-flow", nodes=[start, triage, route, end]))


def test_a_judgement_feeds_a_decision_table_and_uncertain_verdicts_go_to_a_person(post, mock_tracing_client):
    tracing = TracingCallbackHandler(client=mock_tracing_client())
    workflow = triage_workflow({"judge": SystemOne(connection=TypeSafe(api_key="k"))})

    result = workflow.run(
        input_data={"ticket": "Charged twice, fix it today"}, config=RunnableConfig(callbacks=[tracing])
    )

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["end"]["output"] == {"queue": "billing", "priority": "high"}
    triage_run = next(run for run in tracing.runs.values() if run.name == "triage")
    assert triage_run.metadata["usage"]["prompt_tokens"] == 1200
    assert [question["name"] for question in triage_run.metadata["node"]["questions"]] == ["is_urgent", "team", "anger"]

    # The same ticket with the team split three ways: the confidence falls under the node's floor.
    post.return_value = http(
        system_one_answers(
            team={
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.4, "technical": 0.35, "sales": 0.25},
            }
        )
    )
    result = workflow.run(input_data={"ticket": "Charged twice, fix it today"})

    assert result.output["end"]["output"] == {"queue": "human-review", "priority": "high"}


def test_an_agent_calls_the_judgement_as_a_tool_with_questions_of_its_own(post):
    llm = OpenAI(connection=connections.OpenAI(api_key="k"), model="gpt-4o-mini", is_postponed_component_init=True)
    tool = Judgement(
        id="judge", name="judgement", judge=SystemOne(connection=TypeSafe(api_key="k")), questions=questions()[:2]
    )
    agent = Agent(llm=llm, tools=[tool])
    post.return_value = http(system_one_answers(refund={"type": "noul", "noul": 0.2}))

    content, files, meta = agent._run_tool(
        tool,
        {
            "state": "Charged twice, fix it today",
            "questions": [{"name": "refund", "type": "noul", "instructions": "The customer asks for money back"}],
        },
        RunnableConfig(),
    )

    assert content.startswith("Judgement by jev-1.13.0:")
    assert "- refund: no (probability of yes 0.20, confidence 0.60)" in content
    assert files == []
    assert meta["decisions"] == {"is_urgent": True, "team": "billing", "refund": False}
    assert list(post.call_args.kwargs["json"]["questions"]) == ["is_urgent", "team", "refund"]


def test_yaml_round_trip_keeps_the_judge_and_the_questions(tmp_path, post):
    llm = OpenAI(
        id="judge-llm",
        name="judge",
        connection=connections.OpenAI(id="openai", api_key="k"),
        model="gpt-4o-mini",
        is_postponed_component_init=True,
    )
    workflow = triage_workflow({"judge": llm})

    first = tmp_path / "triage.yaml"
    workflow.to_yaml_file(first)
    loaded = Workflow.from_yaml_file(str(first), init_components=True)
    second = tmp_path / "triage_again.yaml"
    loaded.to_yaml_file(second)
    reloaded = Workflow.from_yaml_file(str(second), init_components=True)

    triage = next(node for node in reloaded.flow.nodes if node.id == "triage")
    assert isinstance(triage.judge, OpenAI)
    assert (triage.judge.model, triage.judge.connection.api_key) == ("openai/gpt-4o-mini", "k")
    assert triage.min_confidence == 0.6
    assert [question.model_dump() for question in triage.questions] == [
        question.model_dump() for question in questions()
    ]
    assert "yes_when: A deadline or a blocked payment" in second.read_text()

    system_one = triage_workflow({"judge": SystemOne(connection=TypeSafe(id="typesafe", api_key="k"))})
    path = tmp_path / "system_one.yaml"
    system_one.to_yaml_file(path)
    reloaded = Workflow.from_yaml_file(str(path), init_components=True)

    triage = next(node for node in reloaded.flow.nodes if node.id == "triage")
    assert isinstance(triage.judge, SystemOne)
    assert (triage.judge.connection.url, triage.judge.connection.api_key, triage.judge.model) == (
        "https://api.typesafe.ai",
        "k",
        "jev-latest",
    )
    assert reloaded.run(input_data={"ticket": "Charged twice"}).output["end"]["output"] == {
        "queue": "billing",
        "priority": "high",
    }
