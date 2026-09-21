import datetime
import decimal
import json
import re
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm import ModelResponse

from dynamiq import connections
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.connections import TypeSafe
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.detectors import SystemOne
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import Judgement, JudgementOption, JudgementQuestion, Python
from dynamiq.nodes.tools.judgement import confidence_of
from dynamiq.nodes.types import NamedField
from dynamiq.runnables import RunnableConfig, RunnableStatus

SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"


def questions() -> list[JudgementQuestion]:
    return [
        JudgementQuestion(
            id="q1",
            name="is_urgent",
            type="noul",
            instructions="The customer needs an answer today",
            yes_when="A deadline or a blocked payment",
            no_when="A question with no time pressure",
        ),
        JudgementQuestion(
            id="q2",
            name="team",
            type="choice",
            instructions="Which team should handle this?",
            options=[
                JudgementOption(name="billing", description="Charges and refunds"),
                JudgementOption(name="technical", description="Errors and outages"),
                JudgementOption(name="sales"),
            ],
        ),
        JudgementQuestion(
            id="q3",
            name="anger",
            type="score",
            instructions="How upset is the customer?",
            options=[JudgementOption(name="calm"), JudgementOption(name="frustrated"), JudgementOption(name="furious")],
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


def http(status: int = 200, body: dict | None = None, headers: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.text = json.dumps(body) if body is not None else ""
    response.headers = headers or {}
    return response


def llm_reply(payload: dict | str) -> ModelResponse:
    response = ModelResponse()
    response["choices"][0]["message"]["content"] = payload if isinstance(payload, str) else json.dumps(payload)
    return response


class UsageRecorder(BaseCallbackHandler):
    def __init__(self):
        self.usage = []

    def on_node_execute_run(self, serialized, **kwargs):
        if usage := kwargs.get("usage_data"):
            self.usage.append((serialized["name"], usage))


@pytest.fixture
def post(mocker):
    return mocker.patch("requests.post", return_value=http(body=system_one_answers()))


@pytest.fixture
def system_one() -> Judgement:
    return Judgement(
        id="triage",
        name="triage",
        judge=SystemOne(connection=TypeSafe(api_key="k")),
        input_fields=[NamedField(name="ticket")],
        questions=questions(),
        min_confidence=0.7,
    )


@pytest.fixture
def llm() -> OpenAI:
    return OpenAI(connection=connections.OpenAI(api_key="k"), model="gpt-4o-mini", is_postponed_component_init=True)


def test_system_one_receives_the_state_and_the_questions_in_wire_form(system_one, post):
    result = system_one.run(input_data={"ticket": {"subject": "Charged twice", "body": "Fix it today"}})

    assert result.status == RunnableStatus.SUCCESS
    post.assert_called_once()
    assert post.call_args.args == (SYSTEM_ONE_URL,)
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer k"}
    assert post.call_args.kwargs["timeout"] == 30
    assert post.call_args.kwargs["json"] == {
        "model": "jev-latest",
        "state": {"ticket": {"subject": "Charged twice", "body": "Fix it today"}},
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "The customer needs an answer today",
                "criteria": {"true": "A deadline or a blocked payment", "false": "A question with no time pressure"},
            },
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Charges and refunds", "technical": "Errors and outages", "sales": "sales"},
            },
            "anger": {
                "type": "score",
                "instructions": "How upset is the customer?",
                "criteria": ["calm", "frustrated", "furious"],
            },
        },
    }


def test_system_one_answers_become_decisions_confidence_and_usage(system_one, post):
    recorder = UsageRecorder()

    result = system_one.run(input_data={"ticket": "Charged twice"}, config=RunnableConfig(callbacks=[recorder]))

    output = result.output
    assert output["decisions"] == {"is_urgent": True, "team": "billing", "anger": "frustrated"}
    # A yes/no answer carries no confidence on the wire; it is the two-outcome case of the same measure.
    assert output["answers"]["is_urgent"] == {
        "type": "noul",
        "probability": 0.92,
        "decision": True,
        "confidence": pytest.approx(0.84),
    }
    assert output["answers"]["team"]["confidence"] == 0.76
    assert output["answers"]["team"]["probabilities"] == pytest.approx(
        {"billing": 0.84, "technical": 0.159, "sales": 0.001}
    )
    anger = output["answers"]["anger"]
    assert (anger["level"], anger["index"], anger["score"], anger["confidence"]) == ("frustrated", 1, 1.035, 0.6475)
    assert anger["probabilities"] == pytest.approx({"calm": 0.1, "frustrated": 0.765, "furious": 0.135})
    assert output["confidence"] == 0.6475
    assert output["low_confidence"] == ["anger"]
    assert output["needs_review"] is True
    assert (output["model"], output["backend"], output["confidence_source"]) == ("jev-1.13.0", "system_one", "model")
    assert output["usage"] == {"input_tokens": 1200, "output_tokens": 0, "cost_usd": pytest.approx(0.0000504)}
    assert output["rationale"] is None and output["evidence"] is None
    assert "- team: billing (confidence 0.76; billing 0.84, technical 0.16, sales 0.00)" in output["content"]
    assert "Needs review (confidence below 0.70): anger" in output["content"]
    # Reported the way LLM nodes report it, so the platform's cost tracking needs no special case.
    assert recorder.usage == [
        (
            "triage",
            {
                "prompt_tokens": 1200,
                "completion_tokens": 0,
                "total_tokens": 1200,
                "prompt_tokens_cost_usd": pytest.approx(0.0000504),
                "completion_tokens_cost_usd": 0.0,
                "total_tokens_cost_usd": pytest.approx(0.0000504),
            },
        )
    ]


def test_call_time_questions_replace_configured_ones_by_name_and_disabled_ones_are_not_asked(system_one, post):
    system_one.questions[2].enabled = False
    post.return_value = http(
        body=system_one_answers(
            team={"type": "choice", "choice": "support", "probabilities": {"support": 0.9, "sales": 0.1}},
            refund={"type": "noul", "noul": 0.2},
        )
    )

    result = system_one.run(
        input_data={
            "ticket": "Charged twice",
            "questions": [
                {
                    "name": "team",
                    "type": "choice",
                    "instructions": "Support or sales?",
                    "options": [{"name": "support"}, {"name": "sales"}],
                },
                {"name": "refund", "type": "noul", "instructions": "The customer asks for money back"},
            ],
        }
    )

    assert result.status == RunnableStatus.SUCCESS
    sent = post.call_args.kwargs["json"]["questions"]
    assert list(sent) == ["is_urgent", "team", "refund"]
    assert sent["team"]["criteria"] == {"support": "support", "sales": "sales"}
    assert result.output["decisions"] == {"is_urgent": True, "team": "support", "refund": False}
    assert result.output["answers"]["team"]["confidence"] == pytest.approx(0.8)


def test_the_state_is_the_named_inputs_unless_given_directly(system_one, post):
    system_one.run(input_data={"ticket": "Charged twice", "customer": "not declared, not sent"})
    assert post.call_args.kwargs["json"]["state"] == {"ticket": "Charged twice"}

    system_one.run(input_data={"state": ["Hi", "Charged twice"], "ticket": "ignored"})
    assert post.call_args.kwargs["json"]["state"] == ["Hi", "Charged twice"]

    system_one.run(input_data={})
    assert post.call_args.kwargs["json"]["state"] == {"ticket": None}

    bare = Judgement(judge=SystemOne(connection=TypeSafe(api_key="k")), questions=questions())
    result = bare.run(input_data={})
    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable is True
    assert "nothing to judge" in result.error.message


@pytest.mark.parametrize(
    ("status", "body", "recoverable", "fragment"),
    [
        (401, {"error": {"message": "invalid api key"}}, False, "rejected the API key (HTTP 401): invalid api key"),
        (
            422,
            {"detail": "state exceeds 32k tokens"},
            True,
            "rejected the request (HTTP 422): state exceeds 32k tokens",
        ),
        (418, None, True, "HTTP 418"),
    ],
)
def test_api_errors_say_what_happened_and_whether_asking_again_can_help(
    system_one, post, status, body, recoverable, fragment
):
    post.return_value = http(status, body)

    result = system_one.run(input_data={"ticket": "x"})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable is recoverable
    assert fragment in result.error.message


def test_rate_limits_are_retried_as_the_server_asks(system_one, post, mocker):
    sleep = mocker.patch("dynamiq.nodes.detectors.system_one.time.sleep")
    post.side_effect = [
        http(429, headers={"retry-after-ms": "250"}),
        http(529, headers={"Retry-After": "2"}),
        http(body=system_one_answers()),
    ]

    result = system_one.run(input_data={"ticket": "x"})

    assert result.status == RunnableStatus.SUCCESS
    assert post.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.25, 2.0]


def test_an_outage_is_given_up_after_the_last_retry(system_one, post, mocker):
    sleep = mocker.patch("dynamiq.nodes.detectors.system_one.time.sleep")
    post.side_effect = [http(529), http(529), http(529)]

    result = system_one.run(input_data={"ticket": "x"})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable is True
    assert "HTTP 529" in result.error.message
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0]


def test_a_malformed_answer_is_a_recoverable_failure_naming_the_question(system_one, post):
    post.return_value = http(body=system_one_answers(is_urgent={"type": "noul", "noul": 1.4}))
    result = system_one.run(input_data={"ticket": "x"})
    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable is True
    assert "answer to 'is_urgent' is unreadable: probability 1.4 is outside 0..1" in result.error.message

    body = system_one_answers()
    del body["answers"]["team"]
    post.return_value = http(body=body)
    result = system_one.run(input_data={"ticket": "x"})
    assert "answer to 'team' is unreadable: no answer" in result.error.message


def test_a_state_beyond_the_context_limit_fails_before_the_request(system_one, post):
    result = system_one.run(input_data={"ticket": "x" * 200_000})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable is True
    assert "tokens; System One reads at most 32,000" in result.error.message
    post.assert_not_called()


def test_agent_questions_can_be_hidden_from_the_tool_schema():
    fixed = Judgement(
        judge=SystemOne(connection=TypeSafe(api_key="k")), questions=questions(), allow_agent_questions=False
    )
    open_ = Judgement(judge=SystemOne(connection=TypeSafe(api_key="k")), questions=questions())

    assert fixed.resolved_input_schema.model_fields["questions"].json_schema_extra == {"is_accessible_to_agent": False}
    assert open_.resolved_input_schema.model_fields["questions"].json_schema_extra is None
    assert fixed.input_param_modes == {}
    # Only the agent-facing schema hides them: a workflow may still pass questions from upstream.
    validated = fixed.validate_input_schema({"state": "x", "questions": [{"name": "q", "instructions": "i"}]})
    assert validated.questions[0].name == "q"


@pytest.mark.parametrize(
    ("fields", "fragment"),
    [
        (lambda llm: {}, "Field required"),
        (
            lambda llm: {"judge": Python(code="def run(_):\n    return {}")},
            "must be a System One, an LLM or an agent node",
        ),
        (
            lambda llm: {
                "judge": SystemOne(connection=TypeSafe(api_key="k")),
                "confidence_mode": "sampling",
                "samples": 3,
            },
            "sampling needs an LLM or agent judge",
        ),
        (lambda llm: {"judge": llm, "confidence_mode": "sampling"}, "sampling needs at least 2 samples"),
        (
            lambda llm: {"judge": llm, "questions": [questions()[0], questions()[0]]},
            "question names used more than once: is_urgent",
        ),
    ],
)
def test_the_node_refuses_a_configuration_that_could_not_judge(llm, fields, fragment):
    with pytest.raises(ValueError, match=re.escape(fragment)):
        Judgement(**fields(llm))


@pytest.mark.parametrize(
    ("fields", "fragment"),
    [
        ({"name": "the claim", "instructions": "x"}, "String should match pattern"),
        ({"name": "q", "instructions": "  "}, "instructions must not be empty"),
        (
            {"name": "q", "type": "choice", "instructions": "x", "options": [{"name": "a"}]},
            "between 2 and 255 options, got 1",
        ),
        (
            {"name": "q", "type": "score", "instructions": "x", "options": [{"name": str(i)} for i in range(11)]},
            "between 2 and 10 levels, got 11",
        ),
        (
            {"name": "q", "type": "choice", "instructions": "x", "options": [{"name": "a"}, {"name": "a"}]},
            "names an option twice",
        ),
        (
            {"name": "q", "type": "score", "instructions": "x", "options": [{"name": "a"}, {"name": " "}]},
            "has an unnamed level",
        ),
    ],
)
def test_a_question_is_checked_when_it_is_written(fields, fragment):
    with pytest.raises(ValueError, match=re.escape(fragment)):
        JudgementQuestion(**fields)


LLM_VERDICT = {
    "is_urgent": {"answer": True, "probability": 0.9, "rationale": "It says today"},
    "team": {
        "answer": "Billing",
        "probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1},
        "rationale": "A charge",
    },
    "anger": {
        "answer": "frustrated",
        "probabilities": {"calm": 0.2, "frustrated": 0.6, "furious": 0.2},
        "rationale": "Civil",
    },
}


def test_an_llm_judge_gets_a_schema_built_from_the_questions_and_states_its_probabilities(llm, mocker):
    completion = mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", return_value=llm_reply(LLM_VERDICT))
    node = Judgement(judge=llm, questions=questions(), include_rationale=True, min_confidence=0.5)

    result = node.run(input_data={"state": "Charged twice, fix it today"})

    assert result.status == RunnableStatus.SUCCESS
    params = completion.call_args.kwargs
    schema = params["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == ["is_urgent", "team", "anger"]
    assert schema["properties"]["team"]["properties"]["answer"]["enum"] == ["billing", "technical", "sales"]
    assert schema["properties"]["team"]["properties"]["probabilities"]["required"] == ["billing", "technical", "sales"]
    assert list(schema["properties"]["is_urgent"]["properties"]) == ["answer", "probability", "rationale"]
    brief = params["messages"][0]["content"]
    assert "<state>\nCharged twice, fix it today\n</state>" in brief
    assert "2. team (choose one): Which team should handle this?\n   - billing: Charges and refunds" in brief
    assert "   Yes when: A deadline or a blocked payment" in brief

    output = result.output
    assert output["decisions"] == {"is_urgent": True, "team": "billing", "anger": "frustrated"}
    assert output["answers"]["is_urgent"]["confidence"] == pytest.approx(0.8)
    assert output["answers"]["team"]["confidence"] == pytest.approx(0.55)
    assert output["rationale"] == {"is_urgent": "It says today", "team": "A charge", "anger": "Civil"}
    assert output["low_confidence"] == ["anger"]
    assert (output["model"], output["backend"], output["confidence_source"]) == (
        "openai/gpt-4o-mini",
        "llm",
        "verbalized",
    )
    assert output["usage"] is None and output["evidence"] is None


def test_a_verbalized_answer_stands_on_its_probabilities_when_the_judge_omits_or_rewords_it(llm, mocker):
    """An agent judge answers the schema as prose, so `answer` may be missing or phrased its own way."""
    mocker.patch(
        "dynamiq.nodes.llms.base.BaseLLM._completion",
        return_value=llm_reply(
            {
                "is_urgent": {"probability": 0.9},
                "team": {"answer": "Billing.", "probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1}},
                "anger": {"probabilities": {"calm": 0.2, "frustrated": 0.7, "furious": 0.1}},
            }
        ),
    )
    node = Judgement(judge=llm, questions=questions())

    result = node.run(input_data={"state": "Charged twice"})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["decisions"] == {"is_urgent": True, "team": "billing", "anger": "frustrated"}


def test_probability_keys_are_matched_the_way_the_answer_is(llm, mocker):
    """A judge that capitalizes its keys states the same distribution, not a certain one."""
    mocker.patch(
        "dynamiq.nodes.llms.base.BaseLLM._completion",
        return_value=llm_reply(
            {"team": {"answer": "Billing", "probabilities": {"Billing": 0.55, " TECHNICAL ": 0.45, "Sales": 0}}}
        ),
    )
    node = Judgement(judge=llm, questions=questions()[1:2])

    output = node.run(input_data={"state": "Charged twice"}).output

    assert output["decisions"] == {"team": "billing"}
    assert output["answers"]["team"]["probabilities"] == {"billing": 0.55, "technical": 0.45, "sales": 0.0}
    assert output["answers"]["team"]["confidence"] == pytest.approx(0.325)


def test_a_verbalized_answer_is_still_needed_when_the_probabilities_cannot_stand_alone(llm, mocker):
    replies = [
        ({"team": {"answer": "Billing", "probabilities": {"billing": 0, "technical": 0, "sales": 0}}}, "billing"),
        ({"team": {"probabilities": {"billing": 0, "technical": 0, "sales": 0}}}, "no usable probabilities"),
        ({"team": {"answer": "Accounts", "probabilities": {}}}, "'Accounts' is not one of billing"),
    ]
    node = Judgement(judge=llm, questions=questions()[1:2])

    for payload, expected in replies:
        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", return_value=llm_reply(payload))
        result = node.run(input_data={"state": "Charged twice"})
        if expected == "billing":
            assert result.output["decisions"] == {"team": "billing"}
        else:
            assert result.status == RunnableStatus.FAILURE
            assert expected in result.error.message


def test_sampling_turns_repeated_answers_into_vote_shares(llm, mocker):
    votes = [("billing", True), ("billing", False), ("technical", True), ("billing", False)]
    completion = mocker.patch(
        "dynamiq.nodes.llms.base.BaseLLM._completion",
        side_effect=[
            llm_reply({"is_urgent": {"answer": urgent}, "team": {"answer": team}, "anger": {"answer": "calm"}})
            for team, urgent in votes
        ],
    )
    node = Judgement(judge=llm, questions=questions(), confidence_mode="sampling", samples=4)

    result = node.run(input_data={"state": "Charged twice"})

    assert result.status == RunnableStatus.SUCCESS
    assert completion.call_count == 4
    schema = completion.call_args.kwargs["response_format"]["json_schema"]["schema"]
    assert "probabilities" not in schema["properties"]["team"]["properties"]
    output = result.output
    assert output["answers"]["team"]["probabilities"] == {"billing": 0.75, "technical": 0.25, "sales": 0.0}
    assert output["answers"]["team"]["confidence"] == pytest.approx(0.625)
    assert output["answers"]["is_urgent"]["probability"] == 0.5
    assert output["decisions"]["is_urgent"] is True
    assert output["confidence_source"] == "sampling"


def test_every_run_gets_its_own_judge_so_two_callers_cannot_share_one(llm, mocker):
    """`is_parallel_execution_allowed` lets a calling agent invoke this node twice at once, and a
    judge resets its per-run state on every execute - an agent wipes its loop state and rebuilds its
    prompt - so two calls sharing one judge would interleave into a single conversation."""
    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", return_value=llm_reply(LLM_VERDICT))
    judges = []
    run = OpenAI.run

    def record(self, *args, **kwargs):
        judges.append(self)
        return run(self, *args, **kwargs)

    mocker.patch.object(OpenAI, "run", record)
    node = Judgement(judge=llm, questions=questions())

    assert node.run(input_data={"state": "Charged twice"}).status == RunnableStatus.SUCCESS
    assert node.run(input_data={"state": "Charged twice"}).status == RunnableStatus.SUCCESS

    assert len(judges) == 2
    assert judges[0] is not judges[1]
    assert all(judge is not node.judge for judge in judges)
    assert [judge.model for judge in judges] == [node.judge.model] * 2


def test_a_state_the_transport_could_not_encode_is_rendered_the_way_the_size_check_measured_it(system_one, post):
    """The size check serializes with `default=str`, so a datetime measures fine. Sending the raw
    object would raise TypeError inside requests - neither a RequestException nor an httpx error,
    so it would escape the retry loop and surface as an encoder traceback."""
    result = system_one.run(
        input_data={"ticket": {"opened": datetime.datetime(2026, 9, 21, 14, 30), "total": decimal.Decimal("42.50")}}
    )

    assert result.status == RunnableStatus.SUCCESS
    sent = post.call_args.kwargs["json"]
    assert sent["state"] == {"ticket": {"opened": "2026-09-21 14:30:00", "total": "42.50"}}
    json.dumps(sent)  # what requests does internally; a raw datetime raises here


def test_the_system_one_judge_also_runs_on_its_own(post):
    """It is a node, not a helper the Judgement reaches into: run standalone it sends the same request
    and returns the service's answers unread, for the caller to interpret."""
    node = SystemOne(connection=TypeSafe(api_key="k"))

    result = node.run(input_data={"state": "Charged twice", "questions": {"is_urgent": {"type": "noul"}}})

    assert result.status == RunnableStatus.SUCCESS
    assert post.call_args.kwargs["json"] == {
        "model": "jev-latest",
        "state": "Charged twice",
        "questions": {"is_urgent": {"type": "noul"}},
    }
    assert result.output["model"] == "jev-1.13.0"
    assert result.output["answers"]["is_urgent"] == {"type": "noul", "noul": 0.92}
    assert result.output["usage"] == {
        "input_tokens": 1200,
        "output_tokens": 0,
        "cost_usd": pytest.approx(0.0000504),
    }


def test_a_judge_that_does_not_answer_with_json_is_a_recoverable_failure(llm, mocker):
    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", return_value=llm_reply("I would rather not say."))
    node = Judgement(judge=llm, questions=questions())

    result = node.run(input_data={"state": "x"})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable is True
    assert "did not answer with JSON" in result.error.message


BRIEFS: list[str] = []


class Lookup(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str | None = "lookup"
    description: str = "Looks an order up."

    def execute(self, input_data, config=None, **kwargs):
        return {"content": "Order A-104 was charged twice on 2026-09-01."}


class StubAgent(Agent):
    """Consults its tool and answers as an agent would, with no LLM in the loop."""

    def execute(self, input_data, config=None, **kwargs):
        BRIEFS.append(input_data.input)
        self.tools[0].run(input_data={}, config=config, parent_run_id=kwargs.get("run_id"))
        return {
            "content": {
                "is_urgent": {"answer": "yes", "probability": 0.95},
                "team": {"answer": "billing", "probabilities": {"billing": 1, "technical": 0, "sales": 0}},
                "anger": {"answer": "calm", "probabilities": {"calm": 0.5, "frustrated": 0.5, "furious": 0}},
            }
        }


def test_an_agent_judge_reads_the_brief_and_its_tool_calls_come_back_as_evidence(llm):
    node = Judgement(judge=StubAgent(llm=llm, tools=[Lookup()]), questions=questions())

    result = node.run(input_data={"state": {"order": "A-104"}})

    assert result.status == RunnableStatus.SUCCESS
    assert '"order": "A-104"' in BRIEFS[-1]
    assert "1. is_urgent (yes/no): The customer needs an answer today" in BRIEFS[-1]
    output = result.output
    assert output["evidence"] == [{"tool": "lookup", "output": "Order A-104 was charged twice on 2026-09-01."}]
    assert output["decisions"] == {"is_urgent": True, "team": "billing", "anger": "calm"}
    assert output["answers"]["anger"]["confidence"] == pytest.approx(0.25)
    assert (output["backend"], output["model"]) == ("agent", "openai/gpt-4o-mini")


class FakeAsyncClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_the_async_path_retries_and_answers_like_the_sync_one(system_one, post, mocker):
    client = FakeAsyncClient([http(429, headers={"retry-after-ms": "10"}), http(body=system_one_answers())])
    mocker.patch.object(TypeSafe, "connect_async", AsyncMock(return_value=client))
    sleep = mocker.patch("dynamiq.nodes.detectors.system_one.asyncio.sleep", AsyncMock())

    result = await system_one.run_async(input_data={"ticket": "Charged twice"})

    assert result.status == RunnableStatus.SUCCESS
    assert [call.args[0] for call in sleep.call_args_list] == [0.01]
    assert client.calls[0][0] == SYSTEM_ONE_URL
    assert client.calls[1][1]["json"]["state"] == {"ticket": "Charged twice"}
    assert result.output["decisions"] == {"is_urgent": True, "team": "billing", "anger": "frustrated"}
    assert result.output["usage"]["input_tokens"] == 1200


@pytest.mark.parametrize(
    ("probabilities", "confidence"),
    [
        ([1.0, 0.0, 0.0], 1.0),
        ([1 / 3, 1 / 3, 1 / 3], 0.0),
        ([0.5, 0.5], 0.0),
        ([0.84, 0.159, 0.001], 0.76),
        ([1.0], 1.0),
    ],
)
def test_confidence_is_how_far_the_top_answer_stands_out(probabilities, confidence):
    assert confidence_of(probabilities) == pytest.approx(confidence)


def test_a_clone_keeps_the_question_ids_and_gets_a_judge_of_its_own(llm):
    node = Judgement(judge=llm, questions=questions())

    clone = node.clone()

    assert [question.id for question in clone.questions] == ["q1", "q2", "q3"]
    assert clone.judge is not node.judge
    assert clone.judge.model == "openai/gpt-4o-mini"
