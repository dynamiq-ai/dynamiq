"""Templates run in Jinja's sandbox: they read the values they are given, not the Python objects behind them."""

import pytest
from jinja2.sandbox import SecurityError
from pydantic import BaseModel, PrivateAttr

from dynamiq.nodes.tools.human_feedback import HumanFeedbackAction, HumanFeedbackTool
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.transformers.text import TextTemplate
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.feedback import ApprovalConfig, ApprovalInputData, FeedbackMethod
from dynamiq.types.mocking import MockConfig


class Account(BaseModel):
    owner: str = "Ada"
    _token: str = PrivateAttr(default="tok-123")

    def close(self) -> str:
        return "closed"

    close.alters_data = True


def render(source: str, **values) -> str:
    return Prompt(messages=[Message(content=source)]).format_messages(**values)[0]["content"]


WORKING_TEMPLATES = [
    (
        "{% for doc in docs %}{{ loop.index }}. {{ doc.title | upper }}\n{% endfor %}",
        {"docs": [{"title": "a"}]},
        "1. A\n",
    ),
    ("{{ record._id }} {{ record.__typename }}", {"record": {"_id": "42", "__typename": "Lead"}}, "42 Lead"),
    ("{{ account.owner }}", {"account": Account()}, "Ada"),
    (
        "{% set seen = [] %}{% for x in xs %}{% set _ = seen.append(x * 2) %}{% endfor %}{{ seen }}",
        {"xs": [1, 2]},
        "[2, 4]",
    ),
    (
        "{% set sum = namespace(of=0) %}{% for x in xs %}{% set sum.of = sum.of + x %}{% endfor %}{{ sum.of }}",
        {"xs": [1, 2]},
        "3",
    ),
    ("{% for i in range(3) %}{{ i }}{% endfor %}", {}, "012"),
    (
        "{{ '%s has %d' | format(name, count) }} {{ '{} items'.format(count) }}",
        {"name": "cart", "count": 2},
        "cart has 2 2 items",
    ),
    ("{{ data | tojson }} {{ text.split(',') | join('|') }}", {"data": {"a": 1}, "text": "x,y"}, '{"a": 1} x|y'),
    ("{{ price * quantity }}", {"price": 3, "quantity": 2}, "6"),
]


@pytest.mark.parametrize(("source", "values", "expected"), WORKING_TEMPLATES)
def test_templates_keep_working_on_the_values_they_are_given(source, values, expected):
    assert render(source, **values) == expected


@pytest.mark.parametrize(
    "source",
    [
        "{{ account._token }}",
        "{{ account.__class__ }}",
        "{{ account.__dict__ }}",
        "{{ text.__class__ }}",
    ],
)
def test_a_template_cannot_read_what_is_private_to_an_object(source):
    assert render(source, account=Account(), text="hello") == ""


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("{% if account._token %}has a token{% else %}no token{% endif %}", "no token"),
        ("{{ account._token | default('none') }}", "none"),
        ("{{ '{0._token}|{0.owner}'.format(account) }}", "|Ada"),
    ],
)
def test_what_is_private_counts_as_missing_in_conditions_defaults_and_format_strings(source, expected):
    assert render(source, account=Account()) == expected


def test_a_template_cannot_reach_other_objects_through_a_private_attribute():
    with pytest.raises(SecurityError):
        render("{{ text.__class__.__name__ }}", text="hello")


def test_a_template_cannot_call_a_method_marked_as_altering_data():
    with pytest.raises(SecurityError):
        render("{{ account.close() }}", account=Account())


def test_a_template_cannot_build_a_huge_range():
    with pytest.raises(OverflowError):
        render("{% for i in range(1000000) %}{% endfor %}")


def test_an_approval_message_cannot_read_what_is_private_to_its_input(mocker):
    tool = Python(
        name="charge",
        code="def run(input_data):\n    return {}",
        approval=ApprovalConfig(
            enabled=True,
            feedback_method=FeedbackMethod.CONSOLE,
            msg_template="{{ name }} for {{ input_data.amount }}{{ input_data.__class__ }}",
        ),
    )
    send = mocker.patch.object(
        Python,
        "send_console_approval_message",
        return_value=ApprovalInputData(feedback="", is_approved=True, data={}),
    )

    tool.get_approved_data_or_origin(input_data={"amount": 42}, config=RunnableConfig())

    assert send.call_args.args[0] == "charge for 42"


def test_a_message_to_a_person_cannot_read_what_is_private_to_its_input(mocker):
    mocker.patch("builtins.print")
    tool = HumanFeedbackTool(
        input_method=FeedbackMethod.CONSOLE,
        output_method=FeedbackMethod.CONSOLE,
        action=HumanFeedbackAction.INFO,
        msg_template="Note: {{ input }}{{ input.__class__ }}",
    )

    result = tool.run(input_data={"input": "{{ total }}"})

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output["content"] == "Message sent: Note: {{ total }}"


def test_a_text_template_node_cannot_read_what_is_private_to_its_input():
    node = TextTemplate(template="{{ account.owner }}{{ account._token }}")

    result = node.run(input_data={"account": Account()})

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output["content"] == "Ada"


def test_a_mocked_output_cannot_read_what_is_private_to_its_input():
    mock = MockConfig(enabled=True, output="would charge {{ input.owner }}{{ input._token }}")

    content = mock.render(node_name="charge", node_id="charge-1", input_data=Account())["content"]

    assert content == "[MOCKED] would charge Ada"
