"""What an agent does with braces in the text of a run: it fills the names of the run's own fields and nothing else."""

import pytest

from dynamiq.nodes.agents.base import _fill_field_names
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.prompts import (
    Message,
    MessageRole,
    Prompt,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)
from dynamiq.runnables import RunnableStatus
from tests.helpers.agent_hooks import ALL_MODES, build_agent, final_step, run_scripted, scripted, tool_step


class Account:
    """An object a caller may pass as a field: its attributes are not data to look up."""

    owner = "Ada"


JSX = 'Fix <div style={{ fontSize: "1.2rem" }} onClick={() => open({ id: 1 })} />'
HELM = "Explain {{ .Values.image.tag }} and {{- range .Values.hosts }}{{ . }}{{- end }}"

RUN_TEXT = [
    pytest.param(
        "Summarize {{topic}} for {{ audience }}",
        {"topic": "the Q3 report", "audience": "the board"},
        "Summarize the Q3 report for the board",
        id="the names of the run's fields are filled",
    ),
    pytest.param(
        "Order {{count}} of {{ sku }}", {"count": 3, "sku": "A-12"}, "Order 3 of A-12", id="a number is filled"
    ),
    pytest.param(
        "Review this line: {{snippet}}",
        {"snippet": "total = {{ price }} {% if vat %}+ vat{% endif %}"},
        "Review this line: total = {{ price }} {% if vat %}+ vat{% endif %}",
        id="a value holding braces is inserted as written",
    ),
    pytest.param(
        "What does {{ release }} mean in this chart?",
        {"locale": "en"},
        "What does {{ release }} mean in this chart?",
        id="a name that is no field of the run is kept",
    ),
    pytest.param("Repeat {{input}} twice", {"locale": "en"}, "Repeat {{input}} twice", id="{{input}} is kept"),
    pytest.param(
        "Greet {{ name | upper }} warmly", {"name": "ada"}, "Greet {{ name | upper }} warmly", id="a filter is kept"
    ),
    pytest.param(
        "Summarise {{ email.subject }} from {{email.sender.name}}",
        {"email": {"subject": "Invoice 42 is overdue", "sender": {"name": "Ada"}}},
        "Summarise Invoice 42 is overdue from Ada",
        id="a key inside a field is filled",
    ),
    pytest.param(
        "Reply to {{ email.to.0 }} about {{ orders.1.id }}",
        {"email": {"to": ["ada@example.com", "bo@example.com"]}, "orders": [{"id": "A-1"}, {"id": "A-2"}]},
        "Reply to ada@example.com about A-2",
        id="an item of a list inside a field is filled",
    ),
    pytest.param(
        "Copy {{ email.cc }} and {{ email.to.5 }}",
        {"email": {"to": ["ada@example.com"]}},
        "Copy {{ email.cc }} and {{ email.to.5 }}",
        id="a key or an item that is missing is kept",
    ),
    pytest.param(
        "How long is {{ topic.length }}?",
        {"topic": "pricing"},
        "How long is {{ topic.length }}?",
        id="a path through text is kept",
    ),
    pytest.param(
        "Charge {{ account.owner }}",
        {"account": Account()},
        "Charge {{ account.owner }}",
        id="an attribute of an object is kept",
    ),
    pytest.param(
        "{% if topic %}Cover {{ topic }}{% else %}Cover everything{% endif %}",
        {"topic": "pricing"},
        "{% if topic %}Cover pricing{% else %}Cover everything{% endif %}",
        id="a block is kept around the field it names",
    ),
    pytest.param(
        "Why does {% if user %} fail to parse",
        {"locale": "en"},
        "Why does {% if user %} fail to parse",
        id="an unclosed block",
    ),
    pytest.param(
        "{# draft #} Summarize the thread", {"locale": "en"}, "{# draft #} Summarize the thread", id="a comment is kept"
    ),
    pytest.param(JSX, {"locale": "en"}, JSX, id="JSX"),
    pytest.param(HELM, {"locale": "en"}, HELM, id="a Helm template"),
    pytest.param("Summarize {{topic}}", {}, "Summarize {{topic}}", id="a run with no other fields is sent as typed"),
]


def sent_as_user(prompt):
    """The content of the user message as the LLM node sends it, which is where a message is rendered last."""
    message = next(message for message in prompt if message.role == MessageRole.USER)
    return Prompt(messages=[message]).format_messages()[0]["content"]


FILLED = [
    ("{{topic}}", {"topic": "pricing"}, "pricing"),
    ("{{ topic }} and {{topic}}", {"topic": "pricing"}, "pricing and pricing"),
    ("{{first}}{{second}}", {"first": "a", "second": "b"}, "ab"),
    ("{{ count }} items", {"count": 3}, "3 items"),
    ("{{ note }}", {"note": None}, "None"),
    ("{{ тема }}", {"тема": "ціни"}, "ціни"),
    ("{{ email.subject }}", {"email": {"subject": "Invoice 42"}}, "Invoice 42"),
    ("{{ email.to.1 }}", {"email": {"to": ["ada@example.com", "bo@example.com"]}}, "bo@example.com"),
    ("{{ orders.0.lines.1.sku }}", {"orders": [{"lines": [{"sku": "A"}, {"sku": "B"}]}]}, "B"),
    ("{{ snippet }}", {"snippet": "{{ topic }} {% if x %}"}, "{{ topic }} {% if x %}"),
]
KEPT = [
    ("{{ topic }}", {}),
    ("{{ release }}", {"topic": "pricing"}),
    ("{{- topic -}}", {"topic": "pricing"}),
    ("{{ topic | upper }}", {"topic": "pricing"}),
    ("{{ topic.upper() }}", {"topic": "pricing"}),
    ("{{ topic.length }}", {"topic": "pricing"}),
    ('{{ email["subject"] }}', {"email": {"subject": "Invoice 42"}}),
    ("{{ email.cc }}", {"email": {"subject": "Invoice 42"}}),
    ("{{ email.to.2 }}", {"email": {"to": ["ada@example.com"]}}),
    ("{{ email.to.-1 }}", {"email": {"to": ["ada@example.com"]}}),
    ("{{ account.owner }}", {"account": Account()}),
    ("{{ 2 + 2 }}", {"topic": "pricing"}),
    ("{% if topic %}yes{% endif %}", {"topic": "pricing"}),
    ("{# topic #}", {"topic": "pricing"}),
    ("{ topic }", {"topic": "pricing"}),
]


@pytest.mark.parametrize(("text", "fields", "expected"), FILLED)
def test_a_name_or_a_path_into_data_is_filled(text, fields, expected):
    assert _fill_field_names(text, fields) == expected


@pytest.mark.parametrize(("text", "fields"), KEPT)
def test_everything_else_is_kept_as_typed(text, fields):
    assert _fill_field_names(text, fields) == text


@pytest.mark.parametrize(("text", "fields", "expected"), RUN_TEXT)
def test_the_input_of_a_run(text, fields, expected):
    agent = build_agent(tools=[])
    mode = agent.inference_mode

    result, recorder = run_scripted(agent, mode, [final_step("ok")], question=text, input_extra=fields)

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert sent_as_user(recorder.prompts[0]) == expected


@pytest.mark.parametrize(("text", "fields", "expected"), RUN_TEXT)
def test_a_message_passed_with_a_run(text, fields, expected):
    agent = build_agent(tools=[])

    with scripted(agent, agent.inference_mode, [final_step("ok")]) as recorder:
        result = agent.run(
            input_data={"input": "", **fields}, input_message=Message(role=MessageRole.USER, content=text)
        )

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert sent_as_user(recorder.prompts[0]) == expected


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("task", [JSX, HELM, "Find {{ competitor }} prices", "{% if urgent %}Hurry{% endif %}"])
def test_a_task_a_parent_agent_writes_for_a_sub_agent_reaches_it_as_written(mode, task):
    child = build_agent(id="child", name="researcher", tools=[])
    parent = build_agent(tools=[SubAgentTool(agent=child, name="Researcher", description="researches")])

    with scripted(child, mode, [final_step("found")]) as child_recorder:
        result, _ = run_scripted(
            parent, mode, [tool_step("Researcher", brief="Research", input=task), final_step("ok")]
        )

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert sent_as_user(child_recorder.prompts[0]) == task


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (
            {"input": "keep it short", "topic": "pricing", "audience": "sales"},
            "Summarize pricing for SALES: keep it short",
        ),
        (
            {"input": "{{ topic | upper }}", "topic": "pricing", "audience": "sales"},
            "Summarize pricing for SALES: {{ topic | upper }}",
        ),
    ],
    ids=["its fields and the input are filled", "the input is a value, never part of the template"],
)
def test_an_input_message_configured_on_the_agent_is_a_full_template(fields, expected):
    template = Message(role=MessageRole.USER, content="Summarize {{topic}} for {{ audience | upper }}: {{input}}")
    agent = build_agent(tools=[], input_message=template)

    with scripted(agent, agent.inference_mode, [final_step("ok")]) as recorder:
        result = agent.run(input_data=fields)

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert sent_as_user(recorder.prompts[0]) == expected


def test_a_configured_input_message_is_rendered_when_the_run_passes_only_the_input():
    agent = build_agent(tools=[], input_message=Message(role=MessageRole.USER, content="Question: {{input}}"))

    with scripted(agent, agent.inference_mode, [final_step("ok")]) as recorder:
        result = agent.run(input_data={"input": "Where is order 42?"})

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert sent_as_user(recorder.prompts[0]) == "Question: Where is order 42?"


@pytest.mark.parametrize("fields", [{}, {"topic": "pricing"}], ids=["no other fields", "with another field"])
def test_a_configured_input_message_marked_static_is_sent_as_written(fields):
    written = "Keep {{topic}} and {{ input | upper }} as they are"
    agent = build_agent(tools=[], input_message=Message(role=MessageRole.USER, content=written, static=True))

    with scripted(agent, agent.inference_mode, [final_step("ok")]) as recorder:
        result = agent.run(input_data={"input": "hello", **fields})

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert sent_as_user(recorder.prompts[0]) == written


@pytest.mark.parametrize("static", [True, False], ids=["marked static", "not marked static"])
def test_a_configured_vision_message_fills_its_image_address(static):
    template = VisionMessage(
        static=static,
        content=[
            VisionMessageTextContent(text="Describe the {{ subject }} in this picture"),
            VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{ image }}")),
        ],
    )
    agent = build_agent(tools=[], input_message=template)

    with scripted(agent, agent.inference_mode, [final_step("ok")]) as recorder:
        result = agent.run(input_data={"subject": "bridge", "image": "https://example.com/bridge.png"})

    assert result.status == RunnableStatus.SUCCESS, result.error
    text = "Describe the {{ subject }} in this picture" if static else "Describe the bridge in this picture"
    assert sent_as_user(recorder.prompts[0]) == [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": "https://example.com/bridge.png", "detail": "auto"}},
    ]


def test_a_message_passed_with_a_run_and_marked_static_is_sent_as_written():
    written = "Keep {{topic}} as it is"
    agent = build_agent(tools=[])

    with scripted(agent, agent.inference_mode, [final_step("ok")]) as recorder:
        result = agent.run(
            input_data={"input": "", "topic": "pricing"},
            input_message=Message(role=MessageRole.USER, content=written, static=True),
        )

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert sent_as_user(recorder.prompts[0]) == written
