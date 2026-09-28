"""The Rules node next to the nodes a workflow built in the editor already has: an agent with a
response format, a document converter, validators and detectors, a Python function, a Choice."""

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.converters import UnstructuredFileConverter
from dynamiq.nodes.detectors import PIIDetector
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators import Choice, ChoiceOption, Expression, Rules
from dynamiq.nodes.tools import Python
from dynamiq.nodes.types import ChoiceCondition, ConditionOperator, DerivedValue, ExpressionItem, NamedField, Rule
from dynamiq.nodes.utils import Input, Output
from dynamiq.nodes.validators import RegexMatch
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.document import Document
from dynamiq.types.mocking import MockConfig


def llm() -> OpenAI:
    return OpenAI(model="gpt-4o-mini", connection=connections.OpenAI(api_key="test"))


def run(workflow: Workflow, input_data: dict):
    return workflow.run(input_data=input_data, config=RunnableConfig(callbacks=[]))


def by_rule(output: dict) -> dict[str, dict]:
    return {finding["rule_id"]: finding for finding in output["findings"]}


# --- Support QA: an agent grades a conversation, the rules decide what the grade means ------------


def grade(**over) -> dict:
    return {
        "greeting": True,
        "resolution": "resolved",
        "empathy": 4,
        "pii_in_reply": False,
        "sentiment": "neutral",
        "escalated": False,
        "handle_time_s": 420,
        **over,
    }


def support_qa_workflow(graded: dict) -> Workflow:
    """Input → Agent (response format, mocked) → Rules → Choice → Agent coaching note, only when needed."""
    start = Input(id="start", name="start")
    grader = Agent(
        id="grader",
        name="grader",
        llm=llm(),
        response_format={"type": "json_schema", "json_schema": {"name": "grade", "schema": {"type": "object"}}},
        mock=MockConfig(enabled=True, output={"content": graded}),
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"input": "$.start.output.transcript"}),
    )
    qa = Rules(
        id="qa",
        name="qa",
        input_fields=[NamedField(name="grade"), NamedField(name="ticket"), NamedField(name="policy")],
        rules=[
            Rule(id="QA-01", name="Customer greeted", severity="warn", check="grade.greeting", reason_code="QA-01"),
            Rule(
                id="QA-02",
                name="Ticket resolved or escalated",
                check="grade.resolution in ['resolved', 'escalated']",
                reason_code="QA-02",
            ),
            Rule(
                id="QA-03",
                name="Empathy at or above the bar",
                severity="warn",
                check="grade.empathy >= policy.min_empathy",
                message="Empathy {{ grade.empathy }} of {{ policy.min_empathy }}",
                reason_code="QA-03",
            ),
            Rule(id="QA-04", name="No personal data in the reply", check="not grade.pii_in_reply", reason_code="QA-04"),
            Rule(
                id="QA-05",
                name="Handled within the channel SLA",
                severity="warn",
                check="grade.handle_time_s <= policy.sla_s[ticket.channel]",
                message="{{ grade.handle_time_s }}s on {{ ticket.channel }}, SLA {{ policy.sla_s[ticket.channel] }}s",
                reason_code="QA-05",
            ),
            Rule(
                id="QA-06",
                name="Unhappy customer escalated",
                applies_when="grade.sentiment == 'negative'",
                check="grade.escalated",
                reason_code="QA-06",
            ),
        ],
        depends=[NodeDependency(node=grader)],
        input_transformer=InputTransformer(
            selector={
                "grade": "$.grader.output.content",
                "ticket": "$.start.output.ticket",
                "policy": "$.start.output.policy",
            }
        ),
    )
    reasons = Expression(
        id="reasons",
        name="reasons",
        input_fields=[NamedField(name="findings")],
        expressions=[
            ExpressionItem(
                key="codes",
                expression=(
                    "findings | selectattr('status', 'in', ['fail', 'warn']) | map(attribute='reason_code') | list"
                ),
            ),
            ExpressionItem(
                key="brief",
                expression=(
                    "'Coach on: ' ~ (findings | selectattr('status', 'in', ['fail', 'warn'])"
                    " | map(attribute='name') | join('; '))"
                ),
            ),
        ],
        depends=[NodeDependency(node=qa)],
        input_transformer=InputTransformer(selector={"findings": "$.qa.output.findings"}),
    )
    route = Choice(
        id="route",
        name="route",
        options=[
            ChoiceOption(
                id="clean",
                name="clean",
                condition=ChoiceCondition(operator=ConditionOperator.STRING_EQUALS, variable="$.status", value="pass"),
            ),
            ChoiceOption(id="coach", name="coach"),
        ],
        depends=[NodeDependency(node=qa)],
        input_transformer=InputTransformer(selector={"status": "$.qa.output.status"}),
    )
    coach = Agent(
        id="coach",
        name="coach",
        llm=llm(),
        mock=MockConfig(enabled=True, output={"content": "Coaching note: acknowledge frustration before the fix."}),
        depends=[NodeDependency(node=route, option="coach"), NodeDependency(node=reasons)],
        input_transformer=InputTransformer(selector={"input": "$.reasons.output.brief"}),
    )
    return Workflow(flow=Flow(nodes=[start, grader, qa, reasons, route, coach]))


TICKET = {
    "transcript": "Customer: my card was blocked abroad...",
    "ticket": {"channel": "chat", "priority": "high"},
    "policy": {"min_empathy": 3, "sla_s": {"chat": 600, "email": 86400}},
}


def test_a_clean_conversation_passes_the_rules_and_the_coach_is_skipped():
    result = run(support_qa_workflow(grade()), TICKET)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["qa"]["output"]["status"] == "pass"
    assert result.output["reasons"]["output"]["codes"] == []
    assert result.output["coach"]["status"] == RunnableStatus.SKIP


def test_an_unhappy_customer_left_unescalated_with_a_slow_reply_reaches_the_coach():
    graded = grade(sentiment="negative", empathy=2, handle_time_s=900, greeting=False)
    result = run(support_qa_workflow(graded), TICKET)

    findings = by_rule(result.output["qa"]["output"])
    assert result.output["qa"]["output"]["status"] == "fail"
    assert findings["QA-06"]["status"] == "fail" and findings["QA-01"]["status"] == "warn"
    assert findings["QA-05"]["message"] == "900s on chat, SLA 600s"
    assert findings["QA-03"]["message"] == "Empathy 2 of 3"
    assert result.output["reasons"]["output"]["codes"] == ["QA-01", "QA-03", "QA-05", "QA-06"]
    assert result.output["reasons"]["output"]["brief"].startswith("Coach on: Customer greeted; Empathy")
    assert result.output["coach"]["status"] == RunnableStatus.SUCCESS
    assert result.output["coach"]["output"]["content"].startswith("Coaching note")


def test_a_grade_the_agent_did_not_produce_holds_the_rule_instead_of_passing_it():
    graded = grade()
    del graded["handle_time_s"]
    result = run(support_qa_workflow(graded), TICKET)

    findings = by_rule(result.output["qa"]["output"])
    assert findings["QA-05"]["status"] == "not_evaluated"
    assert findings["QA-05"]["message"] == "missing value for grade.handle_time_s"
    assert result.output["qa"]["output"]["status"] == "not_evaluated"
    assert result.output["coach"]["status"] == RunnableStatus.SUCCESS


# --- Document processing: a converter's documents are checked as a file ---------------------------


def documents(*docs: dict) -> list[Document]:
    return [Document(content=doc.pop("content", "text"), metadata=doc) for doc in docs]


def file_check_workflow(docs: list[Document]) -> Workflow:
    """Input → Unstructured converter (mocked) → Rules over the documents → Output."""
    start = Input(id="start", name="start")
    convert = UnstructuredFileConverter(
        id="convert",
        name="convert",
        connection=connections.Unstructured(api_key="test", url="https://unstructured.test"),
        mock=MockConfig(enabled=True, output={"documents": docs}),
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"files": "$.start.output.files"}),
    )
    checks = Rules(
        id="file_check",
        name="file_check",
        input_fields=[NamedField(name="documents"), NamedField(name="as_of")],
        derived_values=[
            DerivedValue(
                name="appraisal", expression="documents | selectattr('metadata.type', 'equalto', 'appraisal') | first"
            ),
            DerivedValue(name="pages", expression="documents | map(attribute='metadata.pages') | sum"),
        ],
        rules=[
            Rule(id="FILE-01", name="The file holds at least one document", check="(documents | length) > 0"),
            Rule(
                id="FILE-02",
                name="Every document has readable text",
                check="(documents | rejectattr('content') | list | length) == 0",
                message=(
                    "{{ documents | rejectattr('content') | map(attribute='metadata.file_path') | join(', ') }}"
                    " without text"
                ),
            ),
            Rule(id="FILE-03", name="An appraisal is in the file", check="has(appraisal)"),
            Rule(
                id="FILE-04",
                name="Appraisal dated within 120 days",
                check="days_between(appraisal.metadata.effective_date, as_of) <= 120",
                message="Appraisal is {{ days_between(appraisal.metadata.effective_date, as_of) }} days old",
            ),
            Rule(id="FILE-05", name="File within the review size", severity="warn", check="pages <= 200"),
        ],
        depends=[NodeDependency(node=convert)],
        input_transformer=InputTransformer(
            selector={"documents": "$.convert.output.documents", "as_of": "$.start.output.as_of"}
        ),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=checks)],
        input_transformer=InputTransformer(
            selector={"status": "$.file_check.output.status", "findings": "$.file_check.output.findings"}
        ),
    )
    return Workflow(flow=Flow(nodes=[start, convert, checks, end]))


def test_a_complete_file_of_converted_documents_passes():
    docs = documents(
        {"file_path": "note.pdf", "type": "note", "pages": 4},
        {"file_path": "appraisal.pdf", "type": "appraisal", "pages": 30, "effective_date": "2026-08-01"},
    )
    result = run(file_check_workflow(docs), {"files": ["note.pdf", "appraisal.pdf"], "as_of": "2026-09-19"})

    output = result.output["end"]["output"]
    assert output["status"] == "pass"
    assert {finding["status"] for finding in output["findings"]} == {"pass"}


def test_a_scanned_page_without_text_and_a_stale_appraisal_are_named():
    docs = documents(
        {"content": "", "file_path": "note.pdf", "type": "note", "pages": 4},
        {"file_path": "appraisal.pdf", "type": "appraisal", "pages": 30, "effective_date": "2026-03-01"},
    )
    result = run(file_check_workflow(docs), {"files": ["note.pdf", "appraisal.pdf"], "as_of": "2026-09-19"})

    findings = by_rule(result.output["end"]["output"])
    assert findings["FILE-02"]["message"] == "note.pdf without text"
    assert findings["FILE-04"]["message"] == "Appraisal is 202 days old"
    assert result.output["end"]["output"]["status"] == "fail"


def test_a_file_without_an_appraisal_fails_the_presence_rule_and_holds_the_dated_one():
    docs = documents({"file_path": "note.pdf", "type": "note", "pages": 4})
    result = run(file_check_workflow(docs), {"files": ["note.pdf"], "as_of": "2026-09-19"})

    findings = by_rule(result.output["end"]["output"])
    assert findings["FILE-03"]["status"] == "fail"
    assert findings["FILE-04"]["status"] == "not_evaluated"


# --- Validators, a detector and a Python function feed the rules --------------------------------------


TENURE = """
def run(input_data):
    from datetime import date
    opened = date.fromisoformat(input_data["customer"]["account_opened"])
    as_of = date.fromisoformat(input_data["as_of"])
    return {"tenure_months": (as_of.year - opened.year) * 12 + as_of.month - opened.month}
"""


def screening_workflow(pii_detected: bool) -> Workflow:
    """Input → RegexMatch + PIIDetector (mocked) + Python → Rules → Output."""
    start = Input(id="start", name="start")
    passport = RegexMatch(
        id="passport",
        name="passport",
        regex=r"^[A-Z]{2}\d{6}$",
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"content": "$.start.output.customer.passport_no"}),
    )
    pii = PIIDetector(
        id="pii",
        name="pii",
        mock=MockConfig(
            enabled=True,
            output={"is_detected": pii_detected, "detected_pii": ["EMAIL_ADDRESS"] if pii_detected else []},
        ),
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"message": "$.start.output.note"}),
    )
    tenure = Python(
        id="tenure",
        name="tenure",
        code=TENURE,
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(
            selector={"customer": "$.start.output.customer", "as_of": "$.start.output.as_of"}
        ),
    )
    screen = Rules(
        id="screen",
        name="screen",
        input_fields=[
            NamedField(name="passport"),
            NamedField(name="pii"),
            NamedField(name="tenure"),
            NamedField(name="limits"),
        ],
        rules=[
            Rule(id="ID-01", name="Passport number well formed", check="passport.valid", reason_code="ID-01"),
            Rule(
                id="PRIV-01",
                name="No personal data in the free-text note",
                check="not pii.is_detected",
                message="{{ pii.detected_pii | join(', ') }} in the note",
                reason_code="PRIV-01",
            ),
            Rule(
                id="TEN-01",
                name="Tenure long enough for the limit increase",
                severity="warn",
                check="tenure.content.tenure_months >= limits.min_tenure_months",
                message="{{ tenure.content.tenure_months }} months, {{ limits.min_tenure_months }} required",
                reason_code="TEN-01",
            ),
        ],
        depends=[NodeDependency(node=passport), NodeDependency(node=pii), NodeDependency(node=tenure)],
        input_transformer=InputTransformer(
            selector={
                "passport": "$.passport.output",
                "pii": "$.pii.output",
                "tenure": "$.tenure.output",
                "limits": "$.start.output.limits",
            }
        ),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=screen)],
        input_transformer=InputTransformer(
            selector={"status": "$.screen.output.status", "findings": "$.screen.output.findings"}
        ),
    )
    return Workflow(flow=Flow(nodes=[start, passport, pii, tenure, screen, end]))


CUSTOMER = {
    "customer": {"passport_no": "AB123456", "account_opened": "2024-01-15"},
    "note": "Please raise my limit.",
    "as_of": "2026-09-19",
    "limits": {"min_tenure_months": 12},
}


def test_validator_detector_and_python_outputs_are_read_by_the_rules_as_they_are():
    result = run(screening_workflow(pii_detected=False), CUSTOMER)

    output = result.output["end"]["output"]
    assert output["status"] == "pass"
    assert by_rule(output)["TEN-01"]["evaluated"] == {
        "tenure.content.tenure_months": 32,
        "limits.min_tenure_months": 12,
    }


def test_a_malformed_passport_and_a_leaked_email_fail_the_screening():
    record = {**CUSTOMER, "customer": {"passport_no": "ab-1", "account_opened": "2026-01-15"}}
    result = run(screening_workflow(pii_detected=True), record)

    findings = by_rule(result.output["end"]["output"])
    assert findings["ID-01"]["status"] == "fail"
    assert findings["PRIV-01"]["message"] == "EMAIL_ADDRESS in the note"
    assert findings["TEN-01"]["message"] == "8 months, 12 required"
    assert result.output["end"]["output"]["status"] == "fail"
