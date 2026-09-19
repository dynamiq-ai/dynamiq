import re

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators import Map, Rules
from dynamiq.nodes.types import DerivedValue, NamedField, Rule
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableConfig, RunnableStatus

LOAN = {
    "amount": 340000,
    "purpose": "purchase",
    "program": "FHA",
    "flood_zone": "AE",
    "funding_date": "2026-08-05",
    "property": {"appraised_value": 400000},
    "monthly_income": 9000,
    "monthly_debt": 3600,
}
DOCS = {
    "Note": {"interest_rate": 6.875, "date": "2026-08-01", "signed": True},
    "ClosingDisclosure": {"interest_rate": 6.75, "date": "2026-07-30"},
    "Appraisal": {"effective_date": "2026-03-15"},
}
LIMITS = {"FHA": {"max_dti": 0.43, "max_ltv": 0.965}, "Conventional": {"max_dti": 0.45, "max_ltv": 0.97}}


def pre_purchase_rules(**overrides) -> Rules:
    fields = {
        "id": "review",
        "name": "review",
        "input_fields": [NamedField(name="loan"), NamedField(name="docs"), NamedField(name="limits")],
        "derived_values": [
            DerivedValue(name="ltv", expression="loan.amount / loan.property.appraised_value"),
            DerivedValue(name="dti", expression="loan.monthly_debt / loan.monthly_income"),
        ],
        "rules": [
            Rule(
                id="CR-014",
                name="Note rate matches Closing Disclosure",
                category="Compliance",
                check="docs.Note.interest_rate == docs.ClosingDisclosure.interest_rate",
                message=(
                    "Note rate {{ docs.Note.interest_rate }}% differs from "
                    "Closing Disclosure rate {{ docs.ClosingDisclosure.interest_rate }}%"
                ),
                reason_code="RATE-CD-MISMATCH",
                references=["Policy §4.3.2"],
                tags=["FHA", "Conventional"],
            ),
            Rule(
                id="EL-032",
                name="DTI within program limit",
                category="Eligibility",
                check="dti <= limits[loan.program].max_dti",
            ),
            Rule(
                id="EL-040",
                name="LTV within program limit",
                category="Eligibility",
                severity="warn",
                check="ltv <= limits[loan.program].max_ltv",
                message="LTV {{ (ltv * 100) | round(1) }}%",
            ),
            Rule(
                id="DC-210",
                name="Flood certificate present in a flood zone",
                category="Documentation",
                applies_when="loan.flood_zone in ['A', 'AE', 'V']",
                check="has(docs.FloodCert)",
                message="Flood certificate missing for zone {{ loan.flood_zone }}",
            ),
            Rule(
                id="CR-102",
                name="Appraisal dated within 120 days of the note",
                category="Credit",
                severity="warn",
                check="days_between(docs.Appraisal.effective_date, docs.Note.date) <= 120",
            ),
            Rule(
                id="LG-017",
                name="Right to cancel period observed",
                category="Legal",
                applies_when="loan.purpose == 'refinance'",
                check="days_between(docs.ClosingDisclosure.date, loan.funding_date) >= 3",
            ),
            Rule(id="DC-118", name="Note is signed", category="Documentation", check="docs.Note.signed"),
            Rule(id="OLD-001", name="Retired check", check="false", effective_until="2025-12-31"),
            Rule(id="NEW-001", name="Future check", check="false", effective_from="2027-01-01"),
            Rule(id="OFF-001", name="Switched off", check="false", enabled=False),
        ],
    }
    return Rules(**(fields | overrides))


def run_node(node: Rules, input_data: dict):
    return node.run(input_data=input_data, config=RunnableConfig(callbacks=[]))


def statuses(output: dict) -> dict[str, str]:
    return {finding["rule_id"]: finding["status"] for finding in output["findings"]}


def test_every_rule_reports_a_status_and_the_summary_adds_up():
    result = run_node(pre_purchase_rules(), {"loan": LOAN, "docs": DOCS, "limits": LIMITS, "as_of": "2026-09-19"})

    assert result.status == RunnableStatus.SUCCESS
    output = result.output
    assert statuses(output) == {
        "CR-014": "fail",
        "EL-032": "pass",
        "EL-040": "pass",
        "DC-210": "fail",
        "CR-102": "warn",
        "LG-017": "not_applicable",
        "DC-118": "pass",
        "OLD-001": "not_applicable",
        "NEW-001": "not_applicable",
    }
    assert output["status"] == "fail"
    assert output["summary"] == {"pass": 3, "fail": 2, "warn": 1, "info": 0, "not_applicable": 3, "not_evaluated": 0}
    assert output["derived"] == {"ltv": 0.85, "dti": 0.4}
    # A switched-off rule is not a check, so it has no finding.
    assert "OFF-001" not in statuses(output)


def test_a_finding_carries_the_message_with_the_values_it_read():
    output = run_node(pre_purchase_rules(), {"loan": LOAN, "docs": DOCS, "limits": LIMITS}).output
    finding = next(item for item in output["findings"] if item["rule_id"] == "CR-014")

    assert finding == {
        "rule_id": "CR-014",
        "name": "Note rate matches Closing Disclosure",
        "category": "Compliance",
        "severity": "fail",
        "status": "fail",
        "message": "Note rate 6.875% differs from Closing Disclosure rate 6.75%",
        "reason_code": "RATE-CD-MISMATCH",
        "references": ["Policy §4.3.2"],
        "tags": ["FHA", "Conventional"],
        "evaluated": {"docs.Note.interest_rate": 6.875, "docs.ClosingDisclosure.interest_rate": 6.75},
    }
    flood = next(item for item in output["findings"] if item["rule_id"] == "DC-210")
    assert flood["message"] == "Flood certificate missing for zone AE"
    assert flood["evaluated"] == {"docs.FloodCert": None}
    ltv = next(item for item in output["findings"] if item["rule_id"] == "EL-040")
    assert (
        ltv["status"] == "pass"
        and ltv["message"] is None
        and ltv["evaluated"] == {"ltv": 0.85, "limits": "{…2 keys}", "loan.program": "FHA"}
    )


def test_a_missing_value_is_a_finding_to_review_not_a_silent_pass():
    docs = {key: value for key, value in DOCS.items() if key != "Appraisal"}
    loan = {key: value for key, value in LOAN.items() if key != "monthly_income"}

    output = run_node(pre_purchase_rules(), {"loan": loan, "docs": docs, "limits": LIMITS}).output

    by_id = {item["rule_id"]: item for item in output["findings"]}
    assert by_id["CR-102"]["status"] == "not_evaluated"
    assert by_id["CR-102"]["message"] == "missing value for docs.Appraisal.effective_date"
    # The derived value could not be computed, so the rule that reads it is a review item too.
    assert output["derived"]["dti"] is None
    assert by_id["EL-032"]["status"] == "not_evaluated"
    assert by_id["EL-032"]["message"] == "missing value for dti"
    assert output["summary"]["not_evaluated"] == 2
    assert output["status"] == "fail"


def test_on_missing_fail_reports_the_severity_with_the_reason():
    docs = {key: value for key, value in DOCS.items() if key != "Appraisal"}

    output = run_node(pre_purchase_rules(on_missing="fail"), {"loan": LOAN, "docs": docs, "limits": LIMITS}).output

    finding = next(item for item in output["findings"] if item["rule_id"] == "CR-102")
    assert finding["status"] == "warn"
    assert finding["message"] == "missing value for docs.Appraisal.effective_date"


def test_a_null_value_counts_as_missing():
    docs = {**DOCS, "ClosingDisclosure": {**DOCS["ClosingDisclosure"], "interest_rate": None}}

    output = run_node(pre_purchase_rules(), {"loan": LOAN, "docs": docs, "limits": LIMITS}).output

    assert statuses(output)["CR-014"] == "not_evaluated"


def test_a_check_that_cannot_be_evaluated_is_a_finding_to_review():
    docs = {**DOCS, "Appraisal": {"effective_date": "March"}}

    output = run_node(pre_purchase_rules(), {"loan": LOAN, "docs": docs, "limits": LIMITS}).output

    finding = next(item for item in output["findings"] if item["rule_id"] == "CR-102")
    assert finding["status"] == "not_evaluated"
    assert finding["message"].startswith("check could not be evaluated: not a date: 'March'")


def test_applies_when_gates_the_rule_and_the_effective_window_follows_as_of():
    refinance = {**LOAN, "purpose": "refinance", "flood_zone": "X"}

    output = run_node(
        pre_purchase_rules(), {"loan": refinance, "docs": DOCS, "limits": LIMITS, "as_of": "2027-02-01"}
    ).output

    found = statuses(output)
    assert found["LG-017"] == "pass"
    assert found["DC-210"] == "not_applicable"
    assert found["NEW-001"] == "fail"
    assert found["OLD-001"] == "not_applicable"


def test_as_of_must_be_a_date():
    result = run_node(pre_purchase_rules(), {"loan": LOAN, "docs": DOCS, "limits": LIMITS, "as_of": "yesterday"})

    assert result.status == RunnableStatus.FAILURE
    assert "'as_of' is not a date" in result.error.message


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"rules": [Rule(id="a", name="dup", check="true"), Rule(id="a", name="again", check="true")]},
            "rule 2 (again): id 'a' is used twice",
        ),
        ({"rules": [Rule(id="a", name="blank", check="   ")]}, "rule 1 (blank): the check is empty"),
        (
            {"rules": [Rule(id="a", name="broken", check="loan.amount >")]},
            "rule 1 (broken): the check is not a valid expression",
        ),
        (
            {"rules": [Rule(id="a", name="gate", applies_when="loan.purpose ==", check="true")]},
            "rule 1 (gate): applies_when is not a valid expression",
        ),
        (
            {"rules": [Rule(id="a", name="dated", check="true", effective_from="soon")]},
            "rule 1 (dated): effective_from is not a date: 'soon'",
        ),
        (
            {
                "rules": [
                    Rule(id="a", name="window", check="true", effective_from="2026-02-01", effective_until="2026-01-01")
                ]
            },
            "rule 1 (window): the effective window ends before it starts",
        ),
        (
            {"rules": [Rule(id="a", name="template", check="true", message="{{ loan.amount")]},
            "rule 1 (template): the message is not a valid template",
        ),
        (
            {"derived_values": [DerivedValue(name="loan", expression="1")]},
            "derived value 'loan' is already the name of an input or a helper",
        ),
        (
            {"derived_values": [DerivedValue(name="2fast", expression="1")]},
            "derived value '2fast' is not a valid identifier",
        ),
        (
            {"derived_values": [DerivedValue(name="ratio", expression="loan.amount /")]},
            "derived value 'ratio' is not a valid expression",
        ),
    ],
)
def test_a_malformed_rule_fails_at_build_naming_the_rule(overrides, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        pre_purchase_rules(**overrides)


def test_trace_keeps_the_first_rules_and_the_count():
    rules = [Rule(id=f"r{index}", name=f"rule {index}", check="true") for index in range(60)]
    node = pre_purchase_rules(rules=rules)
    tracing = TracingCallbackHandler()

    node.run(input_data={"loan": LOAN, "docs": DOCS, "limits": LIMITS}, config=RunnableConfig(callbacks=[tracing]))

    traced = next(run for run in tracing.runs.values() if run.name == "review").metadata["node"]
    assert len(traced["rules"]) == 50
    assert traced["rules_count"] == 60
    assert len(node.to_dict()["rules"]) == 60
    assert "rules_count" not in node.to_dict()


def review_workflow() -> Workflow:
    start = Input(id="start", name="start")
    review = pre_purchase_rules(
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(
            selector={
                "loan": "$.start.output.loan",
                "docs": "$.start.output.docs",
                "limits": "$.start.output.limits",
                "as_of": "$.start.output.as_of",
            }
        ),
    )
    end = Output(id="end", name="end", depends=[NodeDependency(node=review)])
    return Workflow(id="workflow", flow=Flow(id="flow", nodes=[start, review, end]))


def test_a_flow_maps_the_record_by_name_and_the_output_node_returns_the_findings():
    result = review_workflow().run(
        input_data={"loan": LOAN, "docs": DOCS, "limits": LIMITS, "as_of": "2026-09-19"},
        config=RunnableConfig(callbacks=[]),
    )

    assert result.status == RunnableStatus.SUCCESS
    review = result.output["review"]["output"]
    assert review["status"] == "fail"
    assert statuses(review)["CR-014"] == "fail"


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "rules.yaml"
    review_workflow().to_yaml_file(path)

    loaded = Workflow.from_yaml_file(str(path), init_components=True)
    node = next(node for node in loaded.flow.nodes if isinstance(node, Rules))

    assert [rule.id for rule in node.rules][:3] == ["CR-014", "EL-032", "EL-040"]
    assert node.rules[0].references == ["Policy §4.3.2"]
    assert node.rules[0].tags == ["FHA", "Conventional"]
    assert [value.name for value in node.derived_values] == ["ltv", "dti"]
    assert node.rules[7].effective_until == "2025-12-31"
    result = loaded.run(
        input_data={"loan": LOAN, "docs": DOCS, "limits": LIMITS, "as_of": "2026-09-19"},
        config=RunnableConfig(callbacks=[]),
    )
    assert statuses(result.output["review"]["output"]) == statuses(
        review_workflow()
        .run(
            input_data={"loan": LOAN, "docs": DOCS, "limits": LIMITS, "as_of": "2026-09-19"},
            config=RunnableConfig(callbacks=[]),
        )
        .output["review"]["output"]
    )


def test_the_same_record_gives_the_same_findings_every_run():
    node = pre_purchase_rules()
    record = {"loan": LOAN, "docs": DOCS, "limits": LIMITS, "as_of": "2026-09-19"}

    outputs = [run_node(node, record).output for _ in range(3)]

    assert outputs[0] == outputs[1] == outputs[2]


def test_a_map_reviews_a_batch_of_records_in_parallel():
    batch = Map(id="batch", name="batch", node=pre_purchase_rules(), max_workers=4)
    clean_docs = {
        **DOCS,
        "ClosingDisclosure": {**DOCS["ClosingDisclosure"], "interest_rate": 6.875},
        "FloodCert": {"present": True},
        "Appraisal": {"effective_date": "2026-07-01"},
    }
    records = [
        {"loan": LOAN, "docs": DOCS, "limits": LIMITS, "as_of": "2026-09-19"},
        {"loan": LOAN, "docs": clean_docs, "limits": LIMITS, "as_of": "2026-09-19"},
    ] * 3

    result = batch.run(input_data={"input": records}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert [item["status"] for item in result.output["output"]] == ["fail", "pass"] * 3


def test_document_check_answers_feed_a_rule_with_a_confidence_threshold():
    node = Rules(
        id="signature",
        name="signature",
        input_fields=[NamedField(name="doc_checks")],
        rules=[
            Rule(
                id="DR-003",
                name="Borrower signature on the Note",
                applies_when="doc_checks.note_signed_confidence >= 0.85",
                check="doc_checks.note_signed",
                message="No borrower signature found on the Note (confidence {{ doc_checks.note_signed_confidence }})",
            )
        ],
    )

    confident = run_node(node, {"doc_checks": {"note_signed": False, "note_signed_confidence": 0.93}}).output
    unsure = run_node(node, {"doc_checks": {"note_signed": False, "note_signed_confidence": 0.4}}).output
    unanswered = run_node(node, {"doc_checks": {}}).output

    assert confident["findings"][0]["status"] == "fail"
    assert confident["findings"][0]["message"] == "No borrower signature found on the Note (confidence 0.93)"
    assert unsure["findings"][0]["status"] == "not_applicable"
    assert unanswered["findings"][0]["status"] == "not_evaluated"
