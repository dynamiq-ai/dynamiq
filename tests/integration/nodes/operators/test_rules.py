import json
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


def test_a_has_guard_lets_the_check_decide_about_the_guarded_document():
    node = Rules(
        id="flood",
        name="flood",
        input_fields=[NamedField(name="docs")],
        rules=[
            Rule(
                id="DC-211",
                name="Flood certificate names a high-risk zone",
                check="has(docs.FloodCert) and docs.FloodCert.zone in ['A', 'AE', 'V']",
                message="Zone {{ docs.FloodCert.zone }} on the certificate",
            )
        ],
    )

    with_cert = run_node(node, {"docs": {"FloodCert": {"zone": "AE"}}}).output["findings"][0]
    without = run_node(node, {"docs": {}}).output["findings"][0]
    unzoned = run_node(node, {"docs": {"FloodCert": {}}}).output["findings"][0]

    assert with_cert["status"] == "pass"
    # The guard decides: no certificate is a failed check, not a value nobody could read.
    assert without["status"] == "fail"
    assert without["message"] == "Zone  on the certificate"
    assert without["evaluated"] == {"docs.FloodCert": None, "docs.FloodCert.zone": None}
    # A certificate without a zone is neither: the guard holds and the read finds nothing.
    assert unzoned["status"] == "not_evaluated"


def test_a_value_left_lazy_reads_the_same_for_every_rule_and_stays_serializable():
    """`map`, `select` and `selectattr` return generators; the first rule to read one would exhaust it."""
    node = Rules(
        id="review",
        input_fields=[NamedField(name="items")],
        derived_values=[DerivedValue(name="prices", expression="items | map(attribute='price')")],
        rules=[
            Rule(id="A", check="(prices | sum) > 100"),
            Rule(id="B", check="(prices | sum) > 100"),
            Rule(id="C", check="(prices | length) == 2"),
            Rule(id="D", check="items | selectattr('flagged')", severity="warn"),
            Rule(id="E", check="items | rejectattr('flagged')", applies_when="items | selectattr('price', 'gt', 50)"),
        ],
    )
    items = [{"price": 60, "flagged": False}, {"price": 60, "flagged": False}]

    result = node.run(input_data={"items": items}, config=RunnableConfig(callbacks=[]))

    # The identical checks agree, the count evaluates, and a check or a condition left lazy is judged by the
    # list it yields rather than by a generator, which is always true.
    assert result.status == RunnableStatus.SUCCESS
    assert {f["rule_id"]: f["status"] for f in result.output["findings"]} == {
        "A": "pass",
        "B": "pass",
        "C": "pass",
        "D": "warn",
        "E": "pass",
    }
    assert result.output["derived"] == {"prices": [60, 60]}
    assert json.dumps(result.output)

    flagged = node.run(
        input_data={"items": [{**items[0], "flagged": True}, items[1]]}, config=RunnableConfig(callbacks=[])
    )
    assert {f["rule_id"]: f["status"] for f in flagged.output["findings"]}["D"] == "pass"


def test_an_upstream_key_named_like_a_derived_value_does_not_break_the_next_one():
    node = Rules(
        id="bands",
        name="bands",
        input_fields=[NamedField(name="fico")],
        derived_values=[
            DerivedValue(name="ratio", expression="fico / 1000"),
            DerivedValue(name="band", expression="'A' if ratio >= 0.7 else 'B'"),
        ],
        rules=[Rule(id="r1", name="Band A", check="band == 'A'")],
    )

    # The upstream payload carries `ratio` too, undeclared; the derived value wins, as it does in the checks.
    output = run_node(node, {"fico": 720, "ratio": 0.1}).output

    assert output["derived"] == {"ratio": 0.72, "band": "A"}
    assert output["findings"][0]["status"] == "pass"


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
            {"rules": [Rule(id="a", name="clash", check="date(date) < today()")]},
            "rule 1 (clash): the check reads 'date' as a value and calls it as a helper",
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


def test_a_method_call_on_a_record_member_reads_the_member_not_the_method():
    """`invoice.get('vat_rate', 0)` needs `invoice`: a dict holds no key called `get`, so the call used to read as a
    missing value and the rule never ran, as a check and as a condition alike."""
    node = Rules(
        name="vat",
        input_fields=[NamedField(name="invoice"), NamedField(name="rec")],
        rules=[
            Rule(id="VAT-01", name="VAT rate present", check="invoice.get('vat_rate', 0) > 0"),
            Rule(
                id="KND-01",
                name="A kind-x record carries more than its kind",
                applies_when="rec.get('kind') == 'x'",
                check="(rec.keys() | list | length) > 1",
            ),
        ],
    )

    present = run_node(node, {"invoice": {"vat_rate": 0.2, "total": 100}, "rec": {"kind": "x", "n": 1}}).output
    absent = run_node(node, {"invoice": {"total": 100}, "rec": {"kind": "y"}}).output
    gone = run_node(node, {"rec": {"kind": "x", "n": 1}}).output

    assert statuses(present) == {"VAT-01": "pass", "KND-01": "pass"}
    assert present["findings"][0]["evaluated"] == {"invoice": "{…2 keys}"}
    # The default passed to `get` stands in for the key, so the rule decides rather than waits for a reviewer.
    assert statuses(absent) == {"VAT-01": "fail", "KND-01": "not_applicable"}
    assert statuses(gone) == {"VAT-01": "not_evaluated", "KND-01": "pass"}
    assert gone["findings"][0]["message"] == "missing value for invoice"


def test_a_record_member_named_like_a_helper_is_the_member_where_it_is_read_and_the_helper_where_it_is_called():
    """`has(date)` used to pass on an empty record, since the bare name fell through to the `date` helper, and a
    record carrying `date` broke every rule that called the helper."""
    node = Rules(
        name="dated",
        input_fields=[NamedField(name="date"), NamedField(name="opened"), NamedField(name="amount")],
        derived_values=[DerivedValue(name="opened_on", expression="date(opened)")],
        rules=[
            Rule(id="DT-01", name="Dated", check="has(date)", severity="warn"),
            Rule(id="DT-02", name="Amount present", check="has(amount)"),
            Rule(id="DT-03", name="Opened on a date", check="date(opened) <= date('2026-12-31')"),
            Rule(
                id="DT-04",
                name="Dated after opening",
                check="days_between(opened, date) >= 0",
                message="Dated {{ date }}, {{ days_between(opened, date) }} days after {{ opened }}",
            ),
            Rule(id="DT-05", name="Dated this decade", check="date > '2020-01-01'"),
        ],
    )

    empty = run_node(node, {}).output
    dated = run_node(node, {"date": "2026-09-01", "opened": "2026-08-20", "amount": 5}).output
    early = run_node(node, {"date": "2026-08-01", "opened": "2026-08-20", "amount": 5}).output

    assert statuses(empty) == {
        "DT-01": "warn",
        "DT-02": "fail",
        "DT-03": "not_evaluated",
        "DT-04": "not_evaluated",
        "DT-05": "not_evaluated",
    }
    assert empty["findings"][0]["evaluated"] == {"date": None}
    assert empty["findings"][4]["message"] == "missing value for date"
    assert statuses(dated) == {rule_id: "pass" for rule_id in ("DT-01", "DT-02", "DT-03", "DT-04", "DT-05")}
    assert str(dated["derived"]["opened_on"]) == "2026-08-20"
    assert statuses(early)["DT-04"] == "fail"
    assert early["findings"][3]["message"] == "Dated 2026-08-01, -19 days after 2026-08-20"


def test_a_lookup_that_finds_nothing_is_not_evaluated_wherever_it_stands():
    """A bare `limits[program]` for a program the table lacks is a value nobody could read: as a check, as a
    condition, through a filter chain and as a derived value."""
    node = Rules(
        name="limits",
        input_fields=[NamedField(name="policy"), NamedField(name="claim"), NamedField(name="things")],
        derived_values=[DerivedValue(name="limit", expression="policy.limits[claim.loss_type]")],
        rules=[
            Rule(id="check", name="covered", check="policy.limits[claim.loss_type]"),
            Rule(
                id="applies",
                name="conditional",
                applies_when="policy.limits[claim.loss_type]",
                check="claim.amount > 0",
            ),
            Rule(id="chain", name="first thing", check="things | selectattr('kind', 'equalto', 'x') | first"),
            Rule(id="derived", name="within the limit", check="claim.amount <= limit"),
        ],
    )
    missing = run_node(
        node,
        {"policy": {"limits": {"fire": 5000}}, "claim": {"loss_type": "theft", "amount": 10}, "things": []},
    ).output
    present = run_node(
        node,
        {
            "policy": {"limits": {"theft": 5000}},
            "claim": {"loss_type": "theft", "amount": 10},
            "things": [{"kind": "x"}],
        },
    ).output

    assert statuses(missing) == {
        "check": "not_evaluated",
        "applies": "not_evaluated",
        "chain": "not_evaluated",
        "derived": "not_evaluated",
    }
    assert missing["status"] == "not_evaluated"
    assert missing["derived"] == {"limit": None}
    assert statuses(present) == {"check": "pass", "applies": "pass", "chain": "pass", "derived": "pass"}
    assert present["derived"] == {"limit": 5000}


def test_a_record_key_with_a_dot_reads_through_the_subscript_form():
    node = Rules(
        name="documents",
        input_fields=[NamedField(name="docs")],
        rules=[Rule(id="pages", name="certificate has pages", check="docs['Flood.Cert'].pages > 0")],
    )

    output = run_node(node, {"docs": {"Flood.Cert": {"pages": 3}}}).output

    assert statuses(output) == {"pages": "pass"}
    assert output["findings"][0]["evaluated"] == {"docs['Flood.Cert'].pages": 3}


def test_a_record_key_named_self_does_not_stop_the_evaluation():
    """A REST payload's top-level `self` link reaches the scope unfiltered and must be an ordinary key there."""
    node = Rules(
        name="amounts",
        input_fields=[NamedField(name="amount")],
        derived_values=[DerivedValue(name="doubled", expression="amount * 2")],
        rules=[
            Rule(
                id="positive",
                name="amount is positive",
                applies_when="amount is defined",
                check="amount > 0 and doubled == 6",
                message="amount is {{ amount }}",
            )
        ],
    )

    output = run_node(node, {"self": "https://api/x/1", "amount": 3}).output

    assert statuses(output) == {"positive": "pass"}
    assert output["derived"] == {"doubled": 6}
    assert output["status"] == "pass"


def test_a_missing_value_under_the_strict_policy_never_reads_as_pass():
    """An info rule reports `info` for a missing value, but the record's screening still did not run."""

    def node(on_missing: str) -> Rules:
        return Rules(
            name="appraisal",
            input_fields=[NamedField(name="docs")],
            on_missing=on_missing,
            rules=[
                Rule(
                    id="value", name="appraisal", severity="info", check="docs.appraisal.value > 0", message="appraisal"
                )
            ],
        )

    lenient = run_node(node("not_evaluated"), {"docs": {}}).output
    strict = run_node(node("fail"), {"docs": {}}).output

    assert statuses(lenient) == {"value": "not_evaluated"}
    assert lenient["status"] == "not_evaluated"
    assert statuses(strict) == {"value": "info"}
    assert strict["findings"][0]["message"] == "appraisal (missing value for docs.appraisal.value)"
    assert strict["status"] == "not_evaluated"
    # A value that is there lets the rule decide, and the record reads as it should.
    assert run_node(node("fail"), {"docs": {"appraisal": {"value": 5}}}).output["status"] == "pass"


def test_a_derived_list_or_dict_keeps_none_where_the_expression_found_no_member():
    """`map(attribute=...)` yields an undefined value for every item that lacks the attribute; inside the list
    or the dict a derived value builds it has to read as None, or the output cannot be serialized."""
    node = Rules(
        name="discounts",
        input_fields=[NamedField(name="items")],
        derived_values=[
            DerivedValue(name="discounts", expression="items | map(attribute='discount') | list"),
            DerivedValue(name="pair", expression="{'first': items[0].discount, 'second': items[1].discount}"),
        ],
        rules=[Rule(id="any", name="has discounts", check="discounts | length > 0")],
    )

    result = run_node(node, {"items": [{"discount": 5}, {"sku": "x"}]}).output

    assert result["derived"] == {"discounts": [5, None], "pair": {"first": 5, "second": None}}
    assert statuses(result) == {"any": "pass"}
    assert json.loads(json.dumps(result)) == result


def test_a_read_rooted_at_self_is_refused_at_build_while_a_nested_self_reads():
    """Jinja binds `self` to its template reference inside every compiled expression, so a top-level key of
    that name is never the value handed over: `has(self)` would find the reference and pass on a record
    without the key. Refusing the read at build names the rule; a `self` inside a record is an ordinary key."""
    for rule, where in [
        (Rule(id="guard", name="guard", check="has(self)"), "the check reads 'self'"),
        (Rule(id="member", name="member", check="self.id > 5"), "the check reads 'self.id'"),
        (Rule(id="when", name="when", applies_when="self is defined", check="amount > 0"), "applies_when reads 'self'"),
        (
            Rule(id="message", name="message", check="amount > 0", message="{{ self.href }}"),
            "the message reads 'self.href'",
        ),
    ]:
        with pytest.raises(ValueError, match=re.escape(where)):
            Rules(name="links", input_fields=[NamedField(name="amount")], rules=[rule])
    with pytest.raises(ValueError, match="derived value 'link' reads 'self.href'"):
        Rules(name="links", derived_values=[DerivedValue(name="link", expression="self.href")], rules=[])
    with pytest.raises(ValueError, match="derived value 'self' could not be read by a rule"):
        Rules(name="links", derived_values=[DerivedValue(name="self", expression="1")], rules=[])

    node = Rules(
        name="links",
        input_fields=[NamedField(name="payload")],
        rules=[Rule(id="link", name="carries its link", check="has(payload.self.href)")],
    )

    assert statuses(run_node(node, {"payload": {"self": {"href": "https://api/x/1"}}}).output) == {"link": "pass"}
    assert statuses(run_node(node, {"payload": {"id": 1}}).output) == {"link": "fail"}


def test_a_record_key_named_like_a_dict_method_reads_as_the_data():
    node = Rules(
        input_fields=[NamedField(name="invoice")],
        rules=[
            Rule(id="r1", name="has items", check="len(invoice.items) > 0"),
            Rule(
                id="r2", name="one a", check="(invoice.items | selectattr('sku', 'equalto', 'a') | list | length) == 1"
            ),
            Rule(id="r3", name="keys and values", check="invoice.keys == ['k'] and invoice.values.total == 10"),
            Rule(id="r4", name="vat default", check="invoice.get('vat_rate', 0) == 0"),
        ],
    )
    record = {"invoice": {"items": [{"sku": "a"}], "keys": ["k"], "values": {"total": 10}, "total": 10}}

    result = node.run(input_data=record, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["status"] == "pass"
    assert {f["rule_id"]: f["status"] for f in result.output["findings"]} == {
        "r1": "pass",
        "r2": "pass",
        "r3": "pass",
        "r4": "pass",
    }
    assert result.output["findings"][0]["evaluated"] == {"invoice.items": "[…1 items]"}


def test_a_path_that_ends_at_a_method_never_leaves_one_in_the_output():
    node = Rules(
        input_fields=[NamedField(name="invoice")],
        derived_values=[DerivedValue(name="counter", expression="invoice.items.count")],
        rules=[
            Rule(id="r1", name="items", check="invoice.items.count > 0"),
            Rule(id="r2", name="bare", check="invoice.items.count"),
            Rule(id="r3", name="applies", applies_when="invoice.items.count", check="invoice.total > 0"),
            Rule(id="r4", name="list", check="invoice.items"),
            Rule(id="r5", name="message", check="invoice.total > 100", message="counter is {{ invoice.items.count }}"),
        ],
    )
    record = {"invoice": {"items": [{"sku": "a"}], "total": 10}}

    result = node.run(input_data=record, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    findings = {f["rule_id"]: f for f in result.output["findings"]}
    # The finding names the method the path found instead of carrying the bound method itself.
    assert findings["r1"]["evaluated"] == {"invoice.items.count": "count(…)"}
    assert findings["r1"]["status"] == "not_evaluated"
    # A check or a condition left at a method is no verdict: every method is truthy, so `invoice.items.count`,
    # the ordinary slip of expecting a count property, would otherwise clear the record.
    assert findings["r2"]["message"] == "check could not be evaluated: read the method count(…), not a value"
    assert findings["r3"]["message"] == ("applies_when could not be evaluated: read the method count(…), not a value")
    assert findings["r2"]["status"] == findings["r3"]["status"] == "not_evaluated"
    assert findings["r4"]["status"] == "pass"  # A list is still judged by its truth.
    assert findings["r5"]["message"] == "counter is count(…)"
    assert result.output["derived"] == {"counter": None}
    # A caller serializing the result with plain `json.dumps` gets no TypeError, and a trace records the
    # values rather than `func: count`.
    json.dumps(result.output)

    # A method reads the same on every run, where its repr carries an address that does not.
    again = node.run(input_data=record, config=RunnableConfig(callbacks=[]))
    assert again.output["findings"] == result.output["findings"]


def test_a_record_key_named_like_a_mutating_method_reads_and_a_refused_attribute_holds_only_its_rule():
    node = Rules(
        input_fields=[NamedField(name="ticket")],
        rules=[
            Rule(id="r1", name="pending update", check="ticket.update == 'pending'"),
            Rule(id="r2", name="open", check="ticket.status == 'open'"),
            Rule(id="r3", name="tags append", check="ticket.tags.append == 1"),
        ],
    )

    result = node.run(
        input_data={"ticket": {"update": "pending", "status": "open", "tags": ["a"]}},
        config=RunnableConfig(callbacks=[]),
    )

    assert result.status == RunnableStatus.SUCCESS
    findings = {f["rule_id"]: f for f in result.output["findings"]}
    assert findings["r1"]["status"] == "pass"
    assert findings["r2"]["status"] == "pass"
    # `append` on a list member is the method, which the sandbox refuses: the one rule is not evaluated.
    assert findings["r3"]["status"] == "not_evaluated"
    assert findings["r3"]["message"] == (
        "check could not be evaluated: access to attribute 'append' of 'list' object is unsafe."
    )
    assert result.output["status"] == "not_evaluated"

    strict = node.model_copy(update={"on_missing": "fail"})
    strict_result = strict.run(
        input_data={"ticket": {"update": "pending", "status": "open", "tags": ["a"]}},
        config=RunnableConfig(callbacks=[]),
    )
    assert {f["rule_id"]: f["status"] for f in strict_result.output["findings"]} == {
        "r1": "pass",
        "r2": "pass",
        "r3": "fail",
    }

    # Without the key, `ticket.update` is a missing value, held before the method is ever reached.
    absent = node.run(input_data={"ticket": {"status": "open", "tags": ["a"]}}, config=RunnableConfig(callbacks=[]))
    findings = {f["rule_id"]: f for f in absent.output["findings"]}
    assert findings["r1"]["status"] == "not_evaluated"
    assert findings["r1"]["message"] == "missing value for ticket.update"
    assert findings["r2"]["status"] == "pass"
