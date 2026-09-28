"""Reviews the way customers run them.

Each scenario is a review one of our prospects described: a mortgage audit, a regulatory alert desk, a payroll
run, a therapy claim, an invoice desk, a KYC file, a contract, a waste facility, a data feed. Together they
exercise the behaviours a rule set relies on in production: lookups by a key the record supplies, list
membership, date arithmetic against a fixed `as_of`, rules that apply to some records only, effective windows,
a strict or a lenient missing-value policy, a batch through Map, a record shaped like an agent's structured
response, routing on the outcome, and a workflow file written the way the platform stores one.
"""

import textwrap

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators import Choice, ChoiceOption, Expression, Map, Rules
from dynamiq.nodes.types import ChoiceCondition, ConditionOperator, DerivedValue, ExpressionItem, NamedField, Rule
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableConfig, RunnableStatus


def run(node: Rules, record: dict) -> dict:
    result = node.run(input_data=record, config=RunnableConfig(callbacks=[]))
    assert result.status == RunnableStatus.SUCCESS, result.error
    return result.output


def by_id(output: dict) -> dict[str, dict]:
    return {finding["rule_id"]: finding for finding in output["findings"]}


def statuses(output: dict) -> dict[str, str]:
    return {rule_id: finding["status"] for rule_id, finding in by_id(output).items()}


# --- Regulatory alert desk: which alerts on the watch list need action, and by when ---------------------------


def alert_triage() -> Rules:
    return Rules(
        id="triage",
        name="triage",
        input_fields=[NamedField(name="alert"), NamedField(name="profile"), NamedField(name="as_of")],
        derived_values=[
            DerivedValue(
                name="in_scope", expression="alert.jurisdictions | select('in', profile.jurisdictions) | list"
            ),
        ],
        rules=[
            Rule(
                id="RW-01",
                name="Alert concerns a jurisdiction on the watch list",
                severity="info",
                check="in_scope | length > 0",
                message="None of {{ alert.jurisdictions | join(', ') }} is on the watch list",
            ),
            Rule(
                id="RW-02",
                name="Enacted change with a deadline inside 30 days",
                severity="warn",
                applies_when="alert.status == 'enacted' and has(alert.deadline)",
                check="days_between(as_of, alert.deadline) > 30",
                message="{{ days_between(as_of, alert.deadline) }} days to the {{ alert.deadline }} deadline",
                reason_code="DEADLINE-30",
            ),
            Rule(
                id="RW-03",
                name="Sanctions and AML changes reviewed by Legal",
                applies_when="alert.topic in ['sanctions', 'aml']",
                check="has(alert.legal_review)",
                message="{{ alert.topic | upper }} change without a Legal review",
                reason_code="ESCALATE-LEGAL",
            ),
            Rule(id="RW-04", name="Evidence attached", check="alert.sources | length >= 1"),
        ],
    )


def test_an_aml_alert_due_in_three_weeks_is_escalated_and_flagged_for_its_deadline():
    output = run(
        alert_triage(),
        {
            "alert": {
                "topic": "aml",
                "jurisdictions": ["EU", "TR"],
                "status": "enacted",
                "deadline": "2026-10-10",
                "sources": ["https://eur-lex.europa.eu/..."],
            },
            "profile": {"jurisdictions": ["EU", "DE"]},
            "as_of": "2026-09-19",
        },
    )

    assert statuses(output) == {"RW-01": "pass", "RW-02": "warn", "RW-03": "fail", "RW-04": "pass"}
    assert output["derived"]["in_scope"] == ["EU"]
    assert by_id(output)["RW-02"]["message"] == "21 days to the 2026-10-10 deadline"
    assert by_id(output)["RW-03"]["message"] == "AML change without a Legal review"
    assert output["status"] == "fail"


def test_a_draft_outside_the_watch_list_is_only_noted():
    output = run(
        alert_triage(),
        {
            "alert": {"topic": "packaging", "jurisdictions": ["MX"], "status": "draft", "sources": ["x"]},
            "profile": {"jurisdictions": ["EU", "DE"]},
            "as_of": "2026-09-19",
        },
    )

    assert statuses(output) == {"RW-01": "info", "RW-02": "not_applicable", "RW-03": "not_applicable", "RW-04": "pass"}
    assert by_id(output)["RW-01"]["message"] == "None of MX is on the watch list"
    # An info finding is worth reading but does not fail the alert.
    assert output["status"] == "pass"


# --- Payroll run: every employee line checked against state minimums, in a batch -----------------------------


def payroll_checks() -> Rules:
    return Rules(
        id="payroll",
        name="payroll",
        input_fields=[NamedField(name="employee"), NamedField(name="minimums")],
        derived_values=[DerivedValue(name="overtime_due", expression="max(employee.hours - 40, 0)")],
        rules=[
            Rule(
                id="PAY-01",
                name="Hourly rate at or above the state minimum",
                check="employee.hourly_rate >= minimums[employee.state]",
                message=(
                    "{{ employee.hourly_rate }}/h is below the {{ employee.state }} minimum "
                    "of {{ minimums[employee.state] }}"
                ),
            ),
            Rule(
                id="PAY-02",
                name="Overtime paid for a non-exempt employee",
                applies_when="not employee.exempt",
                check="employee.overtime_hours_paid >= overtime_due",
                message="{{ overtime_due }} overtime hours due, {{ employee.overtime_hours_paid }} paid",
            ),
            Rule(id="PAY-03", name="W-4 on file", severity="warn", check="employee.w4_on_file"),
        ],
    )


def test_a_payroll_batch_reports_each_line_and_holds_an_unknown_state_for_review():
    minimums = {"CA": 16.5, "TX": 7.25, "WA": 16.66}
    lines = [
        {
            "employee": {
                "state": "CA",
                "hourly_rate": 18.0,
                "hours": 46,
                "overtime_hours_paid": 6,
                "exempt": False,
                "w4_on_file": True,
            }
        },
        {
            "employee": {
                "state": "TX",
                "hourly_rate": 7.0,
                "hours": 38,
                "overtime_hours_paid": 0,
                "exempt": False,
                "w4_on_file": False,
            }
        },
        {
            "employee": {
                "state": "WA",
                "hourly_rate": 60.0,
                "hours": 50,
                "overtime_hours_paid": 0,
                "exempt": True,
                "w4_on_file": True,
            }
        },
        {
            "employee": {
                "state": "PR",
                "hourly_rate": 10.0,
                "hours": 40,
                "overtime_hours_paid": 0,
                "exempt": False,
                "w4_on_file": True,
            }
        },
    ]
    batch = Map(id="batch", name="batch", node=payroll_checks(), max_workers=2)

    result = batch.run(
        input_data={"input": [{**line, "minimums": minimums} for line in lines]},
        config=RunnableConfig(callbacks=[]),
    )

    assert result.status == RunnableStatus.SUCCESS
    outputs = result.output["output"]
    assert [item["status"] for item in outputs] == ["pass", "fail", "pass", "not_evaluated"]
    assert statuses(outputs[1]) == {"PAY-01": "fail", "PAY-02": "pass", "PAY-03": "warn"}
    assert by_id(outputs[1])["PAY-01"]["message"] == "7.0/h is below the TX minimum of 7.25"
    assert statuses(outputs[2])["PAY-02"] == "not_applicable"
    # Puerto Rico is not in the minimums table: the line is held, not passed on a comparison with nothing.
    assert statuses(outputs[3])["PAY-01"] == "not_evaluated"
    assert by_id(outputs[3])["PAY-01"]["message"].startswith("check could not be evaluated")


# --- Therapy claim: payer and state specific rules under a strict missing-value policy ------------------------


def claim_checks() -> Rules:
    return Rules(
        id="claim",
        name="claim",
        input_fields=[NamedField(name="claim")],
        on_missing="fail",
        rules=[
            Rule(
                id="AUTH-01",
                name="Date of service inside the authorization window",
                check="date(claim.dos) >= date(claim.auth.start) and date(claim.dos) <= date(claim.auth.end)",
                message=(
                    "Service on {{ claim.dos }} falls outside the authorization "
                    "{{ claim.auth.start }} to {{ claim.auth.end }}"
                ),
            ),
            Rule(
                id="AUTH-02",
                name="Units within the remaining authorization",
                check="claim.units <= claim.auth.units_remaining",
                message="{{ claim.units }} units billed, {{ claim.auth.units_remaining }} remaining",
            ),
            Rule(
                id="CRED-01",
                name="97155 requires a BCBA",
                applies_when="claim.cpt == '97155'",
                check="claim.provider_credential == 'BCBA'",
                references=["CPT 97155"],
            ),
            Rule(
                id="MEDICAL-CA-01",
                name="Medi-Cal claims name the supervising provider",
                applies_when="claim.state == 'CA' and claim.payer == 'Medi-Cal'",
                check="has(claim.supervising_npi)",
            ),
        ],
    )


def test_a_claim_without_an_authorization_fails_under_the_strict_policy():
    complete = {
        "claim": {
            "payer": "Medi-Cal",
            "state": "CA",
            "cpt": "97155",
            "units": 8,
            "dos": "2026-09-14",
            "provider_credential": "BCBA",
            "supervising_npi": "1234567890",
            "auth": {"start": "2026-07-01", "end": "2026-12-31", "units_remaining": 24},
        }
    }
    unauthorized = {
        "claim": {
            "payer": "BCBS",
            "state": "TX",
            "cpt": "97153",
            "units": 4,
            "dos": "2026-09-14",
            "provider_credential": "RBT",
        }
    }

    assert statuses(run(claim_checks(), complete)) == {
        "AUTH-01": "pass",
        "AUTH-02": "pass",
        "CRED-01": "pass",
        "MEDICAL-CA-01": "pass",
    }
    held = run(claim_checks(), unauthorized)
    assert statuses(held) == {
        "AUTH-01": "fail",
        "AUTH-02": "fail",
        "CRED-01": "not_applicable",
        "MEDICAL-CA-01": "not_applicable",
    }
    # The strict policy reports the severity, and still says what was missing.
    assert by_id(held)["AUTH-01"]["message"].endswith("(missing value for claim.auth.start)")
    assert held["status"] == "fail"


# --- Invoice desk: an agent's extraction checked against the purchase order -----------------------------------


def invoice_checks() -> Rules:
    return Rules(
        id="ap",
        name="ap",
        input_fields=[
            NamedField(name="invoice"),
            NamedField(name="po"),
            NamedField(name="ledger"),
            NamedField(name="vat_rates"),
        ],
        derived_values=[DerivedValue(name="variance", expression="abs(invoice.content.total - po.total) / po.total")],
        rules=[
            Rule(
                id="AP-01",
                name="Invoice references its purchase order",
                check="invoice.content.po_number == po.number",
                message="PO {{ invoice.content.po_number }} on the invoice, {{ po.number }} expected",
            ),
            Rule(
                id="AP-02",
                name="Total within 2% of the order",
                check="variance <= 0.02",
                message="Invoice total is {{ (variance * 100) | round(1) }}% off the order",
            ),
            Rule(
                id="AP-03",
                name="Not paid before",
                check="invoice.content.number not in ledger.paid_numbers",
                reason_code="DUPLICATE",
            ),
            Rule(
                id="AP-04",
                name="VAT rate matches the supplier country",
                check="invoice.content.vat_rate == vat_rates[po.supplier_country]",
            ),
            Rule(
                id="AP-05",
                name="Extraction confident enough to post without a look",
                severity="info",
                check="invoice.confidence >= 0.9",
            ),
        ],
    )


def test_an_extracted_invoice_is_checked_against_its_order_with_the_values_read():
    # The invoice is what an extraction agent returns: the fields under `content`, and how sure it was.
    invoice = {
        "content": {"number": "INV-2041", "po_number": "PO-778", "total": 10250.0, "vat_rate": 0.19},
        "confidence": 0.82,
    }
    po = {"number": "PO-778", "total": 10000.0, "supplier_country": "DE"}
    ledger = {"paid_numbers": ["INV-1990", "INV-2041"]}
    vat_rates = {"DE": 0.19, "FR": 0.2}

    output = run(invoice_checks(), {"invoice": invoice, "po": po, "ledger": ledger, "vat_rates": vat_rates})

    assert statuses(output) == {"AP-01": "pass", "AP-02": "fail", "AP-03": "fail", "AP-04": "pass", "AP-05": "info"}
    assert by_id(output)["AP-02"]["message"] == "Invoice total is 2.5% off the order"
    assert by_id(output)["AP-02"]["evaluated"] == {"variance": 0.025}
    assert by_id(output)["AP-03"]["reason_code"] == "DUPLICATE"
    assert output["derived"] == {"variance": 0.025}


def test_a_supplier_country_missing_from_the_rate_table_holds_the_vat_check():
    output = run(
        invoice_checks(),
        {
            "invoice": {
                "content": {"number": "INV-1", "po_number": "PO-1", "total": 100.0, "vat_rate": 0.0},
                "confidence": 0.95,
            },
            "po": {"number": "PO-1", "total": 100.0, "supplier_country": "AE"},
            "ledger": {"paid_numbers": []},
            "vat_rates": {"DE": 0.19},
        },
    )

    # An equality against a lookup that found nothing is not a failed check: nobody knows the rate yet.
    assert statuses(output)["AP-04"] == "not_evaluated"
    assert output["status"] == "not_evaluated"


# --- KYC file: a screening that never ran is not a clear file -------------------------------------------------


def kyc_checks() -> Rules:
    return Rules(
        id="kyc",
        name="kyc",
        input_fields=[NamedField(name="applicant"), NamedField(name="screening"), NamedField(name="as_of")],
        rules=[
            Rule(
                id="KYC-01",
                name="Sanctions screening clear",
                check="screening.sanctions_hits == 0",
                message="{{ screening.sanctions_hits }} sanctions hit(s) on {{ applicant.name }}",
                references=["OFAC SDN", "EU consolidated list"],
            ),
            Rule(
                id="KYC-04",
                name="Identity document in date",
                check="days_between(as_of, applicant.id_expiry) > 0",
                message="ID expired on {{ applicant.id_expiry }}",
            ),
            Rule(
                id="KYC-05",
                name="Proof of address within 90 days",
                check="days_between(applicant.address_proof_date, as_of) <= 90",
            ),
            Rule(
                id="KYC-07",
                name="Enhanced due diligence for a politically exposed person",
                severity="warn",
                applies_when="applicant.pep",
                check="has(applicant.edd_report)",
                references=["FATF R.12"],
            ),
        ],
    )


def test_a_clean_file_passes_and_a_pep_without_a_report_is_flagged():
    applicant = {"name": "A. Sample", "id_expiry": "2028-03-01", "address_proof_date": "2026-08-01", "pep": False}
    as_of = "2026-09-19"

    clean = run(kyc_checks(), {"applicant": applicant, "screening": {"sanctions_hits": 0}, "as_of": as_of})
    pep = run(
        kyc_checks(), {"applicant": {**applicant, "pep": True}, "screening": {"sanctions_hits": 0}, "as_of": as_of}
    )

    assert clean["status"] == "pass"
    assert statuses(pep) == {"KYC-01": "pass", "KYC-04": "pass", "KYC-05": "pass", "KYC-07": "warn"}
    assert pep["status"] == "warn"


def test_a_file_whose_screening_never_ran_is_not_clear():
    output = run(
        kyc_checks(),
        {
            "applicant": {
                "name": "B. Sample",
                "id_expiry": "2026-09-01",
                "address_proof_date": "2026-08-01",
                "pep": False,
            },
            "screening": {},
            "as_of": "2026-09-19",
        },
    )

    assert statuses(output) == {
        "KYC-01": "not_evaluated",
        "KYC-04": "fail",
        "KYC-05": "pass",
        "KYC-07": "not_applicable",
    }
    assert by_id(output)["KYC-01"]["message"] == "missing value for screening.sanctions_hits"
    assert by_id(output)["KYC-04"]["message"] == "ID expired on 2026-09-01"


# --- Contract review: a policy that changes on a date, applied to files dated before and after -----------------


def contract_checks() -> Rules:
    return Rules(
        id="contract",
        name="contract",
        input_fields=[NamedField(name="contract"), NamedField(name="policy")],
        derived_values=[DerivedValue(name="cap_multiple", expression="contract.liability_cap / contract.annual_fees")],
        rules=[
            Rule(
                id="LGL-01",
                name="Governing law on the approved list",
                check="contract.governing_law in policy.allowed_law",
                message="{{ contract.governing_law }} law is not approved",
            ),
            Rule(
                id="LGL-02",
                name="Liability cap at least one year of fees",
                check="cap_multiple >= 1",
                message="Cap is {{ cap_multiple | round(2) }}x annual fees",
            ),
            Rule(
                id="LGL-03",
                name="Auto-renewal gives at least 60 days notice",
                applies_when="contract.auto_renewal",
                check="contract.renewal_notice_days >= 60",
            ),
            Rule(
                id="LGL-04",
                name="Data processing addendum attached",
                check="has(contract.dpa)",
                effective_from="2026-10-01",
                references=["Policy update 2026-Q4"],
            ),
        ],
    )


def test_a_policy_that_starts_in_october_leaves_september_contracts_alone():
    contract = {
        "governing_law": "England and Wales",
        "liability_cap": 90000,
        "annual_fees": 120000,
        "auto_renewal": True,
        "renewal_notice_days": 30,
    }
    policy = {"allowed_law": ["England and Wales", "Delaware", "New York"]}

    september = run(contract_checks(), {"contract": contract, "policy": policy, "as_of": "2026-09-15"})
    october = run(contract_checks(), {"contract": contract, "policy": policy, "as_of": "2026-10-15"})

    assert statuses(september) == {"LGL-01": "pass", "LGL-02": "fail", "LGL-03": "fail", "LGL-04": "not_applicable"}
    assert by_id(september)["LGL-02"]["message"] == "Cap is 0.75x annual fees"
    assert statuses(october)["LGL-04"] == "fail"


# --- Waste facility: obligations that depend on the province, and one bylaw on one city ------------------------


def facility_checks() -> Rules:
    return Rules(
        id="facility",
        name="facility",
        input_fields=[NamedField(name="facility"), NamedField(name="obligations")],
        rules=[
            Rule(
                id="ENV-01",
                name="Organics collected where the province requires it",
                applies_when="obligations[facility.province].organics_required",
                check="'organics' in facility.streams",
                message="{{ facility.province }} requires organics collection at {{ facility.name }}",
                references=["O. Reg. 101/94"],
                tags=["organics"],
            ),
            Rule(
                id="ENV-02",
                name="Annual diversion report filed",
                check="facility.diversion_report_year >= obligations[facility.province].report_from_year",
            ),
            Rule(
                id="ENV-TO-01",
                name="Toronto facilities carry the city's waste diversion permit",
                applies_when="facility.city == 'Toronto'",
                check="has(facility.permits.toronto_diversion)",
                references=["Toronto Municipal Code c. 844"],
            ),
        ],
    )


def test_obligations_follow_the_province_and_a_city_bylaw_applies_to_one_city_only():
    obligations = {
        "ON": {"organics_required": True, "report_from_year": 2026},
        "AB": {"organics_required": False, "report_from_year": 2025},
    }
    toronto = {
        "name": "Scarborough MRF",
        "province": "ON",
        "city": "Toronto",
        "streams": ["paper", "plastics"],
        "diversion_report_year": 2026,
        "permits": {},
    }
    calgary = {
        "name": "Calgary depot",
        "province": "AB",
        "city": "Calgary",
        "streams": ["paper"],
        "diversion_report_year": 2025,
        "permits": {},
    }

    ontario = run(facility_checks(), {"facility": toronto, "obligations": obligations})
    alberta = run(facility_checks(), {"facility": calgary, "obligations": obligations})

    assert statuses(ontario) == {"ENV-01": "fail", "ENV-02": "pass", "ENV-TO-01": "fail"}
    assert by_id(ontario)["ENV-01"]["message"] == "ON requires organics collection at Scarborough MRF"
    assert by_id(ontario)["ENV-01"]["references"] == ["O. Reg. 101/94"]
    assert statuses(alberta) == {"ENV-01": "not_applicable", "ENV-02": "pass", "ENV-TO-01": "not_applicable"}


# --- Data feed: a quality gate on a batch of records, the way a data workflow uses it -------------------------


def record_gate() -> Rules:
    return Rules(
        id="gate",
        name="gate",
        input_fields=[NamedField(name="record"), NamedField(name="reference"), NamedField(name="as_of")],
        rules=[
            Rule(id="DQ-01", name="Email present and addressable", check="has(record.email) and '@' in record.email"),
            Rule(id="DQ-02", name="Amount within the expected range", check="0 < record.amount <= 50000"),
            Rule(
                id="DQ-03",
                name="Customer known",
                check="record.customer_id in reference.customers",
                message="Customer {{ record.customer_id }} is not in the master list",
            ),
            Rule(id="DQ-04", name="Created in the past", check="days_between(record.created_at, as_of) >= 0"),
            Rule(
                id="DQ-05",
                name="Currency stated",
                severity="info",
                check="has(record.currency)",
                message="Currency missing; the feed default applies",
            ),
        ],
    )


def test_a_batch_of_records_is_gated_and_the_bad_rows_say_why():
    reference = {"customers": ["C-1", "C-2"]}
    records = [
        {"customer_id": "C-1", "email": "a@x.io", "amount": 120.0, "currency": "EUR", "created_at": "2026-09-01"},
        {"customer_id": "C-9", "email": "b@x.io", "amount": 80000.0, "created_at": "2026-09-02"},
        {"customer_id": "C-2", "email": "no-at-sign", "amount": 15.0, "currency": "USD", "created_at": "2026-12-01"},
    ]
    batch = Map(id="batch", name="batch", node=record_gate(), max_workers=3)

    result = batch.run(
        input_data={"input": [{"record": record, "reference": reference, "as_of": "2026-09-19"} for record in records]},
        config=RunnableConfig(callbacks=[]),
    )

    outputs = result.output["output"]
    assert [item["status"] for item in outputs] == ["pass", "fail", "fail"]
    assert statuses(outputs[1]) == {"DQ-01": "pass", "DQ-02": "fail", "DQ-03": "fail", "DQ-04": "pass", "DQ-05": "info"}
    assert by_id(outputs[1])["DQ-03"]["message"] == "Customer C-9 is not in the master list"
    assert statuses(outputs[2]) == {"DQ-01": "fail", "DQ-02": "pass", "DQ-03": "pass", "DQ-04": "fail", "DQ-05": "pass"}
    assert outputs[1]["summary"] == {
        "pass": 2,
        "fail": 2,
        "warn": 0,
        "info": 1,
        "not_applicable": 0,
        "not_evaluated": 0,
    }


# --- Post-close audit: the findings routed to a review queue, the clean files straight through -----------------


def audit_workflow() -> Workflow:
    start = Input(id="start", name="start")
    review = Rules(
        id="review",
        name="review",
        input_fields=[NamedField(name="loan"), NamedField(name="docs")],
        derived_values=[DerivedValue(name="ltv", expression="loan.amount / docs.Appraisal.value")],
        rules=[
            Rule(
                id="CR-014",
                name="Note rate matches the Closing Disclosure",
                check="docs.Note.rate == docs.ClosingDisclosure.rate",
                reason_code="CR-014",
            ),
            Rule(
                id="EL-040",
                name="LTV above 80% carries mortgage insurance",
                severity="warn",
                applies_when="ltv > 0.8",
                check="has(docs.MICert)",
                reason_code="EL-040",
            ),
            Rule(id="DC-118", name="Note signed", check="docs.Note.signed", reason_code="DC-118"),
        ],
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"loan": "$.start.output.loan", "docs": "$.start.output.docs"}),
    )
    triage = Expression(
        id="triage",
        name="triage",
        expressions=[
            ExpressionItem(
                key="codes",
                expression=(
                    "findings | selectattr('status', 'in', ['fail', 'warn']) " "| map(attribute='reason_code') | list"
                ),
            ),
            ExpressionItem(key="needs_review", expression="status != 'pass'"),
            ExpressionItem(key="ltv_pct", expression="(ltv * 100) | round(1)"),
        ],
        depends=[NodeDependency(node=review)],
        input_transformer=InputTransformer(
            selector={
                "findings": "$.review.output.findings",
                "status": "$.review.output.status",
                "ltv": "$.review.output.derived.ltv",
            }
        ),
    )
    route = Choice(
        id="route",
        name="route",
        options=[
            ChoiceOption(
                id="queue",
                condition=ChoiceCondition(
                    operator=ConditionOperator.BOOLEAN_EQUALS, variable="$.needs_review", value=True
                ),
            ),
            ChoiceOption(id="clear"),
        ],
        depends=[NodeDependency(node=triage)],
        input_transformer=InputTransformer(selector={"needs_review": "$.triage.output.needs_review"}),
    )
    ticket = Output(
        id="ticket",
        name="ticket",
        depends=[NodeDependency(node=route, option="queue"), NodeDependency(node=triage)],
        input_transformer=InputTransformer(
            selector={"codes": "$.triage.output.codes", "ltv_pct": "$.triage.output.ltv_pct"}
        ),
    )
    return Workflow(id="audit", flow=Flow(id="audit-flow", nodes=[start, review, triage, route, ticket]))


def test_a_file_with_findings_opens_a_ticket_with_the_codes_and_a_clean_file_does_not():
    docs = {
        "Note": {"rate": 6.875, "signed": True},
        "ClosingDisclosure": {"rate": 6.75},
        "Appraisal": {"value": 500000},
    }

    flagged = audit_workflow().run(
        input_data={"loan": {"amount": 450000}, "docs": docs}, config=RunnableConfig(callbacks=[])
    )
    clean = audit_workflow().run(
        input_data={"loan": {"amount": 300000}, "docs": {**docs, "ClosingDisclosure": {"rate": 6.875}}},
        config=RunnableConfig(callbacks=[]),
    )

    assert flagged.status == RunnableStatus.SUCCESS
    assert flagged.output["ticket"]["output"] == {"codes": ["CR-014", "EL-040"], "ltv_pct": 90.0}
    assert clean.output["triage"]["output"]["codes"] == []
    assert clean.output["ticket"]["status"] == RunnableStatus.SKIP


# --- The workflow file the platform writes for a review built in the editor -----------------------------------

PLATFORM_YAML = textwrap.dedent(
    """
    nodes:
      start:
        type: dynamiq.nodes.utils.Input
        name: start

      review:
        type: dynamiq.nodes.operators.Rules
        name: review
        input_fields:
          - { id: 6f1a, name: loan }
          - { id: 6f1b, name: docs }
        derived_values:
          - { id: d1, name: ltv, expression: loan.amount / docs.Appraisal.value }
        rules:
          - id: 9c0e1f2a
            name: LTV within the program limit
            category: Eligibility
            severity: fail
            applies_when: ""
            check: ltv <= loan.max_ltv
            message: "LTV {{ (ltv * 100) | round(1) }}% is above the limit"
            reason_code: EL-032
            references: []
            tags: [eligibility]
            effective_from: ""
            effective_until: ""
          - id: 9c0e1f2b
            name: Flood certificate on file
            category: Documents
            severity: warn
            applies_when: loan.flood_zone in ['A', 'AE', 'V']
            check: has(docs.FloodCert)
            message: ""
            reason_code: ""
            references: [Selling Guide B7-3-07]
            tags: []
            effective_from: "2026-01-01"
            effective_until: ""
          - id: 9c0e1f2c
            name: Retired DU version check
            category: ""
            severity: fail
            applies_when: ""
            check: docs.DU.version == '11.0'
            message: ""
            reason_code: ""
            references: []
            tags: []
            effective_from: ""
            effective_until: ""
            enabled: false
        on_missing: not_evaluated
        depends:
          - node: start
        input_transformer:
          selector:
            loan: $.start.output.loan
            docs: $.start.output.docs

      end:
        type: dynamiq.nodes.utils.Output
        name: end
        depends:
          - node: review
        input_transformer:
          selector:
            status: $.review.output.status
            findings: $.review.output.findings

    flows:
      review-flow:
        name: Review
        nodes: [start, review, end]

    workflows:
      review:
        flow: review-flow
    """
)


def test_a_review_saved_from_the_editor_loads_with_its_empty_fields_and_runs(tmp_path):
    path = tmp_path / "review.yaml"
    path.write_text(PLATFORM_YAML)

    workflow = Workflow.from_yaml_file(str(path), init_components=True)
    review = next(node for node in workflow.flow.nodes if isinstance(node, Rules))
    result = workflow.run(
        input_data={
            "loan": {"amount": 425000, "max_ltv": 0.8, "flood_zone": "AE"},
            "docs": {"Appraisal": {"value": 500000}},
        },
        config=RunnableConfig(callbacks=[]),
    )

    assert [rule.reason_code for rule in review.rules] == ["EL-032", "", ""]
    assert review.rules[1].effective_from == "2026-01-01" and review.rules[2].enabled is False
    assert result.status == RunnableStatus.SUCCESS
    end = result.output["end"]["output"]
    assert end["status"] == "fail"
    assert [finding["rule_id"] for finding in end["findings"]] == ["9c0e1f2a", "9c0e1f2b"]
    assert end["findings"][0]["message"] == "LTV 85.0% is above the limit"
    assert end["findings"][1]["status"] == "warn" and end["findings"][1]["message"] is None
