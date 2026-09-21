"""Five reviews from five industries, far apart, and the properties they share.

Insurance adjudicates a claim, a telecom desk stops a fraudulent port-out, a pharmacy checks a medication
order, a housing agency screens an application, and a plant releases a production batch. None of them is a
loan. Between them they cover a lookup by a key the record supplies, list membership and filters over lists of
records, dates against a fixed point in time, derived values that hold a dict, rules for one payer or one
request type, a policy that starts on a date, the strict and the lenient missing-value policy, a Rules node
inside a Map with an Expression folding the batch, routing on the outcome, and the YAML the platform stores.
The last tests are about the node itself: a rule that reaches for Python internals, five hundred rules against
the clock, and a batch across workers.
"""

import time

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators import DecisionTable, Expression, Map, Rules
from dynamiq.nodes.types import DecisionRule, DerivedValue, ExpressionItem, NamedField, Rule
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


def messages(output: dict) -> dict[str, str | None]:
    return {finding["rule_id"]: finding["message"] for finding in output["findings"]}


def reason_codes(output: dict) -> list[str]:
    return [finding["reason_code"] for finding in output["findings"] if finding["status"] in ("fail", "warn")]


# --- Insurance: a motor claim adjudicated against the policy and the claimant's history --------------------


def claim_adjudication() -> Rules:
    return Rules(
        id="adjudication",
        name="adjudication",
        input_fields=[NamedField(name="claim"), NamedField(name="policy"), NamedField(name="history")],
        derived_values=[DerivedValue(name="payable", expression="max(claim.estimate - policy.deductible, 0)")],
        rules=[
            Rule(
                id="POL-01",
                name="Policy in force on the loss date",
                check=(
                    "date(policy.effective_from) <= date(claim.loss_date)"
                    " and date(claim.loss_date) <= date(policy.effective_until)"
                ),
                message=(
                    "Loss on {{ claim.loss_date }}, policy in force"
                    " {{ policy.effective_from }} to {{ policy.effective_until }}"
                ),
            ),
            Rule(id="COV-01", name="Loss type covered", check="claim.loss_type in policy.coverages"),
            Rule(
                id="COV-02",
                name="Estimate within the coverage limit",
                severity="warn",
                check="claim.estimate <= policy.limits[claim.loss_type]",
                message=(
                    "Estimate {{ claim.estimate }} is above the {{ claim.loss_type }} limit"
                    " of {{ policy.limits[claim.loss_type] }}"
                ),
            ),
            Rule(
                id="DOC-01",
                name="Police report on file for a theft",
                applies_when="claim.loss_type == 'theft'",
                check="has(claim.police_report_number)",
            ),
            Rule(
                id="FRD-01",
                name="Reported within 30 days of the loss",
                severity="warn",
                check="days_between(claim.loss_date, claim.reported_date) <= 30",
                message="Reported {{ days_between(claim.loss_date, claim.reported_date) }} days after the loss",
            ),
            Rule(
                id="FRD-02",
                name="Third claim in twelve months",
                severity="info",
                applies_when="history.claims_last_12_months >= 2",
                check="false",
                message=(
                    "{{ history.claims_last_12_months + 1 }} claims in twelve months;"
                    " refer to the special investigations unit"
                ),
            ),
        ],
    )


POLICY = {
    "effective_from": "2026-01-01",
    "effective_until": "2026-12-31",
    "deductible": 500,
    "coverages": ["collision", "theft", "glass"],
    "limits": {"collision": 25000, "theft": 30000, "glass": 1500},
}


def test_a_covered_collision_pays_the_estimate_less_the_deductible():
    output = run(
        claim_adjudication(),
        {
            "claim": {
                "loss_type": "collision",
                "loss_date": "2026-08-02",
                "reported_date": "2026-08-04",
                "estimate": 6200,
            },
            "policy": POLICY,
            "history": {"claims_last_12_months": 0},
        },
    )

    assert output["status"] == "pass"
    assert output["derived"] == {"payable": 5700}
    assert statuses(output)["FRD-02"] == "not_applicable"


def test_a_late_theft_claim_without_a_police_report_is_held_with_every_reason():
    output = run(
        claim_adjudication(),
        {
            "claim": {
                "loss_type": "theft",
                "loss_date": "2026-06-01",
                "reported_date": "2026-07-20",
                "estimate": 41000,
            },
            "policy": POLICY,
            "history": {"claims_last_12_months": 2},
        },
    )

    assert statuses(output) == {
        "POL-01": "pass",
        "COV-01": "pass",
        "COV-02": "warn",
        "DOC-01": "fail",
        "FRD-01": "warn",
        "FRD-02": "info",
    }
    assert by_id(output)["COV-02"]["message"] == "Estimate 41000 is above the theft limit of 30000"
    assert by_id(output)["FRD-01"]["message"] == "Reported 49 days after the loss"
    assert by_id(output)["FRD-02"]["message"].startswith("3 claims in twelve months")


def test_an_uncovered_loss_fails_the_coverage_rule_and_holds_the_limit_check():
    output = run(
        claim_adjudication(),
        {
            "claim": {"loss_type": "flood", "loss_date": "2026-08-02", "reported_date": "2026-08-03", "estimate": 9000},
            "policy": POLICY,
            "history": {"claims_last_12_months": 0},
        },
    )

    # No flood coverage: the coverage rule fails, and the limit lookup finds nothing to compare with.
    assert statuses(output)["COV-01"] == "fail"
    assert statuses(output)["COV-02"] == "not_evaluated"
    assert output["status"] == "fail"


# --- Telecom: a port-out request checked for account takeover before it goes through ----------------------


def port_out_workflow() -> Workflow:
    start = Input(id="start", name="start")
    checks = Rules(
        id="takeover",
        name="takeover",
        input_fields=[NamedField(name="request"), NamedField(name="account"), NamedField(name="watchlist")],
        on_missing="fail",
        rules=[
            Rule(
                id="SEC-01",
                name="Account PIN verified on this request",
                check="request.pin_verified",
                reason_code="PIN",
            ),
            Rule(
                id="SEC-02",
                name="No SIM change in the last three days",
                applies_when="has(account.sim_changed_at)",
                check="days_between(account.sim_changed_at, request.requested_at) >= 3",
                message="SIM changed {{ days_between(account.sim_changed_at, request.requested_at) }} day(s) ago",
                reason_code="RECENT-SIM",
            ),
            Rule(
                id="SEC-03",
                name="Contact email unchanged for a week",
                severity="warn",
                applies_when="has(account.email_changed_at)",
                check="days_between(account.email_changed_at, request.requested_at) >= 7",
                reason_code="RECENT-EMAIL",
            ),
            Rule(
                id="SEC-04",
                name="Account older than 30 days for a port-out",
                applies_when="request.type == 'port_out'",
                check="days_between(account.opened_at, request.requested_at) >= 30",
                reason_code="NEW-ACCOUNT",
            ),
            Rule(
                id="SEC-05",
                name="Destination carrier not on the watch list",
                severity="warn",
                applies_when="request.type == 'port_out'",
                check="request.destination_carrier not in watchlist.carriers",
                reason_code="CARRIER",
            ),
        ],
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(
            selector={
                "request": "$.start.output.request",
                "account": "$.start.output.account",
                "watchlist": "$.start.output.watchlist",
            }
        ),
    )
    decision = Expression(
        id="decision",
        name="decision",
        expressions=[
            ExpressionItem(key="decision", expression="'approve' if status == 'pass' else 'hold'"),
            ExpressionItem(
                key="reasons",
                expression=(
                    "findings | selectattr('status', 'in', ['fail', 'warn'])" " | map(attribute='reason_code') | list"
                ),
            ),
        ],
        depends=[NodeDependency(node=checks)],
        input_transformer=InputTransformer(
            selector={"status": "$.takeover.output.status", "findings": "$.takeover.output.findings"}
        ),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=decision)],
        input_transformer=InputTransformer(
            selector={"decision": "$.decision.output.decision", "reasons": "$.decision.output.reasons"}
        ),
    )
    return Workflow(id="port-out", flow=Flow(id="port-out-flow", nodes=[start, checks, decision, end]))


def test_a_port_out_after_a_sim_swap_is_held_and_a_clean_one_approved():
    watchlist = {"carriers": ["QuickPort Mobile"]}
    takeover = {
        "request": {
            "type": "port_out",
            "requested_at": "2026-09-19",
            "pin_verified": True,
            "destination_carrier": "QuickPort Mobile",
        },
        "account": {"opened_at": "2024-03-10", "sim_changed_at": "2026-09-18", "email_changed_at": "2026-09-17"},
        "watchlist": watchlist,
    }
    routine = {
        "request": {
            "type": "port_out",
            "requested_at": "2026-09-19",
            "pin_verified": True,
            "destination_carrier": "Northern Cellular",
        },
        "account": {"opened_at": "2021-06-01"},
        "watchlist": watchlist,
    }

    held = port_out_workflow().run(input_data=takeover, config=RunnableConfig(callbacks=[]))
    approved = port_out_workflow().run(input_data=routine, config=RunnableConfig(callbacks=[]))

    assert held.output["end"]["output"] == {"decision": "hold", "reasons": ["RECENT-SIM", "RECENT-EMAIL", "CARRIER"]}
    assert approved.output["end"]["output"] == {"decision": "approve", "reasons": []}


def test_a_request_without_the_pin_answer_is_refused_under_the_strict_policy():
    result = port_out_workflow().run(
        input_data={
            "request": {"type": "sim_swap", "requested_at": "2026-09-19"},
            "account": {"opened_at": "2025-01-01"},
            "watchlist": {"carriers": []},
        },
        config=RunnableConfig(callbacks=[]),
    )

    findings = by_id(result.output["takeover"]["output"])
    assert findings["SEC-01"]["status"] == "fail"
    assert findings["SEC-01"]["message"] == "missing value for request.pin_verified"
    assert result.output["end"]["output"]["decision"] == "hold"


# --- Healthcare: a medication order checked against the formulary and the patient -------------------------


def order_checks() -> Rules:
    return Rules(
        id="pharmacy",
        name="pharmacy",
        input_fields=[NamedField(name="order"), NamedField(name="patient"), NamedField(name="formulary")],
        derived_values=[
            DerivedValue(name="entry", expression="formulary[order.drug]"),
            DerivedValue(name="daily_mg", expression="order.dose_mg * order.doses_per_day"),
            DerivedValue(name="mg_per_kg", expression="order.dose_mg / patient.weight_kg"),
        ],
        rules=[
            Rule(
                id="MED-01",
                name="Daily dose within the formulary maximum",
                check="daily_mg <= entry.max_daily_mg",
                message="{{ daily_mg }} mg/day ordered, {{ entry.max_daily_mg }} mg/day maximum",
            ),
            Rule(
                id="MED-02",
                name="Weight-based dose within the paediatric limit",
                applies_when="patient.age < 18",
                check="mg_per_kg <= entry.mg_per_kg_max",
                message="{{ mg_per_kg | round(2) }} mg/kg per dose, {{ entry.mg_per_kg_max }} mg/kg maximum",
            ),
            Rule(
                id="MED-03",
                name="No recorded allergy to the drug class",
                check="entry.allergy_class not in patient.allergies",
                message="Patient allergic to {{ entry.allergy_class }}",
            ),
            Rule(
                id="MED-04",
                name="Renal adjustment recorded for reduced kidney function",
                severity="warn",
                applies_when="patient.egfr < entry.renal_threshold_egfr",
                check="has(order.renal_adjustment)",
            ),
        ],
    )


FORMULARY = {
    "amoxicillin": {
        "max_daily_mg": 3000,
        "mg_per_kg_max": 45,
        "allergy_class": "penicillin",
        "renal_threshold_egfr": 30,
    },
    "metformin": {"max_daily_mg": 2550, "mg_per_kg_max": 30, "allergy_class": "biguanide", "renal_threshold_egfr": 45},
}


def test_a_paediatric_order_over_the_weight_limit_and_an_allergy_are_both_found():
    child = run(
        order_checks(),
        {
            "order": {"drug": "amoxicillin", "dose_mg": 800, "doses_per_day": 3},
            "patient": {"age": 6, "weight_kg": 16, "allergies": [], "egfr": 110},
            "formulary": FORMULARY,
        },
    )
    allergic = run(
        order_checks(),
        {
            "order": {"drug": "amoxicillin", "dose_mg": 500, "doses_per_day": 3},
            "patient": {"age": 41, "weight_kg": 80, "allergies": ["penicillin"], "egfr": 95},
            "formulary": FORMULARY,
        },
    )

    assert statuses(child) == {"MED-01": "pass", "MED-02": "fail", "MED-03": "pass", "MED-04": "not_applicable"}
    assert by_id(child)["MED-02"]["message"] == "50.0 mg/kg per dose, 45 mg/kg maximum"
    assert statuses(allergic)["MED-03"] == "fail" and statuses(allergic)["MED-02"] == "not_applicable"


def test_reduced_kidney_function_without_an_adjustment_is_a_warning():
    output = run(
        order_checks(),
        {
            "order": {"drug": "metformin", "dose_mg": 500, "doses_per_day": 2},
            "patient": {"age": 70, "weight_kg": 72, "allergies": [], "egfr": 38},
            "formulary": FORMULARY,
        },
    )

    assert statuses(output) == {"MED-01": "pass", "MED-02": "not_applicable", "MED-03": "pass", "MED-04": "warn"}
    assert output["status"] == "warn"


def test_a_drug_missing_from_the_formulary_holds_every_rule_that_needs_its_entry():
    output = run(
        order_checks(),
        {
            "order": {"drug": "novadrug", "dose_mg": 10, "doses_per_day": 1},
            "patient": {"age": 30, "weight_kg": 70, "allergies": [], "egfr": 100},
            "formulary": FORMULARY,
        },
    )

    assert output["derived"]["entry"] is None
    assert statuses(output) == {
        "MED-01": "not_evaluated",
        "MED-02": "not_applicable",
        "MED-03": "not_evaluated",
        "MED-04": "not_evaluated",
    }
    assert by_id(output)["MED-01"]["message"] == "missing value for entry.max_daily_mg"
    assert output["status"] == "not_evaluated"


# --- Government: a housing assistance application screened against the programme rules -------------------


def housing_screening() -> Rules:
    return Rules(
        id="screening",
        name="screening",
        input_fields=[NamedField(name="household"), NamedField(name="limits"), NamedField(name="as_of")],
        rules=[
            Rule(
                id="ELG-01",
                name="Income at or below the limit for the household size",
                check="household.monthly_income <= limits[household.state].by_size[household.size - 1]",
                message=(
                    "Income {{ household.monthly_income }} is above the limit of"
                    " {{ limits[household.state].by_size[household.size - 1] }}"
                    " for {{ household.size }} people"
                ),
            ),
            Rule(
                id="ELG-02", name="State residency of at least twelve months", check="household.residency_months >= 12"
            ),
            Rule(
                id="ELG-03",
                name="A dependant under 18 or a member over 62 in the household",
                check=(
                    "(household.members | selectattr('age', 'lt', 18) | list | length) > 0"
                    " or (household.members | selectattr('age', 'gt', 62) | list | length) > 0"
                ),
            ),
            Rule(
                id="DOC-01",
                name="Identity document on file for every adult",
                check=(
                    "(household.members | selectattr('age', 'ge', 18)"
                    " | rejectattr('id_on_file') | list | length) == 0"
                ),
            ),
            Rule(
                id="AST-01",
                name="Assets under the programme ceiling",
                check="household.assets <= 5000",
                effective_from="2027-01-01",
                references=["Programme rule change, 2027 plan year"],
            ),
        ],
    )


LIMITS = {"CA": {"by_size": [2900, 3300, 3700, 4100]}, "TX": {"by_size": [2200, 2500, 2800, 3100]}}


def test_a_family_qualifies_this_year_and_meets_the_asset_test_only_from_next_year():
    household = {
        "state": "CA",
        "size": 3,
        "monthly_income": 3500,
        "residency_months": 30,
        "assets": 7200,
        "members": [{"age": 34, "id_on_file": True}, {"age": 31, "id_on_file": True}, {"age": 4, "id_on_file": False}],
    }

    this_year = run(housing_screening(), {"household": household, "limits": LIMITS, "as_of": "2026-09-19"})
    next_year = run(housing_screening(), {"household": household, "limits": LIMITS, "as_of": "2027-02-01"})

    assert this_year["status"] == "pass"
    assert statuses(this_year)["AST-01"] == "not_applicable"
    assert statuses(next_year)["AST-01"] == "fail"


def test_an_application_over_the_limit_with_a_missing_document_says_both():
    output = run(
        housing_screening(),
        {
            "household": {
                "state": "TX",
                "size": 2,
                "monthly_income": 2700,
                "residency_months": 8,
                "assets": 100,
                "members": [{"age": 45, "id_on_file": True}, {"age": 44, "id_on_file": False}],
            },
            "limits": LIMITS,
            "as_of": "2026-09-19",
        },
    )

    assert statuses(output) == {
        "ELG-01": "fail",
        "ELG-02": "fail",
        "ELG-03": "fail",
        "DOC-01": "fail",
        "AST-01": "not_applicable",
    }
    assert by_id(output)["ELG-01"]["message"] == "Income 2700 is above the limit of 2500 for 2 people"


# --- Manufacturing: a batch released only when every measured attribute is in specification --------------


def release_workflow() -> Workflow:
    start = Input(id="start", name="start")
    attribute_checks = Rules(
        id="attribute",
        name="attribute",
        input_fields=[
            NamedField(name="name"),
            NamedField(name="value"),
            NamedField(name="spec"),
            NamedField(name="method_validated"),
        ],
        derived_values=[DerivedValue(name="attribute", expression="name")],
        rules=[
            Rule(
                id="QC-01",
                name="Result within specification",
                check="spec.min <= value and value <= spec.max",
                message="{{ name }} measured {{ value }}, specification {{ spec.min }} to {{ spec.max }}",
            ),
            Rule(id="QC-02", name="Test method validated", severity="warn", check="method_validated"),
        ],
    )
    release = Map(
        id="release",
        name="release",
        node=attribute_checks,
        max_workers=4,
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"input": "$.start.output.batch.attributes"}),
    )
    verdict = Expression(
        id="verdict",
        name="verdict",
        expressions=[
            ExpressionItem(
                key="released", expression="(results | selectattr('status', 'equalto', 'fail') | list | length) == 0"
            ),
            ExpressionItem(
                key="held_on",
                expression="results | selectattr('status', 'ne', 'pass') | map(attribute='derived.attribute') | list",
            ),
        ],
        depends=[NodeDependency(node=release)],
        input_transformer=InputTransformer(selector={"results": "$.release.output.output"}),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=verdict)],
        input_transformer=InputTransformer(
            selector={"released": "$.verdict.output.released", "held_on": "$.verdict.output.held_on"}
        ),
    )
    return Workflow(id="release", flow=Flow(id="release-flow", nodes=[start, release, verdict, end]))


BATCH = {
    "id": "LOT-2026-0917",
    "attributes": [
        {"name": "assay", "value": 99.1, "spec": {"min": 98.0, "max": 102.0}, "method_validated": True},
        {"name": "moisture", "value": 0.9, "spec": {"min": 0.0, "max": 0.5}, "method_validated": True},
        {"name": "particle_size", "value": 42, "spec": {"min": 30, "max": 60}, "method_validated": False},
    ],
}


def test_a_batch_with_a_result_out_of_specification_is_held_on_that_attribute():
    result = release_workflow().run(input_data={"batch": BATCH}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["end"]["output"] == {"released": False, "held_on": ["moisture", "particle_size"]}
    per_attribute = result.output["release"]["output"]["output"]
    assert [item["status"] for item in per_attribute] == ["pass", "fail", "warn"]
    assert by_id(per_attribute[1])["QC-01"]["message"] == "moisture measured 0.9, specification 0.0 to 0.5"


def test_the_release_workflow_round_trips_through_yaml_with_the_rules_inside_the_map(tmp_path):
    path = tmp_path / "release.yaml"
    release_workflow().to_yaml_file(path)

    loaded = Workflow.from_yaml_file(str(path), init_components=True)
    inner = next(node for node in loaded.flow.nodes if isinstance(node, Map)).node
    result = loaded.run(input_data={"batch": BATCH}, config=RunnableConfig(callbacks=[]))

    assert isinstance(inner, Rules) and [rule.id for rule in inner.rules] == ["QC-01", "QC-02"]
    assert result.output["end"]["output"] == {"released": False, "held_on": ["moisture", "particle_size"]}


# --- The node under pressure: an escape attempt, five hundred rules, a batch across workers ---------------


def test_a_rule_that_reaches_for_python_internals_is_refused_when_the_node_is_built():
    with pytest.raises(ValueError, match=r"rule 1 \(escape\): the check reads a private attribute"):
        Rules(
            id="x",
            name="x",
            input_fields=[NamedField(name="record")],
            rules=[Rule(id="r", name="escape", check="record.__class__.__mro__[1].__subclasses__() | length > 0")],
        )


def test_a_record_key_with_one_leading_underscore_is_ordinary_data():
    # A document store hands back `_id` and `_source`; only Python internals are refused.
    node = Rules(
        id="x",
        name="x",
        input_fields=[NamedField(name="record")],
        rules=[Rule(id="r", name="has id", check="record._id != '' and record._source.status == 'ok'")],
    )

    output = run(node, {"record": {"_id": "abc", "_source": {"status": "ok"}}})

    assert output["status"] == "pass"
    assert output["findings"][0]["evaluated"] == {"record._id": "abc", "record._source.status": "ok"}


def test_an_escape_through_a_filter_is_refused_and_holds_only_its_rule():
    node = Rules(
        id="x",
        name="x",
        input_fields=[NamedField(name="record")],
        rules=[
            Rule(id="r", name="escape", check="(record | attr('__class__')) == 'dict'"),
            Rule(id="ok", name="ok", check="record.a == 1"),
        ],
    )

    result = node.run(input_data={"record": {"a": 1}}, config=RunnableConfig(callbacks=[]))

    # The sandbox refuses the read: the rule that tried it is not evaluated, the rest of the record still reports.
    assert result.status == RunnableStatus.SUCCESS
    findings = {finding["rule_id"]: finding for finding in result.output["findings"]}
    assert findings["r"]["status"] == "not_evaluated"
    assert "unsafe" in findings["r"]["message"]
    assert findings["ok"]["status"] == "pass"
    assert result.output["status"] == "not_evaluated"


def test_five_hundred_rules_cost_milliseconds_per_record_and_a_map_shares_the_compiled_rules(monkeypatch):
    rules = [
        Rule(
            id=f"CHK-{index:03d}",
            name=f"Reading {index} within limit",
            check=f"record.readings[{index % 20}] <= limits.max",
            message="Reading {{ record.readings[" + str(index % 20) + "] }} above {{ limits.max }}",
        )
        for index in range(500)
    ]
    node = Rules(
        id="sensors", name="sensors", input_fields=[NamedField(name="record"), NamedField(name="limits")], rules=rules
    )
    record = {"record": {"readings": [float(i) for i in range(20)]}, "limits": {"max": 10}}

    run(node, record)
    started = time.perf_counter()
    output = run(node, record)
    per_record = time.perf_counter() - started

    assert len(output["findings"]) == 500 and output["summary"]["fail"] == 500 * 9 // 20
    assert len(node.to_dict(for_tracing=True)["rules"]) == 50
    # A few milliseconds locally; the bound only catches a gross regression, such as compiling per run,
    # without tying the suite to the speed of the machine it runs on.
    assert per_record < 2, f"500 rules took {per_record:.3f}s for one record"

    compile_rules = Rules._compile_rules
    compiled_by = []

    def counting(self):
        compiled_by.append(self.id)
        return compile_rules(self)

    monkeypatch.setattr(Rules, "_compile_rules", counting)
    batch = Map(id="batch", name="batch", node=node, max_workers=8)
    result = batch.run(input_data={"input": [record] * 16}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert [item["summary"]["fail"] for item in result.output["output"]] == [500 * 9 // 20] * 16
    # Each item runs on a clone that shares the compiled rules and keeps their ids: compiling five hundred
    # templates per item would cost far more than evaluating them, which the wall clock of a loaded CI
    # runner cannot be trusted to show.
    assert compiled_by == []
    assert {finding["rule_id"] for item in result.output["output"] for finding in item["findings"]} == {
        rule.id for rule in rules
    }


# --- Customs: a broker checks an import declaration before it is lodged ---------------------------


def customs_checks() -> Rules:
    return Rules(
        id="customs",
        name="customs",
        input_fields=[
            NamedField(name="declaration"),
            NamedField(name="documents"),
            NamedField(name="tariff"),
            NamedField(name="restricted"),
            NamedField(name="as_of"),
        ],
        derived_values=[
            DerivedValue(name="declared_value", expression="declaration.lines | map(attribute='value') | sum"),
            DerivedValue(
                name="unknown_codes",
                expression="declaration.lines | map(attribute='hs_code') | reject('in', tariff) | list",
            ),
            DerivedValue(
                name="restricted_lines",
                expression="declaration.lines | selectattr('hs_code', 'in', restricted) | list",
            ),
        ],
        rules=[
            Rule(
                id="REG-01",
                name="Importer registration valid on the lodgement date",
                check="date(declaration.importer.registered_until) >= date(as_of)",
                message="Registration lapsed on {{ declaration.importer.registered_until }}",
                reason_code="REG-01",
            ),
            Rule(id="DOC-01", name="Commercial invoice attached", check="has(documents.invoice)", reason_code="DOC-01"),
            Rule(id="DOC-02", name="Packing list attached", check="has(documents.packing_list)", reason_code="DOC-02"),
            Rule(
                id="DOC-03",
                name="Certificate of origin for a preferential origin claim",
                applies_when="declaration.preferential_origin_claimed",
                check="has(documents.certificate_of_origin)",
                reason_code="DOC-03",
            ),
            Rule(
                id="HS-01",
                name="Every line carries a tariff code the schedule knows",
                check="(unknown_codes | length) == 0",
                message="Unknown tariff codes: {{ unknown_codes | join(', ') }}",
                reason_code="HS-01",
            ),
            Rule(
                id="HS-02",
                name="Restricted goods carry an import permit",
                applies_when="(restricted_lines | length) > 0",
                check="has(documents.permit)",
                message="{{ restricted_lines | length }} restricted line(s) and no permit",
                reason_code="HS-02",
            ),
            Rule(
                id="VAL-01",
                name="Invoice total matches the declared value",
                check="abs(documents.invoice.total - declared_value) <= 1",
                message="Invoice {{ documents.invoice.total }}, declared {{ declared_value }}",
                reason_code="VAL-01",
            ),
            Rule(
                id="VAL-02",
                name="Unit value at or above the reference price",
                severity="warn",
                check=(
                    "declared_value / (declaration.lines | map(attribute='quantity') | sum)"
                    " >= tariff[declaration.lines[0].hs_code].reference_unit_value"
                ),
                reason_code="VAL-02",
                tags=["valuation"],
            ),
        ],
    )


TARIFF = {
    "8471.30": {"duty_rate": 0.0, "reference_unit_value": 300},
    "2208.30": {"duty_rate": 0.5, "reference_unit_value": 20},
}


def declaration(lines: list[dict], **over) -> dict:
    return {
        "declaration": {
            "importer": {"id": "IMP-4471", "registered_until": "2027-03-31"},
            "preferential_origin_claimed": False,
            "lines": lines,
            **over,
        },
        "documents": {"invoice": {"total": sum(line["value"] for line in lines)}, "packing_list": {"pages": 2}},
        "tariff": TARIFF,
        "restricted": ["2208.30"],
        "as_of": "2026-09-19",
    }


def test_a_complete_declaration_of_known_goods_is_cleared():
    output = run(customs_checks(), declaration([{"hs_code": "8471.30", "value": 24000, "quantity": 40}]))

    assert output["status"] == "pass"
    assert output["derived"]["declared_value"] == 24000 and output["derived"]["unknown_codes"] == []
    assert statuses(output)["DOC-03"] == "not_applicable" and statuses(output)["HS-02"] == "not_applicable"


def test_restricted_goods_without_a_permit_and_an_unknown_code_are_both_named():
    lines = [
        {"hs_code": "2208.30", "value": 5000, "quantity": 200},
        {"hs_code": "9999.99", "value": 800, "quantity": 10},
    ]
    output = run(customs_checks(), declaration(lines))

    assert output["status"] == "fail"
    assert statuses(output)["HS-02"] == "fail" and statuses(output)["HS-01"] == "fail"
    assert messages(output)["HS-01"] == "Unknown tariff codes: 9999.99"
    assert messages(output)["HS-02"] == "1 restricted line(s) and no permit"
    assert statuses(output)["VAL-02"] == "pass"


def test_a_preferential_origin_claim_needs_its_certificate_and_the_invoice_must_match():
    record = declaration([{"hs_code": "8471.30", "value": 24000, "quantity": 40}], preferential_origin_claimed=True)
    record["documents"]["invoice"]["total"] = 22000
    output = run(customs_checks(), record)

    assert output["status"] == "fail"
    assert statuses(output)["DOC-03"] == "fail"
    assert messages(output)["VAL-01"] == "Invoice 22000, declared 24000"
    assert set(reason_codes(output)) == {"DOC-03", "VAL-01"}


# --- Mobility: a ride-hailing platform onboards a driver under per-city policy --------------------


def driver_onboarding() -> Rules:
    return Rules(
        id="onboarding",
        name="onboarding",
        input_fields=[NamedField(name="driver"), NamedField(name="policy"), NamedField(name="as_of")],
        derived_values=[
            DerivedValue(name="city_policy", expression="policy.cities[driver.city]"),
            DerivedValue(name="age", expression="days_between(driver.dob, as_of) // 365"),
        ],
        rules=[
            Rule(
                id="LIC-01",
                name="Licence valid on the onboarding date",
                check="date(driver.licence.expires) >= date(as_of)",
                reason_code="LIC-01",
            ),
            Rule(
                id="LIC-02",
                name="Licence class allowed in the city",
                check="driver.licence.category in city_policy.licence_classes",
                message="Class {{ driver.licence.category }} is not accepted in {{ driver.city }}",
                reason_code="LIC-02",
            ),
            Rule(
                id="AGE-01",
                name="Minimum age for the city",
                check="age >= city_policy.min_age",
                message="Driver is {{ age }}, the city requires {{ city_policy.min_age }}",
                reason_code="AGE-01",
            ),
            Rule(
                id="BGC-01",
                name="Background check cleared",
                check="driver.background_check.result == 'clear'",
                reason_code="BGC-01",
            ),
            Rule(
                id="BGC-02",
                name="Background check within twelve months",
                severity="warn",
                check="days_between(driver.background_check.date, as_of) <= 365",
                reason_code="BGC-02",
            ),
            Rule(
                id="VEH-01",
                name="Vehicle year within the city minimum",
                check="driver.vehicle.year >= city_policy.min_vehicle_year",
                message="{{ driver.vehicle.year }} is older than the {{ city_policy.min_vehicle_year }} minimum",
                reason_code="VEH-01",
            ),
            Rule(
                id="VEH-02",
                name="Electric vehicle required",
                applies_when="city_policy.electric_only",
                check="driver.vehicle.is_electric",
                reason_code="VEH-02",
                effective_from="2027-01-01",
            ),
        ],
    )


CITY_POLICY = {
    "cities": {
        "DXB": {"min_age": 21, "licence_classes": ["3", "5"], "min_vehicle_year": 2019, "electric_only": True},
        "RUH": {
            "min_age": 20,
            "licence_classes": ["private", "public"],
            "min_vehicle_year": 2018,
            "electric_only": False,
        },
    }
}


def driver(**over) -> dict:
    return {
        "driver": {
            "city": "DXB",
            "dob": "1998-04-12",
            "licence": {"category": "3", "expires": "2028-01-31"},
            "background_check": {"result": "clear", "date": "2026-06-01"},
            "vehicle": {"year": 2021, "is_electric": False},
            **over,
        },
        "policy": CITY_POLICY,
        "as_of": "2026-09-19",
    }


def test_a_driver_who_meets_the_city_policy_is_approved_and_next_years_rule_waits():
    output = run(driver_onboarding(), driver())

    assert output["status"] == "pass"
    assert output["derived"]["age"] == 28
    assert statuses(output)["VEH-02"] == "not_applicable"

    output = run(driver_onboarding(), {**driver(), "as_of": "2027-02-01"})
    assert statuses(output)["VEH-02"] == "fail" and output["status"] == "fail"


def test_the_same_driver_is_judged_by_the_policy_of_the_city_applied_for():
    too_young = driver(dob="2006-01-15", licence={"category": "3", "expires": "2028-01-31"})
    assert statuses(run(driver_onboarding(), too_young))["AGE-01"] == "fail"

    riyadh = driver(city="RUH", dob="2006-01-15", licence={"category": "public", "expires": "2028-01-31"})
    output = run(driver_onboarding(), riyadh)
    assert statuses(output)["AGE-01"] == "pass" and statuses(output)["LIC-02"] == "pass"


def test_a_city_the_policy_does_not_cover_is_held_rather_than_approved():
    output = run(driver_onboarding(), driver(city="CAI"))

    assert output["status"] == "not_evaluated"
    assert output["derived"]["city_policy"] is None
    assert statuses(output)["LIC-02"] == "not_evaluated" and statuses(output)["BGC-01"] == "pass"


# --- Customer support: a ticket triaged by a table, screened by rules and routed to a queue -------------


def support_triage() -> Workflow:
    """A support desk's first pass: the table picks the queue and the SLA from the plan and the channel, the
    rules screen what a human must see before the queue does, and the expression turns the SLA into a due time."""
    start = Input(id="start", name="start")
    queue = DecisionTable(
        id="queue",
        name="queue",
        hit_policy="first",
        input_columns=[
            NamedField(name="plan", type="string"),
            NamedField(name="channel", type="string"),
            NamedField(name="sentiment", type="float"),
        ],
        output_columns=[NamedField(name="queue", type="string"), NamedField(name="sla_hours", type="int")],
        rules=[
            DecisionRule(id="q1", name="enterprise phone", when=["enterprise", "phone", ""], then=["tier2", "1"]),
            DecisionRule(id="q2", name="enterprise", when=["enterprise", "", ""], then=["tier2", "4"]),
            DecisionRule(id="q3", name="angry customer", when=["", "", "< -0.5"], then=["retention", "2"]),
            DecisionRule(id="q4", name="everyone else", when=["", "", ""], then=["tier1", "24"]),
        ],
        depends=[NodeDependency(start)],
        input_transformer=InputTransformer(
            selector={
                "plan": "$.start.output.customer.plan",
                "channel": "$.start.output.ticket.channel",
                "sentiment": "$.start.output.analysis.sentiment",
            }
        ),
    )
    screen = Rules(
        id="screen",
        name="screen",
        input_fields=[NamedField(name="ticket"), NamedField(name="customer"), NamedField(name="analysis")],
        derived_values=[DerivedValue(name="open_tickets", expression="customer.open_tickets | length")],
        rules=[
            Rule(
                id="ESC-01",
                name="Legal or regulator mentioned",
                severity="fail",
                check="not (analysis.topics | select('in', ['legal', 'regulator', 'chargeback']) | list)",
                message="Ticket mentions {{ analysis.topics | join(', ') }}; route to a supervisor",
                reason_code="ESC-01",
            ),
            Rule(
                id="ESC-02",
                name="Repeat contact on an open ticket",
                severity="warn",
                applies_when="open_tickets > 0",
                check="ticket.subject not in (customer.open_tickets | map(attribute='subject') | list)",
                message="The customer already has an open ticket with the same subject",
                reason_code="ESC-02",
            ),
            Rule(
                id="PII-01",
                name="No card number in the ticket",
                severity="fail",
                check="not analysis.contains_card_number",
                message="The ticket body carries a card number; redact before it reaches the queue",
                reason_code="PII-01",
            ),
            Rule(
                id="LANG-01",
                name="Language the queue speaks",
                severity="info",
                check="ticket.language in ['en', 'ar']",
                message="Ticket is in {{ ticket.language }}; a translation step is needed",
            ),
        ],
        depends=[NodeDependency(start)],
        input_transformer=InputTransformer(
            selector={
                "ticket": "$.start.output.ticket",
                "customer": "$.start.output.customer",
                "analysis": "$.start.output.analysis",
            }
        ),
    )
    due = Expression(
        id="due",
        name="due",
        input_fields=[NamedField(name="opened_at"), NamedField(name="sla_hours"), NamedField(name="status")],
        expressions=[
            ExpressionItem(key="due_at", expression="date(opened_at) | string"),
            ExpressionItem(key="hours", expression="sla_hours if status == 'pass' else 1"),
            ExpressionItem(key="needs_human", expression="status != 'pass'"),
        ],
        depends=[NodeDependency(queue), NodeDependency(screen)],
        input_transformer=InputTransformer(
            selector={
                "opened_at": "$.start.output.ticket.opened_at",
                "sla_hours": "$.queue.output.sla_hours",
                "status": "$.screen.output.status",
            }
        ),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(queue), NodeDependency(screen), NodeDependency(due)],
        input_transformer=InputTransformer(
            selector={
                "queue": "$.queue.output.queue",
                "hours": "$.due.output.hours",
                "needs_human": "$.due.output.needs_human",
                "findings": "$.screen.output.findings",
                "status": "$.screen.output.status",
            }
        ),
    )
    return Workflow(flow=Flow(nodes=[start, queue, screen, due, end]))


def ticket(**overrides) -> dict:
    record = {
        "ticket": {"subject": "Refund not received", "channel": "email", "language": "en", "opened_at": "2026-09-19"},
        "customer": {"plan": "enterprise", "open_tickets": []},
        "analysis": {"sentiment": 0.1, "topics": ["billing"], "contains_card_number": False},
    }
    for key, value in overrides.items():
        record[key] = record[key] | value
    return record


def test_a_clean_enterprise_ticket_goes_to_tier_two_with_its_sla():
    result = support_triage().run(input_data=ticket(), config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS, result.error
    output = result.output["end"]["output"]
    assert output["queue"] == "tier2" and output["hours"] == 4
    assert output["status"] == "pass" and output["needs_human"] is False


def test_an_angry_customer_mentioning_a_chargeback_is_held_for_a_supervisor():
    record = ticket(
        customer={"plan": "free", "open_tickets": [{"subject": "Refund not received"}]},
        analysis={"sentiment": -0.8, "topics": ["billing", "chargeback"], "contains_card_number": True},
    )

    result = support_triage().run(input_data=record, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS, result.error
    output = result.output["end"]["output"]
    # The table still says where the ticket belongs; the screen says a human sees it first, within the hour.
    assert output["queue"] == "retention"
    assert output["status"] == "fail" and output["needs_human"] is True and output["hours"] == 1
    by_rule = {finding["rule_id"]: finding for finding in output["findings"]}
    assert by_rule["ESC-01"]["status"] == "fail"
    assert by_rule["ESC-01"]["message"] == "Ticket mentions billing, chargeback; route to a supervisor"
    assert by_rule["ESC-02"]["status"] == "warn"
    assert by_rule["PII-01"]["status"] == "fail"
    assert [f["reason_code"] for f in output["findings"] if f["status"] in ("fail", "warn")] == [
        "ESC-01",
        "ESC-02",
        "PII-01",
    ]


def test_a_ticket_in_another_language_is_only_noted():
    record = ticket(ticket={"language": "fr"})

    result = support_triage().run(input_data=record, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS, result.error
    output = result.output["end"]["output"]
    by_rule = {finding["rule_id"]: finding for finding in output["findings"]}
    assert by_rule["LANG-01"]["status"] == "info"
    assert by_rule["ESC-02"]["status"] == "not_applicable"
    assert output["status"] == "pass" and output["hours"] == 4
