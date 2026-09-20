import json
from datetime import date

import pytest
from pydantic import BaseModel

from dynamiq.nodes.operators.rules import (
    EVALUATION_ERRORS,
    RecordSandbox,
    RuleUndefined,
    concrete,
    days_between,
    has,
    read_paths,
    read_template,
    reserved_read,
    resolve_path,
    scope_for,
    to_date,
)


@pytest.mark.parametrize(
    ("expression", "required", "optional"),
    [
        (
            "docs.Note.interest_rate == docs.ClosingDisclosure.interest_rate",
            ["docs.Note.interest_rate", "docs.ClosingDisclosure.interest_rate"],
            [],
        ),
        ("loan.dti <= limits[loan.program].max_dti", ["loan.dti", "limits", "loan.program"], []),
        ("has(docs.FloodCert)", [], ["docs.FloodCert"]),
        ("docs.FloodCert is defined and loan.zone in ['A', 'AE']", ["loan.zone"], ["docs.FloodCert"]),
        ("(loan.fees | default(0)) < 3000", [], ["loan.fees"]),
        ("days_between(docs.Appraisal.date, docs.Note.date) <= 120", ["docs.Appraisal.date", "docs.Note.date"], []),
        ("items[0].amount > 0 and items['first'].amount > 0", ["items[0].amount", "items.first.amount"], []),
        ("ltv <= 0.8 and ltv > 0", ["ltv"], []),
        ("len(docs.pages) > 3", ["docs.pages"], []),
        # A read under a guarded path is the guard's to decide, so it must not block the check.
        (
            "has(docs.FloodCert) and docs.FloodCert.zone in ['A', 'V']",
            [],
            ["docs.FloodCert", "docs.FloodCert.zone"],
        ),
        ("zone is defined and zone != 'A'", [], ["zone"]),
        ("has(items[0]) and items[0].amount > 0 and total > 0", ["total"], ["items[0]", "items[0].amount"]),
        # A method call reads the object it is called on and its arguments, not a member of the method's name.
        ("invoice.get('vat_rate', 0) > 0", ["invoice"], []),
        ("subject.lower().startswith('urgent')", ["subject"], []),
        ("invoice['lines'].count(item) > 0 and (rec.keys() | list | length) > 0", ["invoice.lines", "item", "rec"], []),
        ("has(invoice.get('vat_rate'))", [], ["invoice"]),
        # A helper's name is a member where it is read as a value and a call where it is called.
        ("has(date)", [], ["date"]),
        ("date > '2026-01-01' and len(items) > 0", ["date", "items"], []),
        ("days_between(opened, date(today())) < 30", ["opened"], []),
    ],
)
def test_read_paths_tells_required_from_optional(expression, required, optional):
    reads = read_paths(expression)

    assert reads.required == required
    assert reads.optional == optional


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("loan.amount", 100),
        ("docs.Note.rate", 6.5),
        ("docs.pages[1]", "b"),
        ("docs.pages[-1]", "c"),
        ("ltv", 0.8),
    ],
)
def test_resolve_path_walks_dicts_and_lists(path, expected):
    context = {"loan": {"amount": 100}, "docs": {"Note": {"rate": 6.5}, "pages": ["a", "b", "c"]}, "ltv": 0.8}

    assert resolve_path(context, path) == expected


@pytest.mark.parametrize(
    "path", ["loan.missing", "docs.Appraisal.date", "docs.pages[7]", "nothing", "loan.amount.deeper"]
)
def test_resolve_path_reports_a_missing_step(path):
    context = {"loan": {"amount": 100}, "docs": {"pages": ["a"]}}

    assert has(resolve_path(context, path)) is False


def test_helpers_read_dates_in_the_shapes_documents_carry():
    assert to_date("2026-08-01") == date(2026, 8, 1)
    assert to_date("2026-08-01T10:15:00Z") == date(2026, 8, 1)
    assert to_date("08/01/2026") == date(2026, 8, 1)
    assert days_between("2026-08-01", "2026-08-31") == 30
    assert days_between("2026-08-31", "2026-08-01") == -30
    with pytest.raises(ValueError):
        to_date("August first")
    with pytest.raises(ValueError):
        days_between(None, "2026-08-01")


def test_a_key_that_is_not_a_plain_name_stays_one_segment():
    reads = read_paths("docs['Flood.Cert'].pages > 0 and docs['FloodCert'].pages > 0")
    assert reads.required == ["docs['Flood.Cert'].pages", "docs.FloodCert.pages"]
    assert resolve_path({"docs": {"Flood.Cert": {"pages": 3}}}, "docs['Flood.Cert'].pages") == 3
    assert read_paths('docs["it\'s"].n > 0').required == ["docs['it\\'s'].n"]
    assert resolve_path({"docs": {"it's": {"n": 1}}}, "docs['it\\'s'].n") == 1
    assert resolve_path({"items": [{"name": "a"}]}, "items[0].name") == "a"


def test_a_helper_name_is_a_member_where_it_is_read_and_a_call_where_it_is_called():
    reads = read_paths("has(date) and days_between(opened, date) >= 0")

    assert (reads.required, reads.optional) == (["opened"], ["date"])
    assert reads.helpers_called == ("has", "days_between")
    assert reads.helpers_read == ("date",)
    assert read_paths("date(opened) <= today()").helpers_read == ()
    assert read_template("{{ date }} after {{ days_between(opened, date) }} days").helpers_read == ("date",)


def test_the_scope_hides_a_member_where_the_helper_is_called_and_marks_an_absent_member_undefined():
    scope = {"date": "2026-09-01", "opened": "2026-08-20"}

    assert scope_for(read_paths("date(opened)"), scope, RuleUndefined) == {"opened": "2026-08-20"}
    assert scope_for(read_paths("has(date)"), scope, RuleUndefined) is scope
    marked = scope_for(read_paths("has(date)"), {}, RuleUndefined)
    assert isinstance(marked["date"], RuleUndefined) and not has(marked["date"])


def test_concrete_materializes_a_lazy_iterable_and_leaves_a_string_and_an_object_alone():
    class Record(BaseModel):
        name: str = "doc"

    record = Record()
    undefined = RuleUndefined(name="gone")
    value = concrete(
        {
            "generator": (item for item in [1, undefined]),
            "mapped": map(str, [1, 2]),
            "keys": {"a": 1, "b": 2}.keys(),
            "items": {"a": undefined}.items(),
            "span": range(3),
            "unique": {3},
            "text": "abc",
            "record": record,
        }
    )
    assert value == {
        "generator": [1, None],
        "mapped": ["1", "2"],
        "keys": ["a", "b"],
        "items": [("a", None)],
        "span": [0, 1, 2],
        "unique": [3],
        "text": "abc",
        "record": record,
    }
    assert value["record"] is record
    assert json.dumps({key: item for key, item in value.items() if key != "record"})


def test_concrete_replaces_an_undefined_member_at_any_depth():
    undefined = RuleUndefined(name="gone")

    assert concrete(undefined) is None
    assert concrete([1, undefined, (2, undefined)]) == [1, None, (2, None)]
    assert concrete({"a": undefined, "b": {"c": [undefined]}}) == {"a": None, "b": {"c": [None]}}
    assert concrete({"a": 1, "b": [2, "x"]}) == {"a": 1, "b": [2, "x"]}


@pytest.mark.parametrize(
    "expression, reserved",
    [
        ("has(self)", "self"),
        ("self.href", "self.href"),
        ("self['id'] > 5", "self.id"),
        ("payload.self.href", None),
        ("selfish > 1", None),
    ],
)
def test_a_read_rooted_at_self_is_the_one_read_an_expression_cannot_make(expression, reserved):
    assert reserved_read(read_paths(expression)) == reserved


def test_a_record_key_is_read_before_a_method_of_the_same_name():
    sandbox = RecordSandbox(undefined=RuleUndefined)
    record = {"items": [1, 2], "update": "pending", "nested": {"keys": ["k"]}}

    assert sandbox.getattr(record, "items") == [1, 2]
    assert sandbox.getattr(record, "update") == "pending"
    assert sandbox.getattr(record["nested"], "keys") == ["k"]
    # A key the record lacks reaches the method, which is what `invoice.get('vat_rate', 0)` relies on.
    assert sandbox.getattr(record, "get")("vat_rate", 0) == 0
    assert isinstance(sandbox.getattr(record, "absent"), RuleUndefined)


def test_using_an_attribute_the_sandbox_refuses_is_an_evaluation_error():
    sandbox = RecordSandbox(undefined=RuleUndefined)
    expression = sandbox.compile_expression("tags.append == 1", undefined_to_none=False)

    with pytest.raises(EVALUATION_ERRORS):
        expression(tags=["a"])
