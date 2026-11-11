import json
import traceback

import pytest

from dynamiq.evaluations.memory.evidence import MemoryEvidence, MemorySpecification
from dynamiq.evaluations.memory.serialization import (
    dump_evidence,
    dump_suite,
    load_evidence,
    load_suite,
    read_evidence,
    write_evidence,
)
from dynamiq.evaluations.memory.suite import MemoryCase, MemoryEvaluationLimits, MemorySuite, Remember


def suite():
    return MemorySuite(
        name="suite",
        cutoffs=(1,),
        cases=(MemoryCase(id="case", steps=(Remember(id="w", actor="subject", alias="a", content="PRIVATE"),)),),
    )


def evidence():
    return MemoryEvidence(run_id="run", specification=MemorySpecification.from_suite(suite()))


def test_explicit_raw_suite_export_and_text_free_evidence_roundtrip():
    assert load_suite(dump_suite(suite())) == suite()
    assert "PRIVATE" in dump_suite(suite())
    assert "PRIVATE" not in dump_evidence(evidence())
    assert load_evidence(dump_evidence(evidence())) == evidence()


@pytest.mark.parametrize(
    "raw",
    [
        '{"version":1,"version":1}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1e999}',
        '{"x":"\\ud800"}',
        "\ufeff{}",
        '{"x":[[[[[]]]]]}',
    ],
)
def test_reject_nonportable_ambiguous_or_deep_json(raw):
    with pytest.raises(ValueError):
        load_evidence(raw, limits=MemoryEvaluationLimits(max_json_depth=4))


def test_versions_are_strict_and_limits_cannot_be_raised_in_file():
    data = json.loads(dump_evidence(evidence()))
    for bad in (True, 1.0, "1", 2):
        with pytest.raises(ValueError):
            load_evidence(json.dumps({**data, "version": bad}))
    with pytest.raises(ValueError):
        load_evidence(json.dumps({**data, "limits": {"max_bytes": 999999999}}))
    with pytest.raises(ValueError):
        load_evidence(dump_evidence(evidence()), limits=MemoryEvaluationLimits(max_bytes=10))


def test_atomic_write_never_overwrites_without_explicit_request(tmp_path):
    target = tmp_path / "evidence.json"
    target.write_text("keep")
    with pytest.raises(FileExistsError):
        write_evidence(target, evidence())
    assert target.read_text() == "keep"
    write_evidence(target, evidence(), overwrite=True)
    assert read_evidence(target) == evidence()
    assert list(tmp_path.iterdir()) == [target]


def test_serialization_errors_do_not_echo_raw_input():
    raw = dump_suite(suite()).replace('"content": "PRIVATE"', '"content": 123')
    # Regardless of pretty-print spacing, use an unambiguously invalid field.
    data = json.loads(raw)
    data["suite"]["cases"][0]["steps"][0]["content"] = {"secret": "PRIVATE"}
    with pytest.raises(ValueError) as error:
        load_suite(json.dumps(data))
    assert "PRIVATE" not in str(error.value)
    assert "PRIVATE" not in "".join(traceback.format_exception(error.value))


def test_saved_report_roundtrip_recomputes_derived_results(tmp_path):
    from dynamiq.evaluations.memory.replay import replay_memory
    from dynamiq.evaluations.memory.serialization import dump_report, load_report, read_report

    original = replay_memory(evidence())
    data = json.loads(dump_report(original))
    data["verdict"] = "pass"
    data["completed_operations"] = 999
    serialized = json.dumps(data)
    assert load_report(serialized) == original
    path = tmp_path / "report.json"
    path.write_text(serialized)
    assert read_report(path) == original
    data["version"] = True
    with pytest.raises(ValueError, match="schema"):
        load_report(json.dumps(data))
