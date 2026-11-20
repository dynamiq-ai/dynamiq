import json

from click.testing import CliRunner

from dynamiq.cli.commands.evaluation import evaluation
from dynamiq.evaluations.memory.evidence import MemoryEvidence, MemorySpecification
from dynamiq.evaluations.memory.serialization import dump_evidence, dump_suite
from dynamiq.evaluations.memory.suite import MemoryCase, MemorySuite, Recall


def scenario():
    return MemorySuite(
        name="offline",
        cutoffs=(1,),
        cases=(
            MemoryCase(id="c", steps=(Recall(id="q", actor="subject", query="PRIVATE", limit=1, expect_empty=True),)),
        ),
    )


def test_validate_and_replay_do_not_create_api_client(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("offline CLI must not create ApiClient")

    monkeypatch.setattr("dynamiq.cli.commands.context.ApiClient", unexpected)
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(dump_suite(scenario()))
    runner = CliRunner()
    result = runner.invoke(evaluation, ["memory", "validate", str(suite_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["expected_operations"] == 1
    assert "PRIVATE" not in result.output
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        dump_evidence(MemoryEvidence(run_id="r", specification=MemorySpecification.from_suite(scenario())))
    )
    output = tmp_path / "report.json"
    result = runner.invoke(
        evaluation, ["memory", "replay", str(evidence_path), "--format", "json", "--out", str(output)]
    )
    assert result.exit_code == 2, result.output
    assert json.loads(output.read_text())["verdict"] == "inconclusive"
    result = runner.invoke(evaluation, ["memory", "replay", str(evidence_path), "--no-check"])
    assert result.exit_code == 0


def test_cli_reports_errors_without_echoing_scenario_text(tmp_path):
    path = tmp_path / "bad.json"
    data = json.loads(dump_suite(scenario()))
    data["suite"]["cases"][0]["steps"][0]["query"] = {"PRIVATE": "secret"}
    path.write_text(json.dumps(data))
    result = CliRunner().invoke(evaluation, ["memory", "validate", str(path)])
    assert result.exit_code != 0
    assert "PRIVATE" not in result.output


def test_cli_output_requires_explicit_overwrite(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        dump_evidence(MemoryEvidence(run_id="r", specification=MemorySpecification.from_suite(scenario())))
    )
    output = tmp_path / "report.md"
    output.write_text("keep")
    result = CliRunner().invoke(evaluation, ["memory", "replay", str(evidence_path), "--out", str(output)])
    assert result.exit_code == 1
    assert output.read_text() == "keep"


def test_offline_compare_checks_revision_and_requires_complete_gate_options(tmp_path):
    suite = scenario().model_copy(update={"dataset_revision": "sample-v1"})
    path = tmp_path / "evidence.json"
    path.write_text(dump_evidence(MemoryEvidence(run_id="r", specification=MemorySpecification.from_suite(suite))))
    runner = CliRunner()
    result = runner.invoke(evaluation, ["memory", "compare", str(path), str(path), "--format", "json", "--no-check"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["verdict"] == "inconclusive"
    result = runner.invoke(evaluation, ["memory", "compare", str(path), str(path), "--metric", "recall"])
    assert result.exit_code == 2
    assert "must be supplied together" in result.output
