"""The committed golden fixtures: rebuilding ``otlp.json`` gives the runs in ``runs.json``.

These tests read only the files. A change to ``rebuild_runs`` that breaks the projection fails here, whatever the
emitter does today; ``test_golden_freshness.py`` reruns the scenarios.
"""

import json
from pathlib import Path

import pytest

from dynamiq.telemetry.rebuild import rebuild_runs, spans_from_otlp_json
from tests.integration.telemetry.equivalence import ScenarioOptions, assert_equivalent
from tests.integration.telemetry.otlp_json import request_from_bytes, request_to_json

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES_DIR / "manifest.json").read_text())


def load(name: str):
    directory = FIXTURES_DIR / name
    return (
        json.loads((directory / "otlp.json").read_text()),
        json.loads((directory / "runs.json").read_text()),
        ScenarioOptions.model_validate(MANIFEST[name]),
    )


def test_every_fixture_directory_is_in_the_manifest():
    directories = {path.name for path in FIXTURES_DIR.iterdir() if path.is_dir()}
    assert directories == set(MANIFEST)
    for name in MANIFEST:
        assert {path.name for path in (FIXTURES_DIR / name).iterdir()} == {"otlp.json", "otlp.pb", "runs.json"}


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_rebuilt_runs_match_the_recorded_runs(name):
    request, recorded, options = load(name)
    spans = spans_from_otlp_json(request)

    rebuilt = rebuild_runs(spans, prompt_compaction_enabled=options.prompt_compaction_enabled)

    assert rebuilt
    assert_equivalent(rebuilt, recorded, spans, options)


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_protobuf_and_json_hold_the_same_request(name):
    request, _, _ = load(name)

    decoded = request_from_bytes((FIXTURES_DIR / name / "otlp.pb").read_bytes())

    assert request_to_json(decoded) == request


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_ids_are_hex_and_64_bit_integers_are_strings(name):
    request, _, _ = load(name)

    for resource_spans in request["resourceSpans"]:
        for scope_spans in resource_spans["scopeSpans"]:
            for span in scope_spans["spans"]:
                assert len(span["traceId"]) == 32 and int(span["traceId"], 16)
                assert len(span["spanId"]) == 16 and int(span["spanId"], 16)
                assert "parentSpanId" not in span or len(span["parentSpanId"]) == 16
                assert isinstance(span["startTimeUnixNano"], str) and isinstance(span["endTimeUnixNano"], str)
                assert isinstance(span["kind"], int)


def test_the_paused_and_compacted_fixtures_hold_what_they_are_named_for():
    paused_request, paused_runs, _ = load("paused")
    statuses = {
        attribute["value"]["stringValue"]
        for span in paused_request["resourceSpans"][0]["scopeSpans"][0]["spans"]
        for attribute in span["attributes"]
        if attribute["key"] == "dynamiq.run.status"
    }
    assert "paused" in statuses
    assert {run["status"] for run in paused_runs} == {"succeeded", "failed"}

    _, compacted_runs, options = load("prompt_compaction")
    assert options.prompt_compaction_enabled
    bases = [
        run["metadata"]["node"]["prompt"]["messages_base"]
        for run in compacted_runs
        if run["type"] == "node" and run["metadata"]["node"]["group"] == "llms"
    ]
    assert bases[0] is None and all(base["count"] > 0 for base in bases[1:])
