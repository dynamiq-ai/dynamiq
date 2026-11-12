"""Rewrites the golden fixtures in ``fixtures/`` from the scenarios in ``golden_scenarios.py``.

    DYNAMIQ_GOLDEN_FIXTURES_WRITE_ENABLED=1 uv run --frozen python -m tests.integration.telemetry.generate_fixtures

Names of scenarios as arguments rewrite only those. Without the environment variable nothing is written. A scenario
whose rebuilt runs do not match its handler's runs is not written.
"""

import json
import os
import sys
from pathlib import Path

import orjson

from dynamiq.telemetry.rebuild import rebuild_runs
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import orjson_encode
from tests.integration.telemetry.equivalence import assert_equivalent
from tests.integration.telemetry.golden_scenarios import SCENARIOS, SCENARIOS_BY_NAME, Scenario
from tests.integration.telemetry.otlp_json import encode_request, request_to_json

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MANIFEST = FIXTURES_DIR / "manifest.json"
WRITE_ENV = "DYNAMIQ_GOLDEN_FIXTURES_WRITE_ENABLED"


def write_scenario(scenario: Scenario) -> None:
    result = scenario.execute()
    recorded = [run.to_dict() for run in result.handler.runs.values()]
    rebuilt = rebuild_runs(result.spans, prompt_compaction_enabled=scenario.options.prompt_compaction_enabled)
    assert_equivalent(rebuilt, recorded, result.spans, scenario.options)

    request = encode_request(result.spans)
    directory = FIXTURES_DIR / scenario.name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "otlp.pb").write_bytes(request.SerializeToString())
    (directory / "otlp.json").write_text(json.dumps(request_to_json(request), indent=1) + "\n")
    # Encoded as DynamiqTracingClient encodes runs, indented for review.
    runs = orjson.dumps(
        recorded,
        default=orjson_encode,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
    )
    (directory / "runs.json").write_bytes(runs)


def write_manifest() -> None:
    manifest = {scenario.name: scenario.options.model_dump() for scenario in SCENARIOS}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def main(names: list[str]) -> int:
    if os.environ.get(WRITE_ENV) != "1":
        logger.error(f"Set {WRITE_ENV}=1 to rewrite the fixtures in {FIXTURES_DIR}.")
        return 2
    unknown = [name for name in names if name not in SCENARIOS_BY_NAME]
    if unknown:
        logger.error(f"Unknown scenarios: {unknown}. Known: {sorted(SCENARIOS_BY_NAME)}")
        return 2
    for scenario in [SCENARIOS_BY_NAME[name] for name in names] or SCENARIOS:
        write_scenario(scenario)
        logger.info(f"Wrote {FIXTURES_DIR / scenario.name}")
    write_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
