"""The golden scenarios run now: their spans rebuild into the runs the handler records for the same run.

The committed fixtures can lag behind the emitter; this test cannot. An emitter change that breaks the "Rebuilding a
run" mapping fails here, and ``generate_fixtures.py`` then refuses to write the fixtures.
"""

import pytest

from dynamiq.telemetry.rebuild import rebuild_runs, spans_from_otlp_json
from tests.integration.telemetry.equivalence import assert_equivalent, to_json
from tests.integration.telemetry.golden_scenarios import SCENARIOS
from tests.integration.telemetry.otlp_json import encode_request, request_to_json


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_spans_of_a_run_rebuild_into_its_handler_runs(scenario):
    result = scenario.execute()
    recorded = [run.to_dict() for run in result.handler.runs.values()]

    rebuilt = rebuild_runs(result.spans, prompt_compaction_enabled=scenario.options.prompt_compaction_enabled)

    assert_equivalent(rebuilt, recorded, result.spans, scenario.options)
    # The same spans read back from their OTLP/JSON encoding rebuild into the same runs.
    from_json = spans_from_otlp_json(request_to_json(encode_request(result.spans)))
    assert to_json(rebuild_runs(from_json, prompt_compaction_enabled=scenario.options.prompt_compaction_enabled)) == (
        to_json(rebuilt)
    )
