import pytest

from dynamiq.evaluations.memory.evidence import MemoryEvidence, MemorySpecification
from dynamiq.evaluations.memory.replay import replay_memory
from dynamiq.evaluations.memory.suite import MemoryCase, MemorySuite, Recall, Remember, Verdict


def suite():
    return MemorySuite(
        name="sample",
        cutoffs=(1,),
        cases=(
            MemoryCase(
                id="case",
                steps=(
                    Remember(id="write", actor="subject", alias="fact", content="PRIVATE CONTENT"),
                    Recall(
                        id="query",
                        actor="subject",
                        query="PRIVATE QUERY",
                        limit=1,
                        relevance=({"alias": "fact", "grade": 1},),
                    ),
                ),
            ),
        ),
    )


def test_missing_trial_preserves_every_expected_slot():
    evidence = MemoryEvidence(run_id="run", specification=MemorySpecification.from_suite(suite()))
    report = replay_memory(evidence)
    assert report.verdict == Verdict.INCONCLUSIVE
    assert report.expected_operations == 2
    assert report.completed_operations == 0
    assert len(report.trials[0].operations) == 2
    assert report.aggregates[0].expected_queries == 1
    assert report.aggregates[0].available_queries == 0
    assert "PRIVATE" not in report.model_dump_json()


def test_replay_rejects_unknown_trials_and_duplicate_trial_slots():
    spec = MemorySpecification.from_suite(suite())
    for trials in (({"case_id": "unknown", "repetition": 0},), ({"case_id": "case", "repetition": 0},) * 2):
        with pytest.raises(ValueError):
            replay_memory(MemoryEvidence(run_id="run", specification=spec, trials=trials))


def test_replay_rejects_a_success_after_missing_mutation():
    observation = {
        "case_id": "case",
        "repetition": 0,
        "operations": [
            {
                "step_id": "query",
                "kind": "recall",
                "status": "completed",
                "recall": {"hits": []},
            }
        ],
    }
    with pytest.raises(ValueError, match="after"):
        replay_memory(
            MemoryEvidence(run_id="run", specification=MemorySpecification.from_suite(suite()), trials=(observation,))
        )
