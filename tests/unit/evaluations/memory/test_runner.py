from datetime import datetime, timezone

import pytest

from dynamiq.evaluations.memory.runner import evaluate_memory
from dynamiq.evaluations.memory.suite import MemoryCase, MemoryEvaluationLimits, MemorySuite, Recall, Remember, Verdict
from dynamiq.memory.long_term.schemas import Fact
from dynamiq.memory.long_term.types import RememberOutcome


class Backend:
    def __init__(self):
        self.facts = {}
        self.scopes = []

    def remember(self, *, content, user_id):
        now = datetime.now(timezone.utc)
        fact = Fact(
            id="PRIVATE-ID", content=content, user_id=user_id, hash="PRIVATE-HASH", created_at=now, updated_at=now
        )
        self.facts[user_id] = fact
        return fact, RememberOutcome.CREATED

    def recall(self, *, query, user_id, limit):
        return [(self.facts[user_id], 1.0)]

    def clear_user(self, *, user_id):
        self.scopes.append(user_id)
        return int(self.facts.pop(user_id, None) is not None)

    def list_all(self, *, user_id, limit):
        return []


def suite(**kwargs):
    return MemorySuite(
        name="trial",
        cutoffs=(1,),
        cases=(
            MemoryCase(
                id="c",
                steps=(
                    Remember(id="w", actor="subject", alias="a", content="PRIVATE-TEXT"),
                    Recall(
                        id="q", actor="subject", query="PRIVATE-QUERY", limit=1, relevance=({"alias": "a", "grade": 1},)
                    ),
                ),
            ),
        ),
        **kwargs,
    )


def test_complete_run_cleanup_privacy_replay_and_unique_scopes():
    backend = Backend()
    report = evaluate_memory(suite(repetitions=2), lambda: backend)
    assert report.verdict == Verdict.PASS
    assert report.completed_operations == 4
    assert len(set(backend.scopes)) == 2
    assert not backend.facts
    assert "PRIVATE" not in report.model_dump_json()
    assert all(scope not in report.model_dump_json() for scope in backend.scopes)
    assert report.trials[0].cleanup[0].verdict == Verdict.PASS


def test_preflight_precedes_factory_and_revalidates_bypassed_models():
    def unexpected():
        pytest.fail("factory must not be reached")

    with pytest.raises(ValueError):
        evaluate_memory(suite(), unexpected, limits=MemoryEvaluationLimits(max_operations=1))
    with pytest.raises(ValueError):
        evaluate_memory(suite().model_copy(update={"repetitions": True}), unexpected)


def test_factory_and_partial_mutation_failures_keep_denominators():
    def failed_factory():
        raise RuntimeError("PRIVATE ERROR")

    report = evaluate_memory(suite(), failed_factory)
    assert report.verdict == Verdict.INCONCLUSIVE
    assert report.expected_operations == 2
    assert report.completed_operations == 0
    assert "PRIVATE ERROR" not in report.model_dump_json()

    class Failing(Backend):
        def remember(self, **kwargs):
            super().remember(**kwargs)
            raise RuntimeError("PRIVATE ERROR")

    backend = Failing()
    report = evaluate_memory(suite(), lambda: backend)
    assert report.verdict == Verdict.INCONCLUSIVE
    assert not backend.facts
    assert len(backend.scopes) == 1


def test_keyboard_interrupt_still_cleans_all_actors_and_is_reraised():
    class Interrupting(Backend):
        def remember(self, **kwargs):
            raise KeyboardInterrupt()

        def clear_user(self, *, user_id):
            self.scopes.append(user_id)
            raise RuntimeError("cleanup error")

    backend = Interrupting()
    scenario = suite().model_dump()
    scenario["cases"][0]["actors"] = ("subject", "other")
    with pytest.raises(KeyboardInterrupt):
        evaluate_memory(MemorySuite.model_validate(scenario), lambda: backend)
    assert len(backend.scopes) == 2
