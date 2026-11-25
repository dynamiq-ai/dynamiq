from datetime import datetime, timezone

import pytest

from dynamiq.evaluations.memory.runner import evaluate_memory
from dynamiq.evaluations.memory.suite import MemoryCase, MemoryGate, MemorySuite, Recall, Remember, Verdict
from dynamiq.memory.long_term.schemas import Fact
from dynamiq.memory.long_term.types import RememberOutcome


class ConditionalBackend:
    def remember(self, *, content, user_id):
        now = datetime.now(timezone.utc)
        self.fact = Fact(id="f", content=content, user_id=user_id, hash="h", created_at=now, updated_at=now)
        return self.fact, RememberOutcome.CREATED

    def recall(self, *, query, **kwargs):
        if query == "error":
            raise RuntimeError("failed query")
        return [(self.fact, 1.0)] if query == "hit" else []

    def clear_user(self, **kwargs):
        return 1

    def list_all(self, **kwargs):
        return []


def case(name, queries, *, judged=True):
    return MemoryCase(
        id=name,
        steps=(Remember(id="w", actor="subject", alias="a", content="x"),)
        + tuple(
            Recall(
                id=f"q{i}",
                actor="subject",
                query=query,
                limit=1,
                relevance=({"alias": "a", "grade": 1},) if judged else (),
            )
            for i, query in enumerate(queries)
        ),
    )


def score(report, metric="recall"):
    return next(row for row in report.aggregates if row.metric == metric)


def test_long_case_does_not_dominate_short_case():
    report = evaluate_memory(
        MemorySuite(
            name="equal-cases",
            cutoffs=(1,),
            cases=(
                case("long", ["hit"] * 9),
                case("short", ["miss"]),
            ),
        ),
        ConditionalBackend,
    )
    assert score(report).value == 0.5
    assert score(report).supported_queries == 10


def test_missing_repetition_is_visible_even_with_perfect_observed_mean():
    index = 0

    def factory():
        nonlocal index
        index += 1
        if index == 2:
            raise RuntimeError("factory unavailable")
        return ConditionalBackend()

    report = evaluate_memory(
        MemorySuite(
            name="missing",
            cutoffs=(1,),
            repetitions=2,
            cases=(case("only", ["hit"]),),
            gates=(MemoryGate(metric="recall", cutoff=1, minimum=0.9),),
        ),
        factory,
    )
    aggregate = score(report)
    assert aggregate.value == 1.0
    assert aggregate.coverage == 0.5
    assert aggregate.expected_trials == 2
    assert aggregate.supported_trials == 1
    assert report.gates[0].verdict == Verdict.INCONCLUSIVE


def test_observed_violation_wins_over_missing_coverage():
    report = evaluate_memory(
        MemorySuite(
            name="violation",
            cutoffs=(1,),
            cases=(case("only", ["miss", "error"]),),
            gates=(MemoryGate(metric="recall", cutoff=1, minimum=0.9),),
        ),
        ConditionalBackend,
    )
    assert report.gates[0].verdict == Verdict.FAIL
    assert report.verdict == Verdict.FAIL
    assert score(report).coverage == 0.5


def test_structural_undefinedness_does_not_become_zero_or_one():
    report = evaluate_memory(
        MemorySuite(
            name="unjudged",
            cutoffs=(1,),
            cases=(case("only", ["hit"], judged=False),),
            gates=(MemoryGate(metric="recall", cutoff=1, minimum=0.0),),
        ),
        ConditionalBackend,
    )
    aggregate = score(report)
    assert aggregate.value is None
    assert aggregate.coverage == 1.0
    assert aggregate.undefined_queries == 1
    assert report.gates[0].verdict == Verdict.INCONCLUSIVE


@pytest.mark.parametrize("metric", ["duplicate_query_rate", "unknown_query_rate", "scope_violation_query_rate"])
def test_adverse_rate_has_explicit_upper_bound(metric):
    report = evaluate_memory(
        MemorySuite(
            name="adverse",
            cutoffs=(1,),
            cases=(case("c", ["hit"]),),
            gates=(MemoryGate(metric=metric, cutoff=1, maximum=0.0),),
        ),
        ConditionalBackend,
    )
    assert report.gates[0].verdict == Verdict.PASS
