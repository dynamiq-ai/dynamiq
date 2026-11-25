import math

import pytest

from dynamiq.evaluations.memory.evidence import CapturedVersion, ExpectedRecall, RecallHit, RecallObservation
from dynamiq.evaluations.memory.metrics import score_recall
from dynamiq.evaluations.memory.suite import MemoryMetric, Relevance, Verdict


def aliases():
    return {
        "a": CapturedVersion(actor="user", object_token="o1", version_token="v1"),
        "b": CapturedVersion(actor="user", object_token="o2", version_token="v2"),
    }


def value(report, metric, cutoff):
    return next(item.value for item in report.metrics if item.metric == metric and item.cutoff == cutoff)


def test_hand_calculated_rank_metrics_keep_unknown_rank_and_duplicate_penalty():
    expected = ExpectedRecall(
        id="query", actor="user", limit=5, relevance=(Relevance(alias="a", grade=3), Relevance(alias="b", grade=1))
    )
    observed = RecallObservation(
        hits=(
            RecallHit(rank=1, score=0.9),
            RecallHit(rank=2, object_token="o1", version_token="v1", score=0.8),
            RecallHit(rank=3, object_token="o1", version_token="v1", score=0.7, duplicate=True),
            RecallHit(rank=4, object_token="o2", version_token="v2", score=0.6),
        )
    )
    report = score_recall(expected, observed, aliases(), (1, 3, 5))
    assert value(report, MemoryMetric.PRECISION, 3) == pytest.approx(1 / 3)
    assert value(report, MemoryMetric.RECALL, 3) == 0.5
    assert value(report, MemoryMetric.RECIPROCAL_RANK, 3) == 0.5
    assert value(report, MemoryMetric.PRECISION, 5) == 0.4
    expected_ndcg = (7 / math.log2(3) + 1 / math.log2(5)) / (7 + 1 / math.log2(3))
    assert value(report, MemoryMetric.NDCG, 5) == pytest.approx(expected_ndcg)
    assert value(report, MemoryMetric.DUPLICATE_QUERY_RATE, 3) == 1


def test_equivalent_aliases_count_once_and_conflicting_grades_reject():
    refs = aliases()
    refs["a-again"] = refs["a"]
    expected = ExpectedRecall(
        id="q", actor="user", limit=1, relevance=(Relevance(alias="a", grade=3), Relevance(alias="a-again", grade=3))
    )
    result = score_recall(
        expected,
        RecallObservation(hits=(RecallHit(rank=1, score=1, object_token="o1", version_token="v1"),)),
        refs,
        (1,),
    )
    assert value(result, MemoryMetric.RECALL, 1) == 1
    conflicting = expected.model_copy(
        update={"relevance": (Relevance(alias="a", grade=3), Relevance(alias="a-again", grade=1))}
    )
    with pytest.raises(ValueError, match="conflicting"):
        score_recall(conflicting, RecallObservation(), refs, (1,))


def test_no_positive_references_is_undefined_not_perfect_recall():
    expected = ExpectedRecall(id="q", actor="user", limit=5, expect_empty=True)
    report = score_recall(expected, RecallObservation(), {}, (1, 5))
    assert value(report, MemoryMetric.RECALL, 1) is None
    assert value(report, MemoryMetric.RECIPROCAL_RANK, 1) is None
    assert next(item for item in report.assertions if item.name == "expected_empty").verdict == Verdict.PASS


def test_empty_assertion_checks_ranks_beyond_smallest_cutoff():
    expected = ExpectedRecall(id="q", actor="user", limit=5, expect_empty=True)
    report = score_recall(expected, RecallObservation(hits=(RecallHit(rank=1, score=0.2),)), {}, (1,))
    assert next(item for item in report.assertions if item.name == "expected_empty").verdict == Verdict.FAIL


def test_scope_and_invalid_outputs_fail_invariants_without_positive_credit():
    expected = ExpectedRecall(id="q", actor="user", limit=2, relevance=(Relevance(alias="a", grade=3),))
    report = score_recall(
        expected,
        RecallObservation(
            hits=(
                RecallHit(rank=1, score=0.9, scope_valid=False),
                RecallHit(rank=2, output_valid=False),
            )
        ),
        aliases(),
        (1, 2),
    )
    assert value(report, MemoryMetric.RECALL, 2) == 0
    assert {item.name for item in report.assertions if item.verdict == Verdict.FAIL} == {
        "scope_isolation",
        "valid_output",
    }


def test_two_positive_versions_of_same_object_are_an_impossible_reference_set():
    refs = aliases()
    refs["b"] = CapturedVersion(actor="user", object_token="o1", version_token="v2")
    expected = ExpectedRecall(
        id="q", actor="user", limit=2, relevance=(Relevance(alias="a", grade=1), Relevance(alias="b", grade=1))
    )
    with pytest.raises(ValueError, match="positive versions"):
        score_recall(expected, RecallObservation(), refs, (1,))
