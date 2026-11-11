from datetime import datetime, timezone

import pytest

from dynamiq.evaluations.memory.comparison import MemoryComparisonGate, compare_memory
from dynamiq.evaluations.memory.runner import evaluate_memory
from dynamiq.evaluations.memory.suite import MemoryCase, MemorySuite, Recall, Remember, Verdict
from dynamiq.memory.long_term.schemas import Fact


class ScriptedBackend:
    def __init__(self, answers):
        self.answers = iter(answers)

    def remember(self, *, content, user_id):
        now = datetime.now(timezone.utc)
        self.fact = Fact(id="id", content=content, user_id=user_id, hash="hash", created_at=now, updated_at=now)
        return self.fact, "created"

    def recall(self, **kwargs):
        answer = next(self.answers)
        if answer is None:
            raise RuntimeError("unavailable")
        return [(self.fact, 1.0)] if answer else []

    def clear_user(self, **kwargs):
        return 1

    def list_all(self, **kwargs):
        return []


def scenario(query_counts=(1,), **kwargs):
    return MemorySuite(
        name="comparison",
        dataset_revision="revision-1",
        cutoffs=(1,),
        cases=tuple(
            MemoryCase(
                id=f"c{index}",
                steps=(Remember(id="w", actor="subject", alias="a", content="x"),)
                + tuple(
                    Recall(id=f"q{q}", actor="subject", query="x", limit=1, relevance=({"alias": "a", "grade": 1},))
                    for q in range(count)
                ),
            )
            for index, count in enumerate(query_counts)
        ),
        **kwargs,
    )


def run(suite, outcomes):
    answers = iter(outcomes)
    return evaluate_memory(suite, lambda: ScriptedBackend(next(answers))).evidence


def row(result, metric="recall"):
    return next(item for item in result.metrics if item.metric == metric)


def test_paired_comparison_equal_weights_cases_and_has_no_implicit_gate():
    suite = scenario((3, 1))
    base = run(suite, [[False] * 3, [True]])
    candidate = run(suite, [[True] * 3, [True]])
    result = compare_memory(base, candidate)
    assert row(result).baseline_mean == 0.5
    assert row(result).candidate_mean == 1.0
    assert row(result).improvement == 0.5
    assert result.verdict == Verdict.NOT_EVALUATED
    gated = compare_memory(
        base, candidate, gates=(MemoryComparisonGate(metric="recall", cutoff=1, minimum_improvement=0.4),)
    )
    assert gated.verdict == Verdict.PASS


def test_missing_candidate_query_cannot_manufacture_a_pass():
    suite = scenario((2,))
    result = compare_memory(
        run(suite, [[False, True]]),
        run(suite, [[True, None]]),
        gates=(MemoryComparisonGate(metric="recall", cutoff=1, minimum_improvement=0.0),),
    )
    score = row(result)
    assert score.baseline_mean == 0.0
    assert score.candidate_mean == 1.0
    assert score.paired_queries == 1
    assert score.expected_queries == 2
    assert score.coverage == 0.5
    assert result.verdict == Verdict.INCONCLUSIVE


def test_source_absolute_gates_are_reported_but_do_not_silently_become_comparison_gates():
    suite = scenario(gates=({"metric": "recall", "cutoff": 1, "minimum": 1.0},))
    base = run(suite, [[False]])
    result = compare_memory(base, base)
    assert result.candidate_absolute_verdict == Verdict.FAIL
    assert result.verdict == Verdict.NOT_EVALUATED


def test_missing_or_different_revision_and_structure_reject():
    suite = scenario()
    base = run(suite, [[True]])
    for candidate in (
        base.model_copy(update={"specification": base.specification.model_copy(update={"dataset_revision": None})}),
        base.model_copy(update={"specification": base.specification.model_copy(update={"dataset_revision": "other"})}),
    ):
        with pytest.raises(ValueError):
            compare_memory(base, candidate)


def test_changed_alias_coalescing_is_not_a_quality_improvement():
    suite = scenario()
    data = suite.model_dump(mode="json")
    data["cases"][0]["steps"].insert(
        1, {"kind": "remember", "id": "w2", "actor": "subject", "alias": "b", "content": "x"}
    )
    data["cases"][0]["steps"][-1]["relevance"] = [{"alias": "a", "grade": 1}, {"alias": "b", "grade": 1}]
    suite = MemorySuite.model_validate(data)

    class DedupBackend(ScriptedBackend):
        def __init__(self, dedup):
            super().__init__([True])
            self.dedup = dedup
            self.writes = 0

        def remember(self, **kwargs):
            self.writes += 1
            fact, outcome = super().remember(**kwargs)
            if self.writes == 2:
                if self.dedup:
                    return fact, "unchanged"
                fact.id = "second"
            return fact, outcome

    baseline = evaluate_memory(suite, lambda: DedupBackend(False)).evidence
    candidate = evaluate_memory(suite, lambda: DedupBackend(True)).evidence
    result = compare_memory(
        baseline, candidate, gates=(MemoryComparisonGate(metric="recall", cutoff=1, minimum_improvement=0.0),)
    )
    assert row(result).incomparable_queries == 1
    assert row(result).paired_queries == 0
    assert row(result).improvement is None
    assert result.verdict == Verdict.INCONCLUSIVE


def test_run_local_tokens_and_trial_record_order_do_not_affect_pairing():
    suite = scenario((1, 1))
    base = run(suite, [[True], [False]])
    data = base.model_dump(mode="json")
    data["trials"].reverse()

    def rename(value):
        if isinstance(value, dict):
            return {
                key: (
                    f"renamed-{item}" if key in ("object_token", "version_token") and item is not None else rename(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rename(item) for item in value]
        return value

    candidate = type(base).model_validate(rename(data))
    result = compare_memory(base, candidate)
    assert row(result).paired_queries == 2
    assert row(result).delta == 0.0


def test_lower_adverse_rate_is_positive_improvement():
    candidate = run(scenario(), [[True]])
    data = candidate.model_dump(mode="json")
    hit = data["trials"][0]["operations"][1]["recall"]["hits"][0]
    hit.update(object_token=None, version_token=None)
    baseline = type(candidate).model_validate(data)
    result = compare_memory(
        baseline,
        candidate,
        gates=(MemoryComparisonGate(metric="unknown_query_rate", cutoff=1, minimum_improvement=1.0),),
    )
    assert row(result, "unknown_query_rate").delta == -1.0
    assert row(result, "unknown_query_rate").improvement == 1.0
    assert result.verdict == Verdict.PASS


def test_candidate_integrity_failure_blocks_passing_comparison_gate():
    base = run(scenario(), [[True]])
    data = base.model_dump(mode="json")
    data["trials"][0]["operations"][0]["remember"]["write_matches"] = False
    candidate = type(base).model_validate(data)
    result = compare_memory(
        base, candidate, gates=(MemoryComparisonGate(metric="recall", cutoff=1, minimum_improvement=0.0),)
    )
    assert result.gates[0].verdict == Verdict.PASS
    assert result.candidate_lifecycle_verdict == Verdict.FAIL
    assert result.verdict == Verdict.FAIL


def test_comparison_renderer_ignores_saved_summary_tampering():
    from dynamiq.evaluations.memory.comparison_reporting import render_memory_comparison

    evidence = run(scenario(), [[True]])
    result = compare_memory(evidence, evidence)
    text = render_memory_comparison(result.model_copy(update={"verdict": "pass"}), format="json")
    assert '"verdict": "not_evaluated"' in text
    assert "PRIVATE" not in text


@pytest.mark.parametrize("limits", [{"max_json_nodes": 1}, {"max_json_depth": 1}])
def test_comparison_json_honors_trusted_structure_limits(limits):
    from dynamiq.evaluations.memory.comparison_reporting import render_memory_comparison
    from dynamiq.evaluations.memory.suite import MemoryEvaluationLimits

    evidence = run(scenario(), [[True]])
    result = compare_memory(evidence, evidence)
    with pytest.raises(ValueError, match="limit exceeded"):
        render_memory_comparison(result, format="json", limits=MemoryEvaluationLimits(**limits))
