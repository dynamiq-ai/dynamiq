import pytest
from pydantic import ValidationError

from dynamiq.evaluations.memory import MemoryCase, MemoryEvaluationLimits, MemorySuite, Recall, Relevance, Remember


def test_metric_cell_budget_rejects_large_query_cutoff_panels():
    suite = MemorySuite(
        name="wide",
        cutoffs=(1, 2),
        cases=(MemoryCase(id="c", steps=(Recall(id="q", actor="subject", query="x", limit=2),)),),
    )
    with pytest.raises(ValueError, match="metric value limit exceeded"):
        suite.check_limits(MemoryEvaluationLimits(max_metric_values=10))


def test_empty_query_panels_still_consume_group_processing_budget():
    suite = MemorySuite(
        name="writes-only",
        cutoffs=(1,),
        cases=(MemoryCase(id="c", steps=(Remember(id="w", actor="subject", alias="a", content="x"),)),),
        groups=({"name": "slice", "cases": ("c",)},),
    )
    with pytest.raises(ValueError, match="group observation limit exceeded"):
        suite.check_limits(MemoryEvaluationLimits(max_group_observations=10))


def study():
    return MemorySuite(
        name="memory-regression",
        cutoffs=(1,),
        cases=(
            MemoryCase(
                id="preference",
                actors=("alice",),
                steps=(
                    Remember(id="write", actor="alice", alias="preference-v1", content="Likes tea"),
                    Recall(
                        id="query",
                        actor="alice",
                        query="drink",
                        limit=1,
                        relevance=(Relevance(alias="preference-v1", grade=3),),
                    ),
                ),
            ),
        ),
    )


def test_suite_round_trip_keeps_order_and_freezes_source_containers():
    suite = study()
    assert MemorySuite.model_validate_json(suite.model_dump_json()) == suite
    assert tuple(step.id for step in suite.cases[0].steps) == ("write", "query")
    with pytest.raises(ValidationError):
        suite.name = "changed"


def test_forward_alias_is_rejected_before_any_backend_is_needed():
    data = study().model_dump()
    data["cases"][0]["steps"] = tuple(reversed(data["cases"][0]["steps"]))
    with pytest.raises(ValidationError, match="earlier"):
        MemorySuite.model_validate(data)


def test_boolean_counts_and_unknown_fields_do_not_coerce():
    data = study().model_dump()
    data["repetitions"] = True
    with pytest.raises(ValidationError):
        MemorySuite.model_validate(data)
    data = study().model_dump()
    data["provider"] = "arbitrary.import"
    with pytest.raises(ValidationError):
        MemorySuite.model_validate(data)


def test_foreign_positive_judgment_is_rejected():
    data = study().model_dump()
    data["cases"][0]["actors"] = ("alice", "bob")
    data["cases"][0]["steps"][1]["actor"] = "bob"
    with pytest.raises(ValidationError, match="actor"):
        MemorySuite.model_validate(data)


def test_trusted_limits_charge_trials_before_running():
    suite = study().model_copy(update={"repetitions": 5})
    with pytest.raises(ValueError, match="operation"):
        suite.check_limits(MemoryEvaluationLimits(max_operations=9))
