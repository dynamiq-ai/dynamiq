"""Offline reconstruction of complete trial ledgers and derived requirements."""

from collections.abc import Sequence

from dynamiq.evaluations.memory.aggregation import aggregate_memory, evaluate_gates
from dynamiq.evaluations.memory.evidence import (
    CapturedVersion,
    CleanupObservation,
    ExpectedCase,
    ExpectedForget,
    ExpectedRecall,
    ExpectedRemember,
    MemoryEvidence,
    StepObservation,
    TrialObservation,
)
from dynamiq.evaluations.memory.metrics import MemoryAssertion, assertion, score_recall, unavailable_recall
from dynamiq.evaluations.memory.results import (
    MemoryEvaluationReport,
    MemoryOperationEvaluation,
    MemoryTrialEvaluation,
    combine_verdicts,
)
from dynamiq.evaluations.memory.suite import MemoryEvaluationLimits, OperationStatus, Verdict
from dynamiq.memory.long_term.types import ForgetStatus, RememberOutcome


def _unavailable(step: ExpectedRemember | ExpectedRecall | ExpectedForget, *, missing_trial: bool) -> StepObservation:
    return StepObservation(
        step_id=step.id,
        kind=step.kind,
        status=OperationStatus.UNAVAILABLE,
        unavailable_reason="missing_trial" if missing_trial else "missing_step",
    )


def _cleanup_results(
    case: ExpectedCase,
    observations: Sequence[CleanupObservation],
    *,
    cleanup_enabled: bool,
    started: bool,
) -> tuple[MemoryAssertion, ...]:
    actors = set(case.actors)
    index = {}
    for observation in observations:
        if observation.actor not in actors or observation.actor in index:
            raise ValueError("cleanup references an unknown or repeated actor")
        index[observation.actor] = observation
        if observation.status == "error":
            if observation.error_type is None:
                raise ValueError("cleanup error requires an exception type")
        elif observation.error_type is not None:
            raise ValueError("successful or absent cleanup cannot contain an error")
        if observation.status in ("verified_empty", "not_empty"):
            if observation.deleted_count is None or not cleanup_enabled or not started:
                raise ValueError("cleanup success requires an enabled, started backend")
        elif observation.deleted_count is not None:
            raise ValueError("absent or failed cleanup cannot claim a deletion count")
        if observation.status == "disabled" and cleanup_enabled:
            raise ValueError("enabled cleanup cannot be marked disabled")
        if observation.status == "not_started" and started:
            raise ValueError("started backend cannot claim cleanup was unnecessary")
    checks = []
    for actor in case.actors:
        observation = index.get(actor)
        if not cleanup_enabled:
            verdict, reason = Verdict.NOT_EVALUATED, "cleanup_explicitly_disabled"
        elif not started:
            verdict, reason = Verdict.NOT_EVALUATED, "backend_not_started"
        elif observation is None:
            verdict, reason = Verdict.INCONCLUSIVE, "cleanup_evidence_missing"
        elif observation.status == "verified_empty":
            verdict, reason = Verdict.PASS, "generated_scope_verified_empty"
        else:
            verdict, reason = Verdict.FAIL, "cleanup_not_verified"
        checks.append(MemoryAssertion(name=actor, verdict=verdict, reason=reason))
    return tuple(checks)


def _write_checks(
    step: ExpectedRemember,
    observation: StepObservation,
    aliases: dict[str, CapturedVersion],
    objects: dict[str, str],
    versions: dict[str, CapturedVersion],
    deleted: set[str],
    current: dict[str, str],
) -> tuple[MemoryAssertion, ...]:
    write = observation.remember
    if write is None:
        raise ValueError("remember observation is missing its payload")
    version = write.version
    if version.actor != step.actor:
        raise ValueError("captured alias belongs to an impossible actor")
    if version.object_token in objects and objects[version.object_token] != step.actor:
        raise ValueError("object token is shared across actors")
    if version.version_token in versions and versions[version.version_token] != version:
        raise ValueError("version token is reused across objects or actors")
    live = version.object_token in objects and version.object_token not in deleted
    known_version = current.get(version.object_token) == version.version_token
    consistent = {
        RememberOutcome.CREATED: not live,
        RememberOutcome.UPDATED: live,
        RememberOutcome.UNCHANGED: live and known_version,
    }[write.outcome]
    if consistent != write.identity_consistent:
        raise ValueError("recorded write identity disagrees with prior evidence")
    checks = [
        assertion(
            "write_integrity",
            write.write_matches,
            success="requested_content_preserved",
            failure="returned_content_mismatch",
        ),
        assertion(
            "write_identity", consistent, success="outcome_matches_identity", failure="outcome_identity_mismatch"
        ),
    ]
    if step.expected_outcome is not None:
        checks.append(
            assertion(
                "expected_outcome",
                write.outcome == step.expected_outcome,
                success="expected_outcome_observed",
                failure="unexpected_write_outcome",
            )
        )
    if step.expected_same_as is not None:
        earlier = aliases[step.expected_same_as]
        checks.append(
            assertion(
                "expected_same_object",
                version.object_token == earlier.object_token,
                success="object_identity_preserved",
                failure="object_identity_changed",
            )
        )
    objects[version.object_token] = step.actor
    versions[version.version_token] = version
    aliases[step.alias] = version
    deleted.discard(version.object_token)
    current[version.object_token] = version.version_token
    return tuple(checks)


def _trial_result(
    case: ExpectedCase,
    repetition: int,
    observation: TrialObservation | None,
    evidence: MemoryEvidence,
) -> MemoryTrialEvaluation:
    operations = observation.operations if observation else ()
    expected_ids = {step.id for step in case.steps}
    order = {step.id: index for index, step in enumerate(case.steps)}
    indexed = {}
    last_index = -1
    for operation in operations:
        if operation.step_id not in expected_ids or operation.step_id in indexed:
            raise ValueError("operation references an unknown or repeated step")
        if order[operation.step_id] <= last_index:
            raise ValueError("operations do not follow declared lifecycle order")
        last_index = order[operation.step_id]
        indexed[operation.step_id] = operation
    aliases: dict[str, CapturedVersion] = {}
    objects: dict[str, str] = {}
    versions: dict[str, CapturedVersion] = {}
    deleted: set[str] = set()
    current: dict[str, str] = {}
    interrupted = observation is None or observation.factory_error_type is not None
    results = []
    calls = 0
    for step in case.steps:
        operation = indexed.get(step.id) or _unavailable(step, missing_trial=observation is None)
        if operation.kind != step.kind:
            raise ValueError("operation kind differs from declared step")
        if interrupted and operation.status != OperationStatus.UNAVAILABLE:
            raise ValueError("executed operation appears after a missing or failed operation")
        query = None
        if operation.status != OperationStatus.COMPLETED:
            interrupted = True
            is_contract = operation.error_category == "contract"
            checks = (
                MemoryAssertion(
                    name="operation_available",
                    verdict=Verdict.FAIL if is_contract else Verdict.INCONCLUSIVE,
                    reason="invalid_backend_output" if is_contract else "operation_unavailable",
                ),
            )
            if operation.status == OperationStatus.ERROR:
                calls += 1
            if isinstance(step, ExpectedRecall):
                query = unavailable_recall(
                    step, evidence.specification.cutoffs, status=operation.status, reason="operation_unavailable"
                )
        else:
            calls += 1
            if isinstance(step, ExpectedRemember):
                checks = _write_checks(step, operation, aliases, objects, versions, deleted, current)
            elif isinstance(step, ExpectedForget):
                version = aliases[step.alias]
                if operation.forget == ForgetStatus.DELETED:
                    deleted.add(version.object_token)
                checks = (
                    assertion(
                        "expected_forget_status",
                        operation.forget == step.expected_status,
                        success="expected_delete_outcome",
                        failure="unexpected_delete_outcome",
                    ),
                )
                if version.actor != step.actor:
                    checks += (
                        assertion(
                            "delete_scope_isolation",
                            operation.forget != ForgetStatus.DELETED,
                            success="foreign_object_not_deleted",
                            failure="foreign_object_deleted",
                        ),
                    )
            else:
                if operation.recall is None:
                    raise ValueError("recall observation is missing its payload")
                seen = set()
                forbidden_versions = {aliases[alias].version_token for alias in step.forbidden}
                for hit in operation.recall.hits:
                    if hit.scope_valid and hit.forbidden_match != (hit.version_token in forbidden_versions):
                        raise ValueError("forbidden match disagrees with captured version")
                    if hit.forbidden_match and not step.forbidden:
                        raise ValueError("forbidden match requires a declared forbidden alias")
                    if hit.object_token is not None:
                        if objects.get(hit.object_token) != step.actor:
                            raise ValueError("recall references an unknown or foreign object token")
                        if hit.deleted != (hit.object_token in deleted):
                            raise ValueError("recall deletion flag disagrees with lifecycle evidence")
                        if hit.duplicate != (hit.object_token in seen):
                            raise ValueError("recall duplicate flag disagrees with ranked evidence")
                        seen.add(hit.object_token)
                try:
                    query = score_recall(step, operation.recall, aliases, evidence.specification.cutoffs)
                except ValueError:
                    # Conflicting runtime judgments invalidate the scenario. Do not
                    # reinterpret them as backend quality failures or drop the slot.
                    raise
                checks = query.assertions
        results.append(
            MemoryOperationEvaluation(
                step_id=step.id,
                kind=step.kind,
                status=operation.status,
                assertions=checks,
                query=query,
                verdict=combine_verdicts([check.verdict for check in checks]),
            )
        )
    if observation is not None and observation.backend_calls != calls:
        raise ValueError("backend call count disagrees with executed operation ledger")
    started = observation is not None and observation.factory_error_type is None
    cleanup = _cleanup_results(
        case, observation.cleanup if observation else (), cleanup_enabled=evidence.cleanup_enabled, started=started
    )
    if observation is not None:
        minimum_calls = sum(item.status in ("verified_empty", "not_empty") for item in observation.cleanup) * 2
        maximum_calls = len(case.actors) * 2 if started and evidence.cleanup_enabled else 0
        if not minimum_calls <= observation.cleanup_calls <= maximum_calls:
            raise ValueError("cleanup call count is inconsistent with cleanup evidence")
    return MemoryTrialEvaluation(
        case_id=case.id,
        repetition=repetition,
        operations=tuple(results),
        cleanup=cleanup,
        expected_operations=len(case.steps),
        completed_operations=sum(item.status == OperationStatus.COMPLETED for item in results),
        verdict=combine_verdicts([item.verdict for item in results] + [item.verdict for item in cleanup]),
    )


def replay_memory(
    evidence: MemoryEvidence,
    *,
    limits: MemoryEvaluationLimits | None = None,
) -> MemoryEvaluationReport:
    """Recompute scores offline without accepting stored summaries as authority.

    Selected observations are attestations, not cryptographic proof of backend
    behavior. Replay checks consistency and completeness, never invokes code,
    and never recovers the text intentionally excluded during capture.
    """
    limits = MemoryEvaluationLimits.model_validate((limits or MemoryEvaluationLimits()).model_dump())
    evidence = MemoryEvidence.model_validate(evidence.model_dump())
    suite = evidence.specification.as_validation_suite()
    suite.check_limits(limits)
    if len(evidence.model_dump_json().encode("utf-8")) > limits.max_bytes:
        raise ValueError("memory evidence byte limit exceeded")
    case_ids = {case.id for case in evidence.specification.cases}
    indexed = {}
    for trial in evidence.trials:
        key = (trial.case_id, trial.repetition)
        if trial.case_id not in case_ids or trial.repetition >= suite.repetitions or key in indexed:
            raise ValueError("unknown, out-of-range or repeated trial slot")
        indexed[key] = trial
    trials = tuple(
        _trial_result(case, repetition, indexed.get((case.id, repetition)), evidence)
        for case in evidence.specification.cases
        for repetition in range(suite.repetitions)
    )
    aggregates = aggregate_memory(evidence.specification, trials)
    gates = evaluate_gates(evidence.specification, aggregates)
    return MemoryEvaluationReport(
        evidence=evidence,
        trials=trials,
        aggregates=aggregates,
        gates=gates,
        expected_operations=sum(trial.expected_operations for trial in trials),
        completed_operations=sum(trial.completed_operations for trial in trials),
        verdict=combine_verdicts([trial.verdict for trial in trials] + [gate.verdict for gate in gates]),
    )
