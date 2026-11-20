"""Sequential native memory evaluation with isolated scopes and bounded cleanup."""

import re
import sys
import time
from collections.abc import Callable
from itertools import islice
from typing import Any
from uuid import uuid4

from dynamiq.evaluations.memory.evidence import (
    CleanupObservation,
    MemoryEvidence,
    MemorySpecification,
    StepObservation,
    TrialObservation,
)
from dynamiq.evaluations.memory.identity import MemoryBackend, MemoryBackendContractError, MemoryIdentityRegistry
from dynamiq.evaluations.memory.replay import replay_memory
from dynamiq.evaluations.memory.results import MemoryEvaluationReport
from dynamiq.evaluations.memory.suite import (
    Forget,
    MemoryCase,
    MemoryEvaluationLimits,
    MemorySuite,
    OperationStatus,
    Recall,
    Remember,
)


def _error_type(error: BaseException) -> str:
    """Never retain messages, tracebacks or nonportable custom exception names."""
    name = type(error).__name__
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,127}", name) else "BackendError"


def _execute(
    backend: MemoryBackend,
    registry: MemoryIdentityRegistry,
    step: Remember | Recall | Forget,
) -> StepObservation:
    started = time.perf_counter()
    payload: dict[str, Any] = {}
    try:
        user_id = registry.user_id(step.actor)
        if isinstance(step, Remember):
            result = backend.remember(content=step.content, user_id=user_id)
            payload["remember"] = registry.capture_remember(step, result)
        elif isinstance(step, Recall):
            result = backend.recall(query=step.query, user_id=user_id, limit=step.limit)
            payload["recall"] = registry.capture_recall(step.actor, result, limit=step.limit, forbidden=step.forbidden)
        else:
            result = backend.forget(fact_id=registry.backend_id(step.alias), user_id=user_id)
            payload["forget"] = registry.capture_forget(step.alias, result)
    except Exception as exc:
        return StepObservation(
            step_id=step.id,
            kind=step.kind,
            status=OperationStatus.ERROR,
            duration_seconds=max(0.0, time.perf_counter() - started),
            error_type=_error_type(exc),
            error_category="contract" if isinstance(exc, MemoryBackendContractError) else "backend",
        )
    return StepObservation(
        step_id=step.id,
        kind=step.kind,
        status=OperationStatus.COMPLETED,
        duration_seconds=max(0.0, time.perf_counter() - started),
        **payload,
    )


def _cleanup(
    backend: MemoryBackend,
    scopes: dict[str, str],
) -> tuple[tuple[CleanupObservation, ...], int, BaseException | None]:
    """Attempt every generated scope even when another cleanup is interrupted.

    BaseException is returned to the run boundary for propagation after all
    scopes have been attempted. A cleanup failure must not replace an original
    cancellation from an operation. Caller-owned clients are never closed.
    """
    observations = []
    calls = 0
    interruption = None
    for actor, scope in scopes.items():
        try:
            calls += 1
            count = backend.clear_user(user_id=scope)
            if type(count) is not int or count < 0:
                raise MemoryBackendContractError("clear_user must return a nonnegative integer")
            calls += 1
            remaining = backend.list_all(user_id=scope, limit=1)
            if isinstance(remaining, (str, bytes, dict)):
                raise MemoryBackendContractError("list_all must return a sequence")
            rows = list(islice(iter(remaining), 1))
            observations.append(
                CleanupObservation(
                    actor=actor,
                    status="not_empty" if rows else "verified_empty",
                    deleted_count=count,
                )
            )
        except BaseException as exc:
            observations.append(CleanupObservation(actor=actor, status="error", error_type=_error_type(exc)))
            if not isinstance(exc, Exception) and interruption is None:
                interruption = exc
    return tuple(observations), calls, interruption


def _run_trial(
    case: MemoryCase,
    repetition: int,
    factory: Callable[[], MemoryBackend],
    *,
    cleanup_enabled: bool,
) -> TrialObservation:
    # UUID values are independent of user-supplied labels and never exported.
    scopes = {actor: f"dynamiq-eval-{uuid4().hex}" for actor in case.actors}
    registry = MemoryIdentityRegistry(scopes)
    try:
        backend = factory()
        if backend is None:
            raise MemoryBackendContractError("backend factory returned None")
    except Exception as exc:
        return TrialObservation(
            case_id=case.id,
            repetition=repetition,
            factory_error_type=_error_type(exc),
            operations=tuple(
                StepObservation(
                    step_id=step.id,
                    kind=step.kind,
                    status=OperationStatus.UNAVAILABLE,
                    unavailable_reason="factory_error",
                )
                for step in case.steps
            ),
            cleanup=tuple(CleanupObservation(actor=actor, status="not_started") for actor in case.actors),
        )
    operations = []
    calls = 0
    interrupted = False
    cleanup_calls = 0
    cleanup = ()
    try:
        for step in case.steps:
            if interrupted:
                operations.append(
                    StepObservation(
                        step_id=step.id,
                        kind=step.kind,
                        status=OperationStatus.UNAVAILABLE,
                        unavailable_reason="prior_operation_error",
                    )
                )
                continue
            calls += 1
            observation = _execute(backend, registry, step)
            operations.append(observation)
            interrupted = observation.status != OperationStatus.COMPLETED
    finally:
        if cleanup_enabled:
            primary_interruption = sys.exc_info()[0] is not None
            cleanup, cleanup_calls, cleanup_interruption = _cleanup(backend, scopes)
            if cleanup_interruption is not None and not primary_interruption:
                raise cleanup_interruption
        else:
            cleanup = tuple(CleanupObservation(actor=actor, status="disabled") for actor in case.actors)
    return TrialObservation(
        case_id=case.id,
        repetition=repetition,
        operations=tuple(operations),
        cleanup=cleanup,
        backend_calls=calls,
        cleanup_calls=cleanup_calls,
    )


def evaluate_memory(
    suite: MemorySuite,
    backend_factory: Callable[[], MemoryBackend],
    *,
    limits: MemoryEvaluationLimits | None = None,
    cleanup_enabled: bool = True,
) -> MemoryEvaluationReport:
    """Run a declared lifecycle against fresh generated test-only actor scopes.

    The factory is called once per case/repetition. It may deliberately return
    a shared backend; scopes remain unique. Its embedder/storage may perform
    network calls, so live execution requires the application's explicit
    factory. Parsing and replay never import a provider from input data.

    No operation is retried. Provider timeouts belong in the supplied backend;
    synchronous Python cannot safely cancel an in-flight mutation. Cleanup is
    enabled by default and verifies emptiness after clearing each test scope.
    """
    if type(cleanup_enabled) is not bool:
        raise ValueError("cleanup_enabled must be a boolean")
    if not callable(backend_factory):
        raise TypeError("backend_factory must be callable")
    suite = MemorySuite.model_validate(suite.model_dump())
    limits = MemoryEvaluationLimits.model_validate((limits or MemoryEvaluationLimits()).model_dump())
    suite.check_limits(limits)
    specification = MemorySpecification.from_suite(suite)
    trials = tuple(
        _run_trial(case, repetition, backend_factory, cleanup_enabled=cleanup_enabled)
        for case in suite.cases
        for repetition in range(suite.repetitions)
    )
    evidence = MemoryEvidence(
        run_id=uuid4().hex,
        specification=specification,
        cleanup_enabled=cleanup_enabled,
        trials=trials,
    )
    return replay_memory(evidence, limits=limits)
