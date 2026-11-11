"""Trial-local native identities, immutable version aliases and safe observations."""

import math
from itertools import islice
from numbers import Real
from typing import Any, Protocol

from dynamiq.evaluations.memory.evidence import CapturedVersion, RecallHit, RecallObservation, RememberObservation
from dynamiq.evaluations.memory.suite import MemoryModel, Name, Remember, Text
from dynamiq.memory.long_term.schemas import Fact
from dynamiq.memory.long_term.types import ForgetStatus, RememberOutcome


class MemoryBackend(Protocol):
    """The existing long-term backend surface needed by the evaluator."""

    def remember(
        self, *, content: str, user_id: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Fact, RememberOutcome]:
        """Write or update one fact inside the supplied generated scope."""
        ...

    def recall(self, *, query: str, user_id: str, limit: int = 5) -> list[tuple[Fact, float]]:
        """Return ranked facts without crossing the supplied scope."""
        ...

    def forget(self, *, fact_id: str, user_id: str) -> ForgetStatus:
        """Delete a current object only when the supplied actor owns it."""
        ...

    def clear_user(self, *, user_id: str) -> int:
        """Clear only the generated test actor's scope."""
        ...

    def list_all(self, *, user_id: str, limit: int = 100) -> list[Fact]:
        """Support a bounded post-cleanup emptiness check."""
        ...


class MemoryBackendContractError(ValueError):
    """A provider returned invalid evidence; messages never contain its payload."""


class _StoredAlias(MemoryModel):
    """Private primitives copied from a Fact; this model is never exported."""

    actor: Name
    backend_id: Name
    content: Text
    object_token: Name
    version_token: Name


def _fact_fields(value: Any) -> tuple[str, str, str]:
    """Validate selected fields without copying metadata or trusting model_copy."""
    if not isinstance(value, Fact):
        raise MemoryBackendContractError("expected a native Fact")
    try:
        fields = (value.id, value.user_id, value.content)
    except AttributeError:
        # model_construct can omit required fields entirely, just as model_copy
        # can bypass their validators. This is invalid output, not a transient
        # backend exception that should merely reduce availability.
        raise MemoryBackendContractError("native Fact is missing required identity or content") from None
    for index, field in enumerate(fields):
        maximum = 1000000 if index == 2 else 256
        if type(field) is not str or not field or len(field) > maximum:
            raise MemoryBackendContractError("invalid native Fact identity or content")
        try:
            field.encode("utf-8")
        except UnicodeError as exc:
            raise MemoryBackendContractError("invalid native Fact Unicode") from exc
    if not fields[2].strip():
        raise MemoryBackendContractError("native Fact content is blank")
    return fields


def _similarity(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MemoryBackendContractError("similarity must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise MemoryBackendContractError("invalid similarity") from exc
    if not math.isfinite(result):
        raise MemoryBackendContractError("similarity must be finite")
    return result


class MemoryIdentityRegistry:
    """Preserve object and version identity without emitting native identifiers.

    The owner scope participates in every lookup. An immutable alias continues
    to refer to its captured content even after a semantic upsert reuses the ID.
    Forget, however, addresses the object and tombstones all of its versions.
    """

    def __init__(self, scopes: dict[str, str]) -> None:
        self._scopes = dict(scopes)
        self._objects: dict[tuple[str, str], str] = {}
        self._versions: dict[tuple[str, str, str], str] = {}
        self._aliases: dict[str, _StoredAlias] = {}
        self._deleted: set[tuple[str, str]] = set()
        self._current: dict[tuple[str, str], str] = {}

    def user_id(self, actor: str) -> str:
        return self._scopes[actor]

    def backend_id(self, alias: str) -> str:
        """Resolve only a previously successful write; never caller-provided IDs."""
        if alias not in self._aliases:
            raise MemoryBackendContractError("required memory alias is unavailable")
        return self._aliases[alias].backend_id

    def version(self, alias: str) -> CapturedVersion:
        value = self._aliases[alias]
        return CapturedVersion(actor=value.actor, object_token=value.object_token, version_token=value.version_token)

    def capture_remember(self, step: Remember, result: Any) -> RememberObservation:
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise MemoryBackendContractError("remember must return a fact/outcome pair")
        backend_id, user_id, content = _fact_fields(result[0])
        if user_id != self.user_id(step.actor):
            raise MemoryBackendContractError("remember returned a foreign actor scope")
        try:
            outcome = RememberOutcome(result[1])
        except (TypeError, ValueError) as exc:
            raise MemoryBackendContractError("invalid remember outcome") from exc
        if step.alias in self._aliases:
            raise MemoryBackendContractError("memory alias is already captured")
        object_key = (user_id, backend_id)
        version_key = (user_id, backend_id, content)
        known_object = object_key in self._objects and object_key not in self._deleted
        known_version = self._current.get(object_key) == content
        if outcome == RememberOutcome.CREATED:
            identity_consistent = not known_object
        elif outcome == RememberOutcome.UPDATED:
            identity_consistent = known_object
        else:
            identity_consistent = known_object and known_version
        if object_key not in self._objects:
            self._objects[object_key] = f"object-{len(self._objects) + 1}"
        if version_key not in self._versions:
            self._versions[version_key] = f"version-{len(self._versions) + 1}"
        self._deleted.discard(object_key)
        snapshot = _StoredAlias(
            actor=step.actor,
            backend_id=backend_id,
            content=content,
            object_token=self._objects[object_key],
            version_token=self._versions[version_key],
        )
        self._aliases[step.alias] = snapshot
        self._current[object_key] = content
        requested = step.content.strip()
        returned = content.strip()
        if step.write_match == "native":
            requested, returned = requested.lower(), returned.lower()
        return RememberObservation(
            version=self.version(step.alias),
            outcome=outcome,
            write_matches=requested == returned,
            identity_consistent=identity_consistent,
        )

    def capture_recall(
        self,
        actor: str,
        result: Any,
        *,
        limit: int,
        forbidden: tuple[str, ...] = (),
    ) -> RecallObservation:
        """Keep invalid rows in their original positions and bound iterables."""
        if isinstance(result, (str, bytes, dict)):
            raise MemoryBackendContractError("recall requires a sequence of fact/score pairs")
        try:
            rows = list(islice(iter(result), limit + 1))
        except (TypeError, ValueError) as exc:
            raise MemoryBackendContractError("recall result is not a valid sequence") from exc
        hits = []
        seen: set[tuple[str, str]] = set()
        expected_scope = self.user_id(actor)
        forbidden_keys = {
            (self.user_id(self._aliases[alias].actor), self._aliases[alias].backend_id, self._aliases[alias].content)
            for alias in forbidden
        }
        for rank, row in enumerate(rows[:limit], start=1):
            try:
                if not isinstance(row, (tuple, list)) or len(row) != 2:
                    raise MemoryBackendContractError("invalid ranked row")
                backend_id, user_id, content = _fact_fields(row[0])
                score = _similarity(row[1])
            except MemoryBackendContractError:
                hits.append(RecallHit(rank=rank, output_valid=False))
                continue
            object_key = (user_id, backend_id)
            duplicate = object_key in seen
            seen.add(object_key)
            forbidden_match = (user_id, backend_id, content) in forbidden_keys
            if user_id != expected_scope:
                hits.append(
                    RecallHit(
                        rank=rank, score=score, scope_valid=False, duplicate=duplicate, forbidden_match=forbidden_match
                    )
                )
                continue
            hits.append(
                RecallHit(
                    rank=rank,
                    score=score,
                    object_token=self._objects.get(object_key),
                    version_token=self._versions.get((user_id, backend_id, content)),
                    duplicate=duplicate,
                    deleted=object_key in self._deleted,
                    forbidden_match=forbidden_match,
                )
            )
        return RecallObservation(hits=tuple(hits), excess_results=len(rows) > limit)

    def capture_forget(self, alias: str, result: Any) -> ForgetStatus:
        try:
            status = ForgetStatus(result)
        except (TypeError, ValueError) as exc:
            raise MemoryBackendContractError("invalid forget outcome") from exc
        if status == ForgetStatus.DELETED:
            self.mark_deleted(alias)
        return status

    def mark_deleted(self, alias: str) -> None:
        value = self._aliases[alias]
        self._deleted.add((self.user_id(value.actor), value.backend_id))
