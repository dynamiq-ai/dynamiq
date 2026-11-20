"""Bounded portable JSON and atomic files for memory scenarios and evidence."""

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, ValidationError

from dynamiq.evaluations.memory.evidence import MemoryEvidence
from dynamiq.evaluations.memory.replay import replay_memory
from dynamiq.evaluations.memory.results import MemoryEvaluationReport
from dynamiq.evaluations.memory.suite import MemoryEvaluationLimits, MemoryModel, MemorySuite


class MemorySuiteArtifact(MemoryModel):
    """Explicit raw-text input export; unlike evidence, this contains private text."""

    format: Literal["dynamiq.memory.suite"] = "dynamiq.memory.suite"
    version: Annotated[StrictInt, Field(ge=1, le=1)] = 1
    suite: MemorySuite


def _limits(value: MemoryEvaluationLimits | None) -> MemoryEvaluationLimits:
    return MemoryEvaluationLimits.model_validate((value or MemoryEvaluationLimits()).model_dump())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("JSON constants must be finite")


def _check_text_depth(text: str, maximum: int) -> None:
    """Bound nesting before the standard decoder can recurse on untrusted input."""
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > maximum:
                raise ValueError("memory JSON depth limit exceeded")
        elif character in "]}":
            depth -= 1
    # The decoder remains responsible for syntax and matching delimiters.


def _check_tree(value: Any, limits: MemoryEvaluationLimits) -> None:
    stack = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > limits.max_json_nodes:
            raise ValueError("memory JSON node limit exceeded")
        if depth > limits.max_json_depth:
            raise ValueError("memory JSON depth limit exceeded")
        if isinstance(item, dict):
            stack.extend((key, depth + 1) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError("memory JSON contains nonportable Unicode") from exc
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("memory JSON numbers must be finite")


def _parse(raw: str | bytes, limits: MemoryEvaluationLimits) -> dict[str, Any]:
    if not isinstance(raw, (str, bytes)):
        raise TypeError("memory JSON input must be text or bytes")
    try:
        if isinstance(raw, bytes):
            if len(raw) > limits.max_bytes:
                raise ValueError("memory JSON byte limit exceeded")
            text = raw.decode("utf-8")
        else:
            if len(raw) > limits.max_bytes or len(raw.encode("utf-8")) > limits.max_bytes:
                raise ValueError("memory JSON byte limit exceeded")
            text = raw
    except UnicodeError as exc:
        raise ValueError("memory JSON must be portable UTF-8") from exc
    _check_text_depth(text, limits.max_json_depth)
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid memory JSON document") from exc
    _check_tree(value, limits)
    if not isinstance(value, dict):
        raise ValueError("memory JSON must contain one object")
    return value


def _encode(value: MemoryModel, limits: MemoryEvaluationLimits) -> str:
    data = value.model_dump(mode="json")
    _check_tree(data, limits)
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if len(text.encode("utf-8")) > limits.max_bytes:
        raise ValueError("memory JSON byte limit exceeded")
    return text


def load_suite(raw: str | bytes, *, limits: MemoryEvaluationLimits | None = None) -> MemorySuite:
    """Load only a versioned scenario, without instantiating a backend/provider."""
    trusted = _limits(limits)
    data = _parse(raw, trusted)
    try:
        artifact = MemorySuiteArtifact.model_validate(data)
        artifact.suite.check_limits(trusted)
    except ValidationError:
        # Pydantic's default message includes the rejected value. Scenario text
        # may contain private data, so callers receive a deliberately terse error.
        raise ValueError("invalid memory suite schema") from None
    return artifact.suite


def dump_suite(suite: MemorySuite, *, limits: MemoryEvaluationLimits | None = None) -> str:
    """Explicitly export a scenario INCLUDING its memory and query text."""
    trusted = _limits(limits)
    suite = MemorySuite.model_validate(suite.model_dump())
    suite.check_limits(trusted)
    return _encode(MemorySuiteArtifact(suite=suite), trusted)


def load_evidence(raw: str | bytes, *, limits: MemoryEvaluationLimits | None = None) -> MemoryEvidence:
    """Load and check a complete or incomplete ledger; never trust stored scores."""
    trusted = _limits(limits)
    data = _parse(raw, trusted)
    try:
        evidence = MemoryEvidence.model_validate(data)
    except ValidationError:
        raise ValueError("invalid memory evidence schema") from None
    return replay_memory(evidence, limits=trusted).evidence


def dump_evidence(evidence: MemoryEvidence, *, limits: MemoryEvaluationLimits | None = None) -> str:
    """Export selected observations only, preserving every replay ledger slot."""
    trusted = _limits(limits)
    checked = replay_memory(evidence, limits=trusted).evidence
    return _encode(checked, trusted)


def dump_report(report: MemoryEvaluationReport, *, limits: MemoryEvaluationLimits | None = None) -> str:
    """Export recomputed summaries together with the full underlying evidence."""
    trusted = _limits(limits)
    recomputed = replay_memory(report.evidence, limits=trusted)
    return _encode(recomputed, trusted)


def load_report(raw: str | bytes, *, limits: MemoryEvaluationLimits | None = None) -> MemoryEvaluationReport:
    """Read a saved report and recompute every derived value from its evidence.

    This accepts the versioned output of dump_report/render_memory_report,
    while load_evidence accepts the smaller capture-only artifact. Stored
    scores, support counts and verdicts never override the execution ledger.
    """
    trusted = _limits(limits)
    data = _parse(raw, trusted)
    try:
        stored = MemoryEvaluationReport.model_validate(data)
    except ValidationError:
        raise ValueError("invalid memory report schema") from None
    return replay_memory(stored.evidence, limits=trusted)


def _read(path: str | Path, limits: MemoryEvaluationLimits) -> bytes:
    with Path(path).open("rb") as stream:
        raw = stream.read(limits.max_bytes + 1)
    if len(raw) > limits.max_bytes:
        raise ValueError("memory JSON byte limit exceeded")
    return raw


def read_suite(path: str | Path, *, limits: MemoryEvaluationLimits | None = None) -> MemorySuite:
    """Read at most the trusted byte budget from a raw-text scenario file."""
    trusted = _limits(limits)
    return load_suite(_read(path, trusted), limits=trusted)


def read_evidence(path: str | Path, *, limits: MemoryEvaluationLimits | None = None) -> MemoryEvidence:
    """Read and validate bounded UTF-8 evidence without a provider connection."""
    trusted = _limits(limits)
    return load_evidence(_read(path, trusted), limits=trusted)


def read_report(path: str | Path, *, limits: MemoryEvaluationLimits | None = None) -> MemoryEvaluationReport:
    """Read a bounded versioned report file and verify its full evidence ledger."""
    trusted = _limits(limits)
    return load_report(_read(path, trusted), limits=trusted)


def write_text_atomic(path: str | Path, text: str, *, overwrite: bool = False) -> None:
    """Publish one complete UTF-8 file, with race-safe no-clobber by default.

    A sibling temporary file has private permissions. Hard-link publication
    makes the no-overwrite check atomic; unsupported filesystems raise instead
    of falling back to a check-then-replace race. Existing parents are required.
    """
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a boolean")
    target = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_suite(
    path: str | Path,
    suite: MemorySuite,
    *,
    overwrite: bool = False,
    limits: MemoryEvaluationLimits | None = None,
) -> None:
    """Explicit raw-text scenario export with atomic publication."""
    write_text_atomic(path, dump_suite(suite, limits=limits), overwrite=overwrite)


def write_evidence(
    path: str | Path,
    evidence: MemoryEvidence,
    *,
    overwrite: bool = False,
    limits: MemoryEvaluationLimits | None = None,
) -> None:
    """Atomically publish full replay evidence, not a presentation-only subset."""
    write_text_atomic(path, dump_evidence(evidence, limits=limits), overwrite=overwrite)
