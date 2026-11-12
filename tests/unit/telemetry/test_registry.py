"""The process-wide registry of the spans of runs in progress."""

import os
import subprocess
import sys
import textwrap
import threading
import uuid
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.context import Context

import dynamiq
from dynamiq.telemetry import registry
from dynamiq.telemetry.registry import RunEntry
from dynamiq.telemetry.types import CaptureSettings, RunKind, SpanType

REPO_ROOT = Path(dynamiq.__file__).parent.parent


def entry(run_id: str, owner: bool = True) -> RunEntry:
    capture = CaptureSettings()
    return RunEntry(
        run_id=run_id,
        span=trace.INVALID_SPAN,
        context=Context(),
        tracer=trace.NoOpTracer(),
        settings=capture,
        root_settings=capture,
        span_type=SpanType.NODE,
        run_kind=RunKind.NODE,
        recording=False,
        owner=owner,
    )


@pytest.fixture(autouse=True)
def empty_registry():
    assert registry.size() == 0
    yield
    assert registry.size() == 0


def test_entry_is_found_by_its_run_id_until_it_is_popped():
    registered = entry("run-1")

    assert registry.get("run-1") is None
    registry.put(registered)

    assert registry.get("run-1") is registered
    assert registry.size() == 1
    assert registry.pop("run-1") is registered
    assert registry.get("run-1") is None
    assert registry.pop("run-1") is None


def test_run_id_is_matched_by_its_string_form():
    run_id = uuid.uuid4()
    registered = entry(str(run_id))
    registry.put(registered)

    assert registry.get(run_id) is registered
    assert registry.pop(run_id) is registered


def test_no_run_id_matches_nothing():
    registry.put(entry("None"))

    assert registry.get(None) is None
    assert registry.pop(None) is None
    assert registry.pop("None") is not None


def test_aliases_are_removed_with_the_entry_they_stand_for():
    owner = entry("workflow-run")
    alias = entry("flow-run", owner=False)
    owner.aliases.append("flow-run")
    registry.put(owner)
    registry.put(alias)

    assert registry.pop("workflow-run") is owner

    assert registry.get("flow-run") is None


def test_alias_removed_first_leaves_its_owner_registered():
    owner = entry("workflow-run")
    owner.aliases.append("flow-run")
    registry.put(owner)
    registry.put(entry("flow-run", owner=False))

    registry.pop("flow-run")

    assert registry.get("workflow-run") is owner
    assert registry.pop("workflow-run") is owner


def test_span_end_is_claimed_once():
    registered = entry("run-1")

    assert registry.claim_end(registered) is True
    assert registered.ended is True
    assert registry.claim_end(registered) is False


def test_span_end_is_claimed_once_across_threads():
    registered = entry("run-1")
    barrier = threading.Barrier(16)
    claims = []

    def claim():
        barrier.wait()
        claims.append(registry.claim_end(registered))

    threads = [threading.Thread(target=claim) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert claims.count(True) == 1


def test_concurrent_runs_register_and_leave_without_losing_entries():
    barrier = threading.Barrier(8)
    missing = []

    def run(worker: int):
        barrier.wait()
        for index in range(500):
            run_id = f"{worker}-{index}"
            registered = entry(run_id)
            registry.put(registered)
            if registry.get(run_id) is not registered or registry.pop(run_id) is not registered:
                missing.append(run_id)

    threads = [threading.Thread(target=run, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert missing == []


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_starts_with_an_empty_registry():
    # In a process of its own: forking the test process, with its threads and loaded SDKs, is not safe.
    script = textwrap.dedent(
        """
        import os
        import signal
        import sys

        from opentelemetry import trace
        from opentelemetry.context import Context

        from dynamiq.telemetry import registry
        from dynamiq.telemetry.types import CaptureSettings, RunKind, SpanType

        def entry(run_id):
            capture = CaptureSettings()
            return registry.RunEntry(
                run_id=run_id, span=trace.INVALID_SPAN, context=Context(), tracer=trace.NoOpTracer(),
                settings=capture, root_settings=capture, span_type=SpanType.NODE, run_kind=RunKind.NODE,
                recording=False,
            )

        registry.put(entry("parent-run"))
        # Held across the fork, as a lock is when another thread of the parent is registering a run.
        registry._lock.acquire()
        pid = os.fork()
        if pid == 0:
            inherited = registry.get("parent-run") is not None or registry.size() != 0
            # Registering would wait forever on a lock inherited from the parent; the alarm ends the child then.
            signal.alarm(10)
            registry.put(entry("child-run"))
            os._exit(1 if inherited or registry.size() != 1 else 0)
        registry._lock.release()
        _, status = os.waitpid(pid, 0)
        sys.exit(0 if os.waitstatus_to_exitcode(status) == 0 and registry.size() == 1 else 1)
        """
    )

    completed = subprocess.run(  # nosec B603
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT, timeout=120
    )

    assert completed.returncode == 0, completed.stderr
