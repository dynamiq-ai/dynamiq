import os
import shutil
import subprocess

import pytest

pytest.importorskip("pydantic_monty")

from dynamiq.utils import monty  # noqa: E402

needs_nice = pytest.mark.skipif(os.name != "posix" or shutil.which("nice") is None, reason="needs the nice command")


def worker_niceness() -> list[int]:
    """Return the nice value of every Monty worker this process has started."""
    rows = subprocess.run(["ps", "-A", "-o", "ppid=,ni=,comm="], capture_output=True, text=True, check=True).stdout
    return [
        int(fields[1])
        for fields in (row.split() for row in rows.splitlines())
        if fields[0] == str(os.getpid()) and fields[-1].endswith("monty")
    ]


@needs_nice
def test_pool_workers_run_at_low_cpu_priority():
    with monty.get_monty_pool().checkout() as session:
        assert session.feed_run("1 + 1") == 2
        niceness = worker_niceness()
    assert niceness
    assert set(niceness) == {monty.MONTY_WORKER_NICENESS}


@needs_nice
def test_pool_runs_workers_at_normal_priority_when_the_launcher_cannot_start(monkeypatch):
    # Leaving the launcher without its execute bit stands in for a temporary directory mounted noexec.
    monkeypatch.setattr(monty.os, "chmod", lambda *args, **kwargs: None)
    pool, launcher_dir = monty._start_pool()
    try:
        assert launcher_dir is None
        with pool.checkout() as session:
            assert session.feed_run("1 + 1") == 2
            assert 0 in worker_niceness()
    finally:
        pool.__exit__(None, None, None)


def test_pool_runs_workers_at_normal_priority_without_nice(monkeypatch):
    monkeypatch.setattr(monty.shutil, "which", lambda name: None)
    pool, launcher_dir = monty._start_pool()
    try:
        assert launcher_dir is None
        with pool.checkout() as session:
            assert session.feed_run("1 + 1") == 2
    finally:
        pool.__exit__(None, None, None)
