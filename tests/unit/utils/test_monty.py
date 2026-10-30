import os
import shutil
import subprocess

import pytest

pytest.importorskip("pydantic_monty")

from dynamiq.utils import monty  # noqa: E402

needs_nice = pytest.mark.skipif(os.name != "posix" or shutil.which("nice") is None, reason="needs the nice command")


def worker_pids() -> list[int]:
    """Return the PIDs of the Monty workers this process has started.

    Reads /proc where there is one, since slim container images have no ps, and asks ps otherwise (macOS).
    """
    if os.path.isdir("/proc"):
        pids = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as stat_file:
                    stat = stat_file.read()
            except OSError:  # the process exited
                continue
            # The command name is in parentheses and may hold spaces; the parent PID follows the state after it.
            name = stat[stat.index("(") + 1 : stat.rindex(")")]
            parent = int(stat[stat.rindex(")") + 2 :].split()[1])
            if parent == os.getpid() and name == "monty":
                pids.append(int(entry))
        return pids
    if shutil.which("ps") is None:
        pytest.skip("needs /proc or ps to find the worker processes")
    rows = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,comm="], capture_output=True, text=True, check=True).stdout
    return [
        int(fields[0])
        for fields in (row.split() for row in rows.splitlines())
        if fields[1] == str(os.getpid()) and fields[-1].endswith("monty")
    ]


def worker_niceness() -> list[int]:
    """Return the nice value of every Monty worker this process has started."""
    return [os.getpriority(os.PRIO_PROCESS, pid) for pid in worker_pids()]


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
