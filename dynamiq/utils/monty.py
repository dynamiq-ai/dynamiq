"""A process-wide pool of Monty worker subprocesses, shared by PythonEvaluator and the PythonMonty node."""

import atexit
import os
import shlex
import shutil
import stat
import tempfile
import threading

import pydantic_monty

# The pool's own lookup of its worker binary (MONTY_BIN, then the environment's scripts directory), which the
# package does not export.
from pydantic_monty._binary import find_monty_binary

from dynamiq.utils.logger import logger

# A worker serves one session and is then replaced, so nothing a snippet leaves behind reaches the next
# evaluation or tenant. Starting a worker costs a few milliseconds.
MONTY_MAX_CHECKOUTS_PER_WORKER = 1
# The pool otherwise starts one worker per CPU, and in a container that is the host's CPU count.
MONTY_MAX_WORKERS = 4
MONTY_CHECKOUT_TIMEOUT_SECONDS = 60.0
# Workers run at the lowest CPU priority, so code that spins gives way to the service it shares a container with.
MONTY_WORKER_NICENESS = 19

_pool: pydantic_monty.Monty | None = None
_pool_pid: int | None = None
_pool_lock = threading.Lock()


def get_monty_pool() -> pydantic_monty.Monty:
    """Return the process-wide Monty worker pool, starting it on first use.

    The pool is safe to use from several threads at once. A checkout waits for a free worker for up to
    MONTY_CHECKOUT_TIMEOUT_SECONDS, then raises TimeoutError. Workers run at MONTY_WORKER_NICENESS where the nice
    command can start them, and at normal priority otherwise. A forked child starts a pool of its own rather than
    sharing the parent's workers.
    """
    global _pool, _pool_pid
    with _pool_lock:
        if _pool is None or _pool_pid != os.getpid():
            pool, launcher_dir = _start_pool()
            atexit.register(_close_pool, pool, os.getpid(), launcher_dir)
            _pool, _pool_pid = pool, os.getpid()
        return _pool


def _start_pool() -> tuple[pydantic_monty.Monty, str | None]:
    """Start a pool whose workers run under nice, or at normal priority when that fails.

    Returns the pool and the directory of its nice launcher, or None when the workers run the binary directly.
    """
    launcher_dir = None
    try:
        launcher_dir = _write_nice_launcher()
        if launcher_dir is not None:
            return _enter_pool(os.path.join(launcher_dir, "monty")), launcher_dir
    except Exception as error:
        # For example a temporary directory mounted noexec.
        logger.warning(f"Monty workers run at normal priority, because starting them under nice failed: {error}")
        if launcher_dir is not None:
            shutil.rmtree(launcher_dir, ignore_errors=True)
    return _enter_pool(None), None


def _write_nice_launcher() -> str | None:
    """Write a script that starts the worker binary under nice, and return its directory.

    Returns None on a system without the nice command.
    """
    nice = shutil.which("nice")
    if os.name != "posix" or nice is None:
        return None
    launcher_dir = tempfile.mkdtemp(prefix="dynamiq-monty-")
    launcher = os.path.join(launcher_dir, "monty")
    with open(launcher, "w") as file:
        # exec keeps the worker's PID, so the pool's kill of a worker past its time limit reaches the interpreter.
        file.write(
            f'#!/bin/sh\nexec {shlex.quote(nice)} -n {MONTY_WORKER_NICENESS} {shlex.quote(find_monty_binary())} "$@"\n'
        )
    os.chmod(launcher, stat.S_IRWXU)
    return launcher_dir


def _enter_pool(binary_path: str | None) -> pydantic_monty.Monty:
    """Start a pool running binary_path, or Monty's own binary when it is None, and check that a worker answers."""
    pool = pydantic_monty.Monty(
        binary_path=binary_path,
        min_processes=1,
        max_processes=MONTY_MAX_WORKERS,
        checkout_timeout=MONTY_CHECKOUT_TIMEOUT_SECONDS,
        max_checkouts_per_worker=MONTY_MAX_CHECKOUTS_PER_WORKER,
    )
    pool.__enter__()
    try:
        # A launcher that starts but cannot run the binary fails only once a session talks to its worker.
        with pool.checkout() as session:
            session.feed_run("None")
    except Exception:
        pool.__exit__(None, None, None)
        raise
    return pool


def _close_pool(pool: pydantic_monty.Monty, owner_pid: int, launcher_dir: str | None) -> None:
    # A forked child inherits this handler, but the workers and the launcher belong to the process that started them.
    if os.getpid() == owner_pid:
        pool.__exit__(None, None, None)
        if launcher_dir is not None:
            shutil.rmtree(launcher_dir, ignore_errors=True)
