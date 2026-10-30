"""A process-wide pool of Monty worker subprocesses, shared by PythonEvaluator and the PythonMonty node."""

import atexit
import os
import threading

import pydantic_monty

# A worker serves one session and is then replaced, so nothing a snippet leaves behind reaches the next
# evaluation or tenant. Starting a worker costs about a millisecond.
MONTY_MAX_CHECKOUTS_PER_WORKER = 1
# The pool otherwise starts one worker per CPU, and in a container that is the host's CPU count.
MONTY_MAX_WORKERS = 4
MONTY_CHECKOUT_TIMEOUT_SECONDS = 60.0

_pool: pydantic_monty.Monty | None = None
_pool_pid: int | None = None
_pool_lock = threading.Lock()


def get_monty_pool() -> pydantic_monty.Monty:
    """Return the process-wide Monty worker pool, starting it on first use.

    The pool is safe to use from several threads at once. A checkout waits for a free worker for up to
    MONTY_CHECKOUT_TIMEOUT_SECONDS, then raises TimeoutError. A forked child starts a pool of its own rather
    than sharing the parent's workers.
    """
    global _pool, _pool_pid
    with _pool_lock:
        if _pool is None or _pool_pid != os.getpid():
            pool = pydantic_monty.Monty(
                min_processes=1,
                max_processes=MONTY_MAX_WORKERS,
                checkout_timeout=MONTY_CHECKOUT_TIMEOUT_SECONDS,
                max_checkouts_per_worker=MONTY_MAX_CHECKOUTS_PER_WORKER,
            )
            pool.__enter__()
            atexit.register(_close_pool, pool, os.getpid())
            _pool, _pool_pid = pool, os.getpid()
        return _pool


def _close_pool(pool: pydantic_monty.Monty, owner_pid: int) -> None:
    # A forked child inherits this handler, but the workers belong to the process that started them.
    if os.getpid() == owner_pid:
        pool.__exit__(None, None, None)
