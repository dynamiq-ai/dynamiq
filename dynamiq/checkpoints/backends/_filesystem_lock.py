"""Advisory locking for compound operations on a checkpoint directory."""

from __future__ import annotations

import errno
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@contextmanager
def directory_lock(directory: Path) -> Iterator[None]:
    """Serialize cooperating backend instances, including separate processes.

    Each acquisition opens its own descriptor. On Unix, flock associates ownership
    with that open file description, so independent threads also exclude one another.
    The lock file must remain in place: unlinking it would allow a new caller to lock
    a different inode while another process still owns the original lock.
    Callers must not recursively acquire this lock.
    """
    with (directory / ".checkpoint.lock").open("a+b") as lock_file:
        if os.name == "nt":
            # Windows byte-range locks require a byte to lock, including on an empty store.
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
        else:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
