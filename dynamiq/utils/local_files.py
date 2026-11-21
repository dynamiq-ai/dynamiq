"""Whether nodes may read files from the local file system by the paths their input names."""

import os
from collections.abc import Iterable

from dynamiq.utils.logger import logger

LOCAL_FILE_PATHS_ALLOWED_ENV_VAR = "DYNAMIQ_LOCAL_FILE_PATHS_ALLOWED"

_TRUE_VALUES = ("1", "true", "yes", "on")
_LOGGED_PATHS_LIMIT = 5


class LocalFilePathsNotAllowedError(PermissionError):
    """Raised when an input names a local file while reading local files by path is turned off."""


def local_file_paths_allowed() -> bool:
    """Whether nodes may read the local files their input names by path.

    Allowed unless the ``DYNAMIQ_LOCAL_FILE_PATHS_ALLOWED`` environment variable is set. Once it is set, any value but
    "1", "true", "yes" or "on" turns reading off, so a mistyped value refuses rather than allows. A server that runs
    workflows for other people turns it off, so that no input can make a node read a file of the server; files then
    arrive as file objects or bytes. Only the process environment decides: no node field, workflow or run setting can
    turn it back on.
    """
    value = os.getenv(LOCAL_FILE_PATHS_ALLOWED_ENV_VAR)
    if value is None:
        return True
    return value.strip().lower() in _TRUE_VALUES


def ensure_local_file_paths_allowed(paths: Iterable[str | os.PathLike]) -> None:
    """Raise LocalFilePathsNotAllowedError when ``paths`` is not empty and reading local files by path is turned off.

    Call it before the paths touch the file system, so that a refused path is not even checked for existence.
    """
    paths = [str(path) for path in paths]
    if not paths or local_file_paths_allowed():
        return

    logger.warning(
        "Refused to read %d local file path(s) %s: %s turns reading local files by path off.",
        len(paths),
        paths[:_LOGGED_PATHS_LIMIT],
        LOCAL_FILE_PATHS_ALLOWED_ENV_VAR,
    )
    raise LocalFilePathsNotAllowedError(
        f"Reading local files by path is turned off by {LOCAL_FILE_PATHS_ALLOWED_ENV_VAR}. "
        "Pass the files as file objects or bytes instead."
    )
