from pathlib import Path

import pytest

from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointConfig

DEFAULT_MAX_CHECKPOINTS = 10


@pytest.fixture
def memory_backend():
    """Create a fresh in-memory checkpoint backend for each test."""
    return InMemory()


@pytest.fixture
def filesystem_backend(tmp_path: Path):
    """Create a filesystem checkpoint backend in a temporary directory."""
    return FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))


@pytest.fixture
def checkpoint(memory_backend):
    """Create a checkpoint config with in-memory backend."""
    return CheckpointConfig(
        enabled=True,
        backend=memory_backend,
        max_checkpoints=DEFAULT_MAX_CHECKPOINTS,
    )


@pytest.fixture
def filesystem_checkpoint(filesystem_backend):
    """Create a checkpoint config with filesystem backend."""
    return CheckpointConfig(
        enabled=True,
        backend=filesystem_backend,
        max_checkpoints=DEFAULT_MAX_CHECKPOINTS,
    )


@pytest.fixture
def disabled_checkpoint():
    """Create a disabled checkpoint config."""
    return CheckpointConfig(enabled=False)
