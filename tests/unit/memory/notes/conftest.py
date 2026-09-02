import pytest

from dynamiq.memory.notes.backends import InMemoryNotesBackend


@pytest.fixture
def user_id() -> str:
    return "user-test-123"


@pytest.fixture
def other_user_id() -> str:
    return "user-other-456"


@pytest.fixture
def backend() -> InMemoryNotesBackend:
    return InMemoryNotesBackend()


@pytest.fixture
def seeded_backend(backend: InMemoryNotesBackend, user_id: str) -> InMemoryNotesBackend:
    backend.write(
        user_id=user_id,
        title="Deploy runbook",
        description="staging->prod steps and the rollback command",
        content="1. Merge to main\n2. ./deploy staging\n3. ./deploy prod",
    )
    backend.write(
        user_id=user_id,
        title="API conventions",
        description="error envelope and pagination rules",
        content="Errors: {code, message}. Pages are cursor-based.",
    )
    return backend
