"""Shared Neo4j fixture for the knowledge-graph E2E tests.

``test_acl_e2e.py``, ``test_retrieval_e2e.py`` and ``test_writer_e2e.py`` isolate themselves by wiping the
WHOLE graph, and CI points all three at a single shared ``neo4j:5-community`` container. Under ``pytest -n``
they land on different workers, so ``graph_connection`` holds an exclusive lock for the lifetime of the
module: one graph file touches Neo4j at a time, whatever the ``--dist`` mode. ``test_extraction_e2e.py``
needs no Neo4j and deliberately does not take this fixture.
"""

import fcntl
import os
import tempfile
from pathlib import Path

import pytest

from dynamiq.connections import Neo4j as Neo4jConnection

NEO4J_LOCK_PATH = Path(tempfile.gettempdir()) / "dynamiq-neo4j-e2e.lock"


@pytest.fixture(scope="module")
def graph_connection():
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("NEO4J_URI")):
        pytest.skip("OPENAI_API_KEY and NEO4J_URI required; skipping credentials-required test.")
    with open(NEO4J_LOCK_PATH, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield Neo4jConnection()
