"""Example: PostgreSQL-backed checkpoint storage.

Runs a small Input -> Python -> Output flow with checkpointing enabled and a
PostgreSQL backend. Demonstrates:

- wiring the backend via ``dynamiq.connections.PostgreSQL``
- automatic checkpoint persistence per node completion
- inspecting checkpoints (latest, list, load by id)
- chain walk over ``parent_checkpoint_id`` (APPEND mode)
- cleanup of old checkpoints
- explicit ``close()`` to release the connection

Set the connection via environment variables (defaults match the
``dynamiq.connections.PostgreSQL`` defaults)::

    POSTGRESQL_HOST=localhost
    POSTGRESQL_PORT=5432
    POSTGRESQL_DATABASE=db
    POSTGRESQL_USER=postgres
    POSTGRESQL_PASSWORD=password
"""

from __future__ import annotations

import logging
import os

from dynamiq import flows
from dynamiq.checkpoints import CheckpointBehavior, CheckpointConfig
from dynamiq.checkpoints.backends import PostgreSQL as PostgresCheckpointBackend
from dynamiq.connections import PostgreSQL as PostgresConn
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.tools import Python
from dynamiq.nodes.utils import Input, Output

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("checkpoints.postgres.demo")

TABLE_NAME = os.getenv("DYNAMIQ_CHECKPOINTS_TABLE", "flow_checkpoints_demo")


def build_flow(backend: PostgresCheckpointBackend) -> flows.Flow:
    """Input -> Python (multiply) -> Python (square) -> Output."""
    inp = Input(id="input", name="Input")
    multiply = Python(
        id="multiply",
        name="multiply-by-10",
        code="def run(input_data): return {'value': input_data.get('value', 0) * 10}",
        depends=[NodeDependency(inp)],
    )
    square = Python(
        id="square",
        name="square",
        code="def run(input_data): return {'value': input_data['multiply']['output']['content']['value'] ** 2}",
        depends=[NodeDependency(multiply)],
    )
    out = Output(id="output", name="Output", depends=[NodeDependency(square)])

    return flows.Flow(
        nodes=[inp, multiply, square, out],
        checkpoint=CheckpointConfig(
            enabled=True,
            backend=backend,
            behavior=CheckpointBehavior.APPEND,
            max_checkpoints=20,
        ),
    )


def main() -> None:
    backend = PostgresCheckpointBackend(
        connection=PostgresConn(),
        table_name=TABLE_NAME,
        create_if_not_exist=True,
    )

    try:
        flow = build_flow(backend)

        for i in range(3):
            result = flow.run_sync(input_data={"value": i + 1})
            LOGGER.info("run %d -> status=%s output=%s", i + 1, result.status, result.output)

        latest = backend.get_latest_by_flow(flow.id)
        LOGGER.info("latest checkpoint id=%s status=%s nodes=%s", latest.id, latest.status, latest.completed_node_ids)

        history = backend.get_list_by_flow(flow.id, limit=10)
        LOGGER.info("history (%d checkpoints):", len(history))
        for cp in history:
            LOGGER.info("  - %s | status=%s | parent=%s", cp.id, cp.status.value, cp.parent_checkpoint_id)

        chain = backend.get_chain(latest.id)
        LOGGER.info("chain from latest (%d entries):", len(chain))
        for cp in chain:
            LOGGER.info("  - %s (parent=%s)", cp.id, cp.parent_checkpoint_id)

        round_tripped = backend.load(latest.id)
        LOGGER.info("load() round-trip; original_input=%s", round_tripped.original_input)

        deleted = backend.cleanup_by_flow(flow.id, keep_count=2)
        LOGGER.info("cleanup_by_flow kept 2 latest, deleted %d", deleted)

        remaining = backend.get_list_by_flow(flow.id, limit=10)
        LOGGER.info("remaining: %d", len(remaining))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
