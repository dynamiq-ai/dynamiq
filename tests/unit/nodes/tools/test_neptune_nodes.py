from unittest.mock import MagicMock

from dynamiq.connections import Neptune as NeptuneConnection
from dynamiq.nodes.tools.cypher_executor import CypherExecutor
from dynamiq.storages.graph.neptune import NeptuneGraphStore


def test_cypher_executor_uses_neptune_graph_store(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(NeptuneConnection, "connect", lambda self: client)

    connection = NeptuneConnection(host="localhost", port=8182, use_https=True, verify_ssl=False)
    node = CypherExecutor(connection=connection)
    node.init_components()

    assert isinstance(node._graph_store, NeptuneGraphStore)
    assert "Neptune" in node.description
