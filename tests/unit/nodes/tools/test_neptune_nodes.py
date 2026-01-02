import re
from unittest.mock import MagicMock

import pytest

from dynamiq.connections import AWSNeptune as AWSNeptuneConnection
from dynamiq.nodes.tools.cypher_executor import CypherExecutor
from dynamiq.storages.graph.neptune import NeptuneGraphStore


def test_cypher_executor_uses_neptune_graph_store(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(AWSNeptuneConnection, "connect", lambda self: client)

    connection = AWSNeptuneConnection(host="localhost", port=8182, use_https=True, verify_ssl=False)
    node = CypherExecutor(connection=connection)
    node.init_components()

    assert isinstance(node._graph_store, NeptuneGraphStore)
    assert "Neptune" in node.description


def test_neptune_validate_label_accepts_valid_labels():
    """Test that valid labels pass validation."""
    valid_labels = [
        "User",
        "Person",
        "Company",
        "_internal",
        "Label_With_Underscores",
        "Label123",
        "a",
        "A",
        "_",
    ]
    for label in valid_labels:
        assert NeptuneGraphStore._validate_label(label) == label


def test_neptune_validate_label_rejects_invalid_labels():
    """Test that invalid labels raise ValueError."""
    invalid_labels = [
        "User`] RETURN * //",  # Cypher injection attempt
        "Label-With-Dash",  # Contains dash
        "Label With Space",  # Contains space
        "123Label",  # Starts with number
        "Label:Type",  # Contains colon
        "Label;DROP",  # Contains semicolon
        "Label`",  # Contains backtick
        "Label'",  # Contains single quote
        'Label"',  # Contains double quote
        "",  # Empty string
        "Label\nNewline",  # Contains newline
        "Label\tTab",  # Contains tab
    ]
    for label in invalid_labels:
        with pytest.raises(ValueError, match=f"Invalid Neptune label: '{re.escape(label)}'"):
            NeptuneGraphStore._validate_label(label)


def test_neptune_sample_node_properties_validates_labels(monkeypatch):
    """Test that _sample_node_properties validates labels before query construction."""
    client = MagicMock()
    store = NeptuneGraphStore(client=client, endpoint="http://localhost:8182/openCypher")

    # Mock run_cypher to track calls
    run_cypher_calls = []

    def mock_run_cypher(query, parameters=None, **kwargs):
        run_cypher_calls.append(query)
        return [], {}, []

    store.run_cypher = mock_run_cypher

    # Valid labels should work
    store._sample_node_properties(["User", "Person"])
    assert len(run_cypher_calls) == 2
    assert "MATCH (a:`User`)" in run_cypher_calls[0]
    assert "MATCH (a:`Person`)" in run_cypher_calls[1]

    # Invalid label should raise
    with pytest.raises(ValueError, match="Invalid Neptune label"):
        store._sample_node_properties(["User`] RETURN * //"])


def test_neptune_sample_edge_properties_validates_labels(monkeypatch):
    """Test that _sample_edge_properties validates labels before query construction."""
    client = MagicMock()
    store = NeptuneGraphStore(client=client, endpoint="http://localhost:8182/openCypher")

    run_cypher_calls = []

    def mock_run_cypher(query, parameters=None, **kwargs):
        run_cypher_calls.append(query)
        return [], {}, []

    store.run_cypher = mock_run_cypher

    # Valid labels should work
    store._sample_edge_properties(["KNOWS", "WORKS_AT"])
    assert len(run_cypher_calls) == 2
    assert "MATCH ()-[e:`KNOWS`]->()" in run_cypher_calls[0]
    assert "MATCH ()-[e:`WORKS_AT`]->()" in run_cypher_calls[1]

    # Invalid label should raise
    with pytest.raises(ValueError, match="Invalid Neptune label"):
        store._sample_edge_properties(["KNOWS`] RETURN * //"])


def test_neptune_sample_triples_validates_labels(monkeypatch):
    """Test that _sample_triples validates labels before query construction."""
    client = MagicMock()
    store = NeptuneGraphStore(client=client, endpoint="http://localhost:8182/openCypher")

    run_cypher_calls = []

    def mock_run_cypher(query, parameters=None, **kwargs):
        run_cypher_calls.append(query)
        return [], {}, []

    store.run_cypher = mock_run_cypher

    # Valid labels should work
    store._sample_triples(["KNOWS"])
    assert len(run_cypher_calls) == 1
    assert "MATCH (a)-[e:`KNOWS`]->(b)" in run_cypher_calls[0]

    # Invalid label should raise
    with pytest.raises(ValueError, match="Invalid Neptune label"):
        store._sample_triples(["KNOWS`] RETURN * //"])
