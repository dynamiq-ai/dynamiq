import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from dynamiq import Workflow
from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.cypher_executor import CypherExecutor, CypherInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.graph.base import BaseGraphStore


class FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class FakeCounters:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def contains_updates(self):
        return True

    def contains_system_updates(self):
        return False


class FakeNode(dict):
    def __init__(self, *, node_id, element_id, labels, properties):
        super().__init__(properties)
        self.id = node_id
        self.element_id = element_id
        self.labels = labels


class FakeRelationship(dict):
    def __init__(
        self,
        *,
        rel_id,
        element_id,
        rel_type,
        start_node_id,
        end_node_id,
        start_node_element_id,
        end_node_element_id,
        properties,
    ):
        super().__init__(properties)
        self.id = rel_id
        self.element_id = element_id
        self.type = rel_type
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.start_node_element_id = start_node_element_id
        self.end_node_element_id = end_node_element_id


@pytest.fixture
def dummy_connection() -> Neo4jConnection:
    return Neo4jConnection(
        uri="bolt://localhost:7687",
        username="user",
        password="pass",
        verify_connectivity=False,
    )


@pytest.fixture
def runnable_config() -> RunnableConfig:
    return RunnableConfig(callbacks=[])


@pytest.fixture
def cypher_executor_factory(dummy_connection):
    def _make(store: MagicMock) -> CypherExecutor:
        executor = CypherExecutor(client=MagicMock(), connection=dummy_connection)
        executor._graph_store = store
        executor._graph_store.format_records = BaseGraphStore.format_records
        if isinstance(getattr(executor._graph_store, "supports_graph_result", None), MagicMock):
            executor._graph_store.supports_graph_result = lambda: False
        return executor

    return _make


def test_cypher_executor_returns_records(cypher_executor_factory, runnable_config):
    mock_store = MagicMock()
    records = [FakeRecord({"name": "Ada"}), FakeRecord({"name": "Bob"})]
    counters = FakeCounters(nodes_created=0, relationships_created=0, properties_set=0)
    summary = SimpleNamespace(query="MATCH (n) RETURN n", counters=counters, result_available_after=12)
    mock_store.run_cypher.return_value = (records, summary, ["name"])

    executor = cypher_executor_factory(mock_store)

    input_data = CypherInputSchema(query="MATCH (n) RETURN n", parameters={"name": "Ada"})
    result = executor.execute(input_data, runnable_config)

    assert result["records"] == [{"name": "Ada"}, {"name": "Bob"}]
    assert result["keys"] == ["name"]
    assert result["summary"]["query"] == "MATCH (n) RETURN n"
    assert "Returned 2 records" in result["content"]
    assert "Params: {'name': 'Ada'}" in result["content"]


def test_cypher_executor_returns_graph(monkeypatch, cypher_executor_factory, runnable_config):
    fake_module = SimpleNamespace(Result=SimpleNamespace(graph="graph_transformer"))
    monkeypatch.setitem(sys.modules, "neo4j", fake_module)

    graph = SimpleNamespace(
        nodes=[FakeNode(node_id=1, element_id="n1", labels=["Person"], properties={"name": "Ada"})],
        relationships=[
            FakeRelationship(
                rel_id=10,
                element_id="r1",
                rel_type="KNOWS",
                start_node_id=1,
                end_node_id=2,
                start_node_element_id="n1",
                end_node_element_id="n2",
                properties={"since": 2020},
            )
        ],
    )
    counters = FakeCounters(nodes_created=0, relationships_created=0, properties_set=0)
    summary = SimpleNamespace(query="MATCH (n)-[r]->(m) RETURN n,r,m", counters=counters, result_available_after=3)

    mock_store = MagicMock()
    mock_store.run_cypher.return_value = (graph, summary, ["n", "r", "m"])
    mock_store.supports_graph_result = lambda: True

    executor = cypher_executor_factory(mock_store)

    input_data = CypherInputSchema(query="MATCH (n)-[r]->(m) RETURN n,r,m", return_graph=True)
    result = executor.execute(input_data, runnable_config)

    assert result["graph"]["nodes"][0]["labels"] == ["Person"]
    assert result["graph"]["relationships"][0]["type"] == "KNOWS"
    assert result["keys"] == []
    assert "Nodes: 1" in result["content"]
    mock_store.run_cypher.assert_called_once_with(
        query="MATCH (n)-[r]->(m) RETURN n,r,m",
        parameters={},
        database=None,
        routing=None,
        result_transformer="graph_transformer",
    )


def test_cypher_executor_introspects_schema(cypher_executor_factory, runnable_config):
    mock_store = MagicMock()
    mock_store.introspect_schema.return_value = {
        "labels": ["Person"],
        "relationship_types": ["KNOWS"],
        "node_properties": [{"nodeLabels": ["Person"], "propertyName": "name", "propertyTypes": ["STRING"]}],
        "relationship_properties": [{"relType": "KNOWS", "propertyName": "since", "propertyTypes": ["INTEGER"]}],
    }

    executor = cypher_executor_factory(mock_store)

    input_data = CypherInputSchema(mode="introspect", include_properties=True)
    result = executor.execute(input_data, runnable_config)

    assert result["labels"] == ["Person"]
    assert result["relationship_types"] == ["KNOWS"]
    assert result["node_properties"][0]["propertyName"] == "name"
    assert result["relationship_properties"][0]["propertyName"] == "since"
    assert "Labels: ['Person']" in result["content"]


def test_cypher_executor_introspect_skips_properties_when_disabled(cypher_executor_factory, runnable_config):
    mock_store = MagicMock()
    mock_store.introspect_schema.return_value = {
        "labels": ["Person"],
        "relationship_types": ["KNOWS"],
        "node_properties": [],
        "relationship_properties": [],
    }

    executor = cypher_executor_factory(mock_store)

    input_data = CypherInputSchema(mode="introspect", include_properties=False)
    result = executor.execute(input_data, runnable_config)

    assert result["node_properties"] == []
    assert result["relationship_properties"] == []
    mock_store.introspect_schema.assert_called_once_with(include_properties=False, database=None)


def test_cypher_executor_rejects_write_when_disallowed(cypher_executor_factory, runnable_config):
    mock_store = MagicMock()
    mock_store.run_cypher.return_value = (
        [],
        SimpleNamespace(query="CREATE (n)", counters=None, result_available_after=0),
        [],
    )
    executor = cypher_executor_factory(mock_store)

    input_data = CypherInputSchema(query="CREATE (n:Person)", allow_writes=False)

    with pytest.raises(ToolExecutionException):
        executor.execute(input_data, runnable_config)


def test_cypher_executor_requires_query_in_execute_mode():
    with pytest.raises(ValidationError):
        CypherInputSchema(mode="execute")


def test_cypher_executor_blocks_cartesian_writes(cypher_executor_factory, runnable_config):
    mock_store = MagicMock()
    mock_store.run_cypher.return_value = (
        [],
        SimpleNamespace(query="MATCH (a),(b)", counters=None, result_available_after=0),
        [],
    )
    executor = cypher_executor_factory(mock_store)

    input_data = CypherInputSchema(
        query="MATCH (a:Person {name: 'A'}), (b:Company {name: 'B'}) MERGE (a)-[:WORKS_AT]->(b)",
        allow_writes=True,
    )

    with pytest.raises(ToolExecutionException):
        executor.execute(input_data, runnable_config)


def test_cypher_executor_allows_cartesian_reads():
    CypherExecutor._validate_query(
        "MATCH (a:Person), (b:Company) RETURN a, b",
        allow_writes=True,
    )


def test_cypher_executor_handles_batch_queries(cypher_executor_factory, runnable_config):
    mock_store = MagicMock()
    counters = FakeCounters(nodes_created=0, relationships_created=0, properties_set=0)
    summaries = [
        SimpleNamespace(query="MATCH (n) RETURN n", counters=counters, result_available_after=1),
        SimpleNamespace(query="MATCH (m) RETURN m", counters=counters, result_available_after=2),
    ]
    mock_store.run_cypher.side_effect = [
        ([FakeRecord({"name": "Ada"})], summaries[0], ["name"]),
        ([FakeRecord({"title": "Dynamiq"})], summaries[1], ["title"]),
    ]

    executor = cypher_executor_factory(mock_store)
    input_data = CypherInputSchema(
        query=["MATCH (n) RETURN n", "MATCH (m) RETURN m"],
        parameters=[{"name": "Ada"}, {"title": "Dynamiq"}],
    )
    result = executor.execute(input_data, runnable_config)

    assert result["results"][0]["records"] == [{"name": "Ada"}]
    assert result["results"][1]["records"] == [{"title": "Dynamiq"}]
    assert "[1] Query: MATCH (n) RETURN n" in result["content"]
    assert "[2] Query: MATCH (m) RETURN m" in result["content"]


def test_cypher_executor_rejects_mismatched_batch_parameters():
    with pytest.raises(ValidationError):
        CypherInputSchema(query=["MATCH (n) RETURN n"], parameters=[{}, {}])


def test_cypher_executor_rejects_list_parameters_for_single_query():
    with pytest.raises(ValidationError):
        CypherInputSchema(query="MATCH (n) RETURN n", parameters=[{"name": "Ada"}])


def test_cypher_executor_yaml_roundtrip(tmp_path, monkeypatch):
    driver = MagicMock()
    monkeypatch.setattr(Neo4jConnection, "connect", lambda self: driver)

    connection = Neo4jConnection(
        id="neo4j-conn",
        uri="bolt://localhost:7687",
        username="user",
        password="pass",
        verify_connectivity=False,
    )
    node = CypherExecutor(id="neo4j-node", connection=connection, database="neo4j")
    workflow = Workflow(id="neo4j-workflow", flow=Flow(id="neo4j-flow", nodes=[node]))

    yaml_path = tmp_path / "neo4j_workflow.yaml"
    workflow.to_yaml_file(yaml_path)

    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
    assert isinstance(loaded.flow.nodes[0], CypherExecutor)
    assert loaded.flow.nodes[0]._graph_store is not None

    roundtrip_path = tmp_path / "neo4j_workflow_roundtrip.yaml"
    loaded.to_yaml_file(roundtrip_path)
    roundtrip = Workflow.from_yaml_file(str(roundtrip_path), init_components=True)

    roundtrip_node = roundtrip.flow.nodes[0]
    assert roundtrip_node.database == "neo4j"
    assert roundtrip_node.connection.id == "neo4j-conn"
    assert roundtrip_node.connection.uri == "bolt://localhost:7687"
