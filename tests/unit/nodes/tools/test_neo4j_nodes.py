import sys
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.neo4j_cypher_executor import Neo4jCypherExecutor, Neo4jCypherInputSchema
from dynamiq.nodes.tools.neo4j_graph_writer import Neo4jGraphWriter, Neo4jGraphWriterInputSchema
from dynamiq.nodes.tools.neo4j_schema_introspector import Neo4jSchemaInputSchema, Neo4jSchemaIntrospector
from dynamiq.nodes.tools.neo4j_text2cypher import Neo4jText2Cypher, Neo4jText2CypherInputSchema
from dynamiq.runnables import RunnableConfig


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


class FakeLLM(Node):
    group: Literal[NodeGroup.LLMS] = NodeGroup.LLMS
    name: str = "Fake LLM"
    response_text: str

    def execute(self, input_data, config=None, **kwargs):
        return {"content": self.response_text}


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
def llm_factory():
    def _make(response_text: str) -> FakeLLM:
        return FakeLLM(response_text=response_text)

    return _make


@pytest.fixture
def cypher_executor_factory(dummy_connection):
    def _make(store: MagicMock) -> Neo4jCypherExecutor:
        executor = Neo4jCypherExecutor(client=MagicMock(), connection=dummy_connection)
        executor._graph_store = store
        return executor

    return _make


@pytest.fixture
def graph_writer_factory(dummy_connection):
    def _make(store: MagicMock) -> Neo4jGraphWriter:
        writer = Neo4jGraphWriter(client=MagicMock(), connection=dummy_connection)
        writer._graph_store = store
        return writer

    return _make


@pytest.fixture
def schema_introspector_factory(dummy_connection):
    def _make(store: MagicMock) -> Neo4jSchemaIntrospector:
        introspector = Neo4jSchemaIntrospector(client=MagicMock(), connection=dummy_connection)
        introspector._graph_store = store
        return introspector

    return _make


def test_cypher_executor_returns_records(cypher_executor_factory, runnable_config):
    mock_store = MagicMock()
    records = [FakeRecord({"name": "Ada"}), FakeRecord({"name": "Bob"})]
    counters = FakeCounters(nodes_created=0, relationships_created=0, properties_set=0)
    summary = SimpleNamespace(query="MATCH (n) RETURN n", counters=counters, result_available_after=12)
    mock_store.run_cypher.return_value = (records, summary, ["name"])

    executor = cypher_executor_factory(mock_store)

    input_data = Neo4jCypherInputSchema(query="MATCH (n) RETURN n", parameters={"name": "Ada"})
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

    executor = cypher_executor_factory(mock_store)

    input_data = Neo4jCypherInputSchema(query="MATCH (n)-[r]->(m) RETURN n,r,m", return_graph=True)
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


def test_graph_writer_executes_write_graph(graph_writer_factory, runnable_config):
    mock_store = MagicMock()
    mock_store.write_graph.return_value = {
        "nodes_created": 1,
        "relationships_created": 2,
        "properties_set": 3,
        "records": [],
        "keys": [],
    }

    writer = graph_writer_factory(mock_store)

    input_data = Neo4jGraphWriterInputSchema(
        nodes=[
            {"labels": ["Person"], "properties": {"id": "1", "name": "Ada"}, "identity_key": "id"},
        ],
        relationships=[
            {
                "type": "KNOWS",
                "start_label": "Person",
                "end_label": "Person",
                "start_identity": "1",
                "end_identity": "2",
                "start_identity_key": "id",
                "end_identity_key": "id",
                "properties": {"since": 2020},
            }
        ],
    )

    result = writer.execute(input_data, runnable_config)

    mock_store.write_graph.assert_called_once()
    assert result["input_preview"]["nodes"][0]["properties"]["name"] == "Ada"
    assert "Nodes created: 1" in result["content"]


def test_graph_writer_validates_identity_key():
    with pytest.raises(ValidationError):
        Neo4jGraphWriterInputSchema(
            nodes=[{"labels": ["Person"], "properties": {"name": "Ada"}}],
        )


def test_schema_introspector_returns_schema(schema_introspector_factory, runnable_config):
    mock_store = MagicMock()
    mock_store.run_cypher.side_effect = [
        ([{"label": "Person"}], None, None),
        ([{"relationshipType": "KNOWS"}], None, None),
        ([FakeRecord({"nodeLabels": ["Person"], "propertyName": "name", "propertyTypes": ["STRING"]})], None, None),
        (
            [FakeRecord({"relType": "KNOWS", "propertyName": "since", "propertyTypes": ["INTEGER"]})],
            None,
            None,
        ),
    ]

    introspector = schema_introspector_factory(mock_store)

    input_data = Neo4jSchemaInputSchema(include_properties=True)
    result = introspector.execute(input_data, runnable_config)

    assert result["labels"] == ["Person"]
    assert result["relationship_types"] == ["KNOWS"]
    assert result["node_properties"][0]["propertyName"] == "name"
    assert result["relationship_properties"][0]["propertyName"] == "since"
    assert "Labels: ['Person']" in result["content"]


def test_schema_introspector_skips_properties_when_disabled(schema_introspector_factory, runnable_config):
    mock_store = MagicMock()
    mock_store.run_cypher.side_effect = [
        ([{"label": "Person"}], None, None),
        ([{"relationshipType": "KNOWS"}], None, None),
    ]

    introspector = schema_introspector_factory(mock_store)

    input_data = Neo4jSchemaInputSchema(include_properties=False)
    result = introspector.execute(input_data, runnable_config)

    assert result["node_properties"] == []
    assert result["relationship_properties"] == []
    assert mock_store.run_cypher.call_count == 2


def test_text2cypher_parses_llm_output(llm_factory, runnable_config):
    llm = llm_factory('```json {"cypher": "MATCH (n) RETURN n LIMIT 1", "params": {}, "reasoning": "ok"} ```')
    tool = Neo4jText2Cypher(llm=llm)

    input_data = Neo4jText2CypherInputSchema(question="List nodes", allow_writes=False)
    result = tool.execute(input_data, runnable_config)

    assert result["cypher"] == "MATCH (n) RETURN n LIMIT 1"
    assert "Cypher: MATCH (n) RETURN n LIMIT 1" in result["content"]
    assert len(tool._run_depends) == 1


def test_text2cypher_blocks_writes_when_disabled(llm_factory, runnable_config):
    llm = llm_factory('{"cypher": "CREATE (n:Person)", "params": {}, "reasoning": "no"}')
    tool = Neo4jText2Cypher(llm=llm)

    input_data = Neo4jText2CypherInputSchema(question="Create a person", allow_writes=False)

    with pytest.raises(ToolExecutionException):
        tool.execute(input_data, runnable_config)


def test_text2cypher_rejects_empty_cypher(llm_factory, runnable_config):
    llm = llm_factory('{"cypher": "", "params": {}, "reasoning": "empty"}')
    tool = Neo4jText2Cypher(llm=llm)

    input_data = Neo4jText2CypherInputSchema(question="Nothing", allow_writes=True)

    with pytest.raises(ToolExecutionException):
        tool.execute(input_data, runnable_config)
