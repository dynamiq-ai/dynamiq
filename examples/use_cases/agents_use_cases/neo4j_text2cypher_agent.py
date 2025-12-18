from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import Neo4jCypherExecutor, Neo4jGraphWriter, Neo4jSchemaIntrospector, Neo4jText2Cypher
from dynamiq.nodes.types import InferenceMode


def build_llm() -> OpenAI:
    return OpenAI(
        name="gpt-4o-mini",
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2048,
    )


def build_connection() -> Neo4jConnection:
    return Neo4jConnection()


def build_readonly_agent() -> Agent:
    """
    Flow 1: assumes graph is already populated. Tools: Text2Cypher (read-only) + CypherExecutor.
    """
    llm = build_llm()
    connection = build_connection()

    text2cypher = Neo4jText2Cypher(llm=llm, name="text2cypher")
    cypher_executor = Neo4jCypherExecutor(connection=connection, name="cypher_executor")
    schema = Neo4jSchemaIntrospector(connection=connection, name="schema_introspector")

    return Agent(
        name="neo4j_reader",
        llm=llm,
        tools=[schema, text2cypher, cypher_executor],
        role=(
            "You answer questions against Neo4j. "
            "If schema is unknown, first call schema_introspector to fetch labels and properties. "
            "Then call text2cypher to get a parameterized query (allow_writes should remain false), "
            "then execute it with cypher_executor. "
            "Use the XML protocol exactly with <action>schema_introspector</action>, "
            "<action>text2cypher</action>, or <action>cypher_executor</action>."
        ),
        inference_mode=InferenceMode.XML,
        max_loops=6,
    )


def build_ingest_agent() -> Agent:
    """
    Flow 2: can ingest facts then query. Tools: GraphWriter + Text2Cypher (writes allowed) + CypherExecutor.
    """
    llm = build_llm()
    connection = build_connection()

    text2cypher = Neo4jText2Cypher(llm=llm, name="text2cypher_with_writes")
    cypher_executor = Neo4jCypherExecutor(connection=connection, name="cypher_executor")
    graph_writer = Neo4jGraphWriter(connection=connection, name="graph_writer")
    schema = Neo4jSchemaIntrospector(connection=connection, name="schema_introspector")

    return Agent(
        name="neo4j_ingest_reader",
        llm=llm,
        tools=[schema, graph_writer, text2cypher, cypher_executor],
        role=(
            "You can ingest data into Neo4j (graph_writer) and then answer questions. "
            "Use the XML protocol exactly: <action>schema_introspector</action>, "
            "<action>graph_writer</action>, <action>text2cypher_with_writes</action>, or "
            "<action>cypher_executor</action> with matching <action_input> JSON. "
            "For new raw text, extract key entities/relations, write them with graph_writer using:\n"
            '  {"nodes": [{"labels": ["Label"], "identity_key": "id", "properties": {"id": "x", "name": "..."}}],\n'
            '   "relationships": [{"type": "REL", "start_label": "A", '
            '"start_identity_key": "id", "start_identity": "x", '
            '"end_label": "B", "end_identity_key": "id", "end_identity": "y", "properties": {}}]}\n'
            "Then generate Cypher via text2cypher with allow_writes=True"
            " when writes are intended, and execute with cypher_executor."
        ),
        inference_mode=InferenceMode.XML,
        max_loops=8,
    )


def seed_sample_graph(connection: Neo4jConnection) -> dict:
    """
    Populate a small demo graph for testing.

    Nodes:
        (Company {id: 'dynamiq', name: 'Dynamiq'})
        (Product {id: 'dynamiq-platform', name: 'Dynamiq Platform'})
        (Person {id: 'alice', name: 'Alice'})
        (Person {id: 'bob', name: 'Bob'})
    Relationships:
        Alice WORKS_AT Dynamiq
        Bob WORKS_AT Dynamiq
        Dynamiq BUILDS Dynamiq Platform
    """
    writer = Neo4jGraphWriter(connection=connection, name="graph_writer_seed")
    nodes = [
        {"labels": ["Company"], "identity_key": "id", "properties": {"id": "dynamiq", "name": "Dynamiq"}},
        {
            "labels": ["Product"],
            "identity_key": "id",
            "properties": {"id": "dynamiq-platform", "name": "Dynamiq Platform"},
        },
        {"labels": ["Person"], "identity_key": "id", "properties": {"id": "alice", "name": "Alice"}},
        {"labels": ["Person"], "identity_key": "id", "properties": {"id": "bob", "name": "Bob"}},
    ]
    relationships = [
        {
            "type": "WORKS_AT",
            "start_label": "Person",
            "start_identity_key": "id",
            "start_identity": "alice",
            "end_label": "Company",
            "end_identity_key": "id",
            "end_identity": "dynamiq",
            "properties": {},
        },
        {
            "type": "WORKS_AT",
            "start_label": "Person",
            "start_identity_key": "id",
            "start_identity": "bob",
            "end_label": "Company",
            "end_identity_key": "id",
            "end_identity": "dynamiq",
            "properties": {},
        },
        {
            "type": "BUILDS",
            "start_label": "Company",
            "start_identity_key": "id",
            "start_identity": "dynamiq",
            "end_label": "Product",
            "end_identity_key": "id",
            "end_identity": "dynamiq-platform",
            "properties": {"since": 2024},
        },
    ]

    result = writer.run(
        input_data={"nodes": nodes, "relationships": relationships},
        config=None,
    )
    return result.output


if __name__ == "__main__":
    connection = build_connection()
    seed_result = seed_sample_graph(connection)
    print("Seeded sample graph:", seed_result)

    agent = build_readonly_agent()
    print("Built read-only agent:", agent.name)
    reader_result = agent.run(input_data={"input": "Who works at Dynamiq? Return names and roles if present."})
    print("Reader agent answer:", reader_result.output)

    ingest_agent = build_ingest_agent()
    print("Built ingest agent:", ingest_agent.name)
    ingest_prompt = "Add that Charlie works at Dynamiq as a Product Manager, then list everyone who works at Dynamiq."
    ingest_result = ingest_agent.run(input_data={"input": ingest_prompt})
    print("Ingest agent answer:", ingest_result.output)
