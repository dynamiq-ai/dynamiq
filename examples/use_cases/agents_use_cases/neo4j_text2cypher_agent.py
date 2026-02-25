from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import CypherExecutor
from dynamiq.nodes.types import InferenceMode


def build_llm() -> OpenAI:
    return OpenAI(
        name="gpt-5",
        model="gpt-5",
        temperature=0,
        max_tokens=2048,
    )


def build_connection() -> Neo4jConnection:
    return Neo4jConnection()


def build_readonly_agent() -> Agent:
    """
    Flow 1: assumes graph is already populated. Tool: CypherExecutor.
    """
    llm = build_llm()
    connection = build_connection()

    cypher_executor = CypherExecutor(connection=connection, name="cypher_executor")

    return Agent(
        name="neo4j_reader",
        llm=llm,
        tools=[cypher_executor],
        role=(
            "You answer questions against Neo4j. "
            "If schema is unknown, call cypher_executor with mode=introspect to fetch labels and properties. "
            "Then execute a parameterized read query with cypher_executor (allow_writes=false). "
            "Use only mode=execute or mode=introspect; do not use mode=r/w. "
            "Use routing='r' for read queries when supported. "
            "Avoid comma-separated MATCH patterns that create cartesian products; prefer chained MATCH clauses "
            "or explicit relationship patterns. "
            "Use the XML protocol exactly with <action>cypher_executor</action> and <action_input> JSON. "
            "Do not use self-closing tool tags like <cypher_executor/>."
        ),
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=10,
    )


def build_ingest_agent() -> Agent:
    """
    Flow 2: can ingest facts then query. Tool: CypherExecutor with allow_writes=true for write queries.
    """
    llm = build_llm()
    connection = build_connection()

    cypher_executor = CypherExecutor(connection=connection, name="cypher_executor")

    return Agent(
        name="neo4j_ingest_reader",
        llm=llm,
        tools=[cypher_executor],
        role=(
            "You can ingest data into Neo4j and then answer questions. "
            "Use the XML protocol exactly: <action>cypher_executor</action> with matching <action_input> JSON. "
            "Do not use self-closing tool tags like <cypher_executor/>. "
            "If schema is unknown, call cypher_executor with mode=introspect. "
            "For new raw text, extract key entities/relations, write them via cypher_executor with allow_writes=true, "
            "then query with allow_writes=false and provide the final answer. "
            "Use only mode=execute or mode=introspect; do not use mode=r/w. "
            "Use routing='r' for read queries when supported. "
            "Avoid comma-separated MATCH patterns (cartesian products). Prefer MATCH ... WITH ... MATCH or "
            "single MATCH with relationship patterns, and use MERGE on the relationship pattern directly. "
            "If duplicates exist, prefer matching by stable ids if available and avoid creating new nodes by name. "
            "For write-then-read flows, you may send a list of queries in one call."
        ),
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=10,
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
    executor = CypherExecutor(connection=connection, name="cypher_executor_seed")
    executor.init_components()
    query = """
    MERGE (c:Company {id: $company_id})
    SET c.name = $company_name
    MERGE (p:Product {id: $product_id})
    SET p.name = $product_name
    MERGE (a:Person {id: $alice_id})
    SET a.name = $alice_name
    MERGE (b:Person {id: $bob_id})
    SET b.name = $bob_name
    MERGE (a)-[:WORKS_AT]->(c)
    MERGE (b)-[:WORKS_AT]->(c)
    MERGE (c)-[:BUILDS {since: $since}]->(p)
    """
    params = {
        "company_id": "dynamiq",
        "company_name": "Dynamiq",
        "product_id": "dynamiq-platform",
        "product_name": "Dynamiq Platform",
        "alice_id": "alice",
        "alice_name": "Alice",
        "bob_id": "bob",
        "bob_name": "Bob",
        "since": 2024,
    }

    result = executor.run(
        input_data={"query": query, "parameters": params, "allow_writes": True},
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
