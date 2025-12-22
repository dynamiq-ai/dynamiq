import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dynamiq.connections import ApacheAge  # noqa: E402
from dynamiq.nodes.agents import Agent  # noqa: E402
from dynamiq.nodes.llms import OpenAI  # noqa: E402
from dynamiq.nodes.tools import CypherExecutor  # noqa: E402
from dynamiq.nodes.types import InferenceMode  # noqa: E402


def build_llm() -> OpenAI:
    return OpenAI(
        name="gpt-5",
        model="gpt-5",
        temperature=0,
        max_tokens=2048,
    )


def build_connection() -> ApacheAge:
    os.environ.setdefault("POSTGRESQL_HOST", "localhost")
    os.environ.setdefault("POSTGRESQL_PORT", "55432")
    os.environ.setdefault("POSTGRESQL_DATABASE", "db")
    os.environ.setdefault("POSTGRESQL_USER", os.environ.get("USER", "postgres"))
    os.environ.setdefault("POSTGRESQL_PASSWORD", "password")
    os.environ.setdefault("APACHE_AGE_GRAPH_NAME", "graph")
    return ApacheAge()


def build_readonly_agent() -> Agent:
    llm = build_llm()
    connection = build_connection()

    cypher_executor = CypherExecutor(connection=connection, name="cypher_executor")

    return Agent(
        name="age_reader",
        llm=llm,
        tools=[cypher_executor],
        role=(
            "You answer questions against Apache AGE using openCypher. "
            "If schema is unknown, call cypher_executor with mode=introspect. "
            "Then execute a parameterized read query with cypher_executor (allow_writes=false). "
            "AGE requires RETURN to include a single column aliased as `result`. "
            "Use a single query string (do not pass a list of queries). "
            "Do not set return_graph; it is only for Neo4j. "
            "Use only mode=execute or mode=introspect; do not use mode=r/w. "
            "Avoid comma-separated MATCH patterns that create cartesian products. "
            "Use the XML protocol exactly with <action>cypher_executor</action> and <action_input> JSON. "
            "Do not use self-closing tool tags like <cypher_executor/>."
        ),
        inference_mode=InferenceMode.XML,
        max_loops=10,
    )


def seed_sample_graph(connection: ApacheAge) -> dict:
    """
    Populate a small demo graph for testing in Apache AGE.

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
    CREATE (c:Company {id: $company_id, name: $company_name})
    CREATE (p:Product {id: $product_id, name: $product_name})
    CREATE (a:Person {id: $alice_id, name: $alice_name})
    CREATE (b:Person {id: $bob_id, name: $bob_name})
    CREATE (a)-[:WORKS_AT]->(c)
    CREATE (b)-[:WORKS_AT]->(c)
    CREATE (c)-[:BUILDS {since: $since}]->(p)
    RETURN c AS result
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
    _close_tool_client(executor)
    return result.output


def _close_tool_client(tool: CypherExecutor) -> None:
    graph_store = getattr(tool, "_graph_store", None)
    if graph_store is None:
        return
    client = getattr(graph_store, "client", None)
    if client and hasattr(client, "close"):
        client.close()


if __name__ == "__main__":
    connection = build_connection()
    seed_result = seed_sample_graph(connection)
    print("Seeded sample graph:", seed_result)

    agent = build_readonly_agent()
    print("Built read-only agent:", agent.name)
    result = agent.run(input_data={"input": "List all nodes in the graph"})
    print("Agent answer:", result.output)
    for tool in agent.tools:
        if isinstance(tool, CypherExecutor):
            _close_tool_client(tool)
