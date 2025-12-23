from dynamiq.connections import Neptune as NeptuneConnection
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


def build_connection() -> NeptuneConnection:
    """
    Uses environment variables like NEPTUNE_HOST, NEPTUNE_PORT, NEPTUNE_USE_HTTPS, and NEPTUNE_VERIFY_SSL.
    """
    return NeptuneConnection(use_https=True, verify_ssl=False)


def build_readonly_agent() -> Agent:
    """
    Flow: assumes Neptune graph is already populated. Tool: CypherExecutor.
    """
    llm = build_llm()
    connection = build_connection()

    cypher_executor = CypherExecutor(connection=connection, name="cypher_executor")

    return Agent(
        name="neptune_reader",
        llm=llm,
        tools=[cypher_executor],
        role=(
            "You answer questions against Amazon Neptune using openCypher. "
            "If schema is unknown, call cypher_executor with mode=introspect. "
            "Then execute a parameterized read query with cypher_executor (allow_writes=false). "
            "Use only mode=execute or mode=introspect. "
            "Avoid comma-separated MATCH patterns that create cartesian products; prefer chained MATCH clauses "
            "or explicit relationship patterns. "
            "Use the XML protocol exactly with <action>cypher_executor</action> and <action_input> JSON. "
            "Do not use self-closing tool tags like <cypher_executor/>."
        ),
        inference_mode=InferenceMode.XML,
        max_loops=10,
    )


def seed_sample_graph(connection: NeptuneConnection) -> dict:
    """
    Populate a small demo graph for testing.

    Nodes:
        (Company {id: 'dynamiq', name: 'Dynamiq'})
        (Person {id: 'alice', name: 'Alice', role: 'Engineer'})
        (Person {id: 'bob', name: 'Bob', role: 'PM'})
    Relationships:
        Alice WORKS_AT Dynamiq
        Bob WORKS_AT Dynamiq
    """
    executor = CypherExecutor(connection=connection, name="cypher_executor_seed")
    executor.init_components()
    query = """
    MERGE (c:Company {id: $company_id})
    SET c.name = $company_name
    MERGE (a:Person {id: $alice_id})
    SET a.name = $alice_name, a.role = $alice_role
    MERGE (b:Person {id: $bob_id})
    SET b.name = $bob_name, b.role = $bob_role
    MERGE (a)-[:WORKS_AT]->(c)
    MERGE (b)-[:WORKS_AT]->(c)
    """
    params = {
        "company_id": "dynamiq",
        "company_name": "Dynamiq",
        "alice_id": "alice",
        "alice_name": "Alice",
        "alice_role": "Engineer",
        "bob_id": "bob",
        "bob_name": "Bob",
        "bob_role": "PM",
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
    print("Built Neptune read-only agent:", agent.name)
    result = agent.run(input_data={"input": "List up to 5 people and their roles."})
    print("Agent answer:", result.output)
