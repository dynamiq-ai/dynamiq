"""E2E extraction test with a real LLM (OPENAI_API_KEY required).

Feeds one document with known facts through ``KnowledgeGraphEntityExtractor`` and verifies that the
entities, the ontology-declared attributes, and the relationships stated in the text are
correctly extracted — and that the ontology / edge-metadata guarantees hold on real output.

The text deliberately contains an off-ontology fact (Acme is located in New York, but the
ontology has no Location type), so the enforcement assertions test a real drop, not a no-op.
"""

import os

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.knowledge_graphs import KnowledgeGraphEntityExtractor, Ontology, Triple
from dynamiq.nodes.knowledge_graphs.entity_extractor import ATTRIBUTE_VALUE_LABEL, HAS_ATTRIBUTE_TYPE
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.types import Document

DOCUMENT_TEXT = (
    "Jane Doe is the Chief Technology Officer (CTO) of Acme Capital, and her salary is $250,000 per year. "
    "Acme Capital is a hedge fund located in New York. Acme Capital uses the TradingX platform for all of "
    "its trades."
)

# No Location entity type and no LOCATED_IN relationship — the "located in New York" fact
# in the text is off-ontology and must not appear in the extracted graph.
ONTOLOGY = Ontology(
    entity_types=["Person", "Organization", "System"],
    relationship_types=["WORKS_AT", "USES"],
    triples=[
        Triple(source="Person", relationship="WORKS_AT", target="Organization"),
        Triple(source="Organization", relationship="USES", target="System"),
    ],
    attributes={"Person": ["title", "salary"]},
)


@pytest.fixture
def extractor():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping credentials-required test.")
    llm = OpenAI(connection=OpenAIConnection(), model="gpt-4o-mini", temperature=0)
    return KnowledgeGraphEntityExtractor(llm=llm, ontology=ONTOLOGY)


def test_extracts_entities_attributes_and_relationships(extractor):
    document = Document(
        content=DOCUMENT_TEXT,
        metadata={"allowed_principals": ["group:finance"], "source": "hr-wiki"},
    )

    result = extractor.execute(KnowledgeGraphEntityExtractor.input_schema(documents=[document]))
    nodes, relationships = result["nodes"], result["relationships"]

    by_label: dict[str, list[dict]] = {}
    for node in nodes:
        by_label.setdefault(node["labels"][0], []).append(node)

    # --- entities: the people/organizations stated in the text were extracted ---
    person_names = [(n["properties"].get("name") or "").lower() for n in by_label.get("PERSON", [])]
    org_names = [(n["properties"].get("name") or "").lower() for n in by_label.get("ORGANIZATION", [])]
    assert any("jane" in name for name in person_names), f"no Jane among PERSON nodes: {person_names}"
    assert any("acme" in name for name in org_names), f"no Acme among ORGANIZATION nodes: {org_names}"

    # --- ontology enforcement: the off-ontology location fact must not survive ---
    allowed_labels = {"PERSON", "ORGANIZATION", "SYSTEM", ATTRIBUTE_VALUE_LABEL}
    assert set(by_label) <= allowed_labels, f"off-ontology labels survived: {set(by_label) - allowed_labels}"
    all_names = [(n["properties"].get("name") or "").lower() for n in nodes]
    assert not any("new york" in name for name in all_names), "off-ontology entity 'New York' survived"

    # --- relationships: Jane works at Acme; only legal types/triples survived ---
    person_ids = {n["properties"]["id"] for n in by_label.get("PERSON", [])}
    org_ids = {n["properties"]["id"] for n in by_label.get("ORGANIZATION", [])}
    works_at = [r for r in relationships if r["type"] == "WORKS_AT"]
    assert any(
        r["start_identity"] in person_ids and r["end_identity"] in org_ids for r in works_at
    ), f"no WORKS_AT edge from a PERSON to an ORGANIZATION: {works_at}"

    rel_types = {r["type"] for r in relationships}
    assert rel_types <= {"WORKS_AT", "USES", HAS_ATTRIBUTE_TYPE}, f"illegal relationship types: {rel_types}"

    # --- attributes: declared Person attributes were promoted to HAS_ATTRIBUTE edges ---
    attr_edges = [r for r in relationships if r["type"] == HAS_ATTRIBUTE_TYPE]
    attr_keys = {r["properties"].get("key") for r in attr_edges}
    assert attr_keys & {"title", "salary"}, f"expected title/salary attribute edges, got keys: {attr_keys}"

    attr_value_ids = {n["properties"]["id"] for n in by_label.get(ATTRIBUTE_VALUE_LABEL, [])}
    for edge in attr_edges:
        assert edge["start_identity"] in person_ids  # attributes are declared for Person only
        assert edge["end_identity"] in attr_value_ids  # every attribute edge points at its value node

    # ...and were NOT left as entity-node properties (they must live on edges only)
    for node in by_label.get("PERSON", []):
        assert "title" not in node["properties"] and "salary" not in node["properties"]

    # --- document metadata + provenance ride on every relationship, never on nodes ---
    for rel in relationships:
        assert rel["properties"]["allowed_principals"] == ["group:finance"]
        assert rel["properties"]["source"] == "hr-wiki"
        assert rel["properties"]["source_doc_ids"] == [str(document.id)]
    for node in nodes:
        assert "allowed_principals" not in node["properties"]
