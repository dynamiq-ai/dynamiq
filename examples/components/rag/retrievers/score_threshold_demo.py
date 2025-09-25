from __future__ import annotations

import os
from typing import Iterable

from dynamiq.components.embedders.cohere import CohereEmbedder
from dynamiq.components.retrievers.pinecone import PineconeDocumentRetriever
from dynamiq.components.retrievers.weaviate import WeaviateDocumentRetriever
from dynamiq.storages.vector import PineconeVectorStore, WeaviateVectorStore
from dynamiq.types import Document

QUERY_TEXT = "Why do retailers rely on AI copilots for store analytics and customer experience?"
DOC_MODEL = "embed-v4.0"
QUERY_MODEL = "embed-v4.0"

RAW_DOCUMENTS = (
    {
        "id": "store-analytics-copilot",
        "title": "AI Copilots for In-Store Analytics",
        "content": (
            "A national grocer deployed tablet-based copilots that summarize footfall data,"
            " promotion uptake, and shelf availability every morning. Managers now adjust"
            " staffing within minutes instead of waiting for nightly reports."
        ),
    },
    {
        "id": "visual-merchandising-ai",
        "title": "Computer Vision for Merchandising Audits",
        "content": (
            "A fashion brand captures aisle images every hour and routes them through a"
            " vision model that flags misplaced items with confidence scores. Store teams"
            " receive curated checklists ranked by potential revenue impact."
        ),
    },
    {
        "id": "clienteling-recommendations",
        "title": "Clienteling Recommendations at Scale",
        "content": (
            "Luxury associates use a chat interface that blends purchase history, local"
            " weather, and upcoming events to draft outreach messages. The co-pilot suggests"
            " bundles with predicted sell-through probability."
        ),
    },
    {
        "id": "ai-returns-triage",
        "title": "Returns Triage Automation",
        "content": (
            "A D2C electronics retailer classifies incoming support tickets by sentiment,"
            " warranty status, and urgency. High-risk cases trigger a proactive video call"
            " offer, cutting return-to-refund time in half."
        ),
    },
    {
        "id": "inventory-forecasting",
        "title": "Generative Demand Forecasts",
        "content": (
            "Regional planners mix point-of-sale feeds with local news embeddings to spot"
            " anomalies. The forecasting agent explains spikes in plain language and cites"
            " source articles for trust."
        ),
    },
    {
        "id": "omnichannel-personalization",
        "title": "Omnichannel Personalization Journeys",
        "content": (
            "A cosmetics chain unifies loyalty data, browse sessions, and in-store skin"
            " scans. The AI recommends video tutorials and follow-up sampling moments,"
            " yielding a 19% lift in repeat visits."
        ),
    },
    {
        "id": "associate-knowledge-base",
        "title": "Associate Knowledge Base Copilot",
        "content": (
            "Floor associates query a natural-language knowledge base that synthesizes"
            " policy manuals, training clips, and prior incident logs. Answers include"
            " clickable policy references and compliant phrasing suggestions."
        ),
    },
    {
        "id": "supply-resilience",
        "title": "Supply Chain Resilience Scenarios",
        "content": (
            "A home goods brand runs nightly simulations that blend vendor risk scores"
            " with geopolitical alerts. Scenario briefs highlight warehouses affected"
            " and propose alternate sourcing plans with cost deltas."
        ),
    },
    {
        "id": "checkout-queue-optimization",
        "title": "Queue Optimization Agent",
        "content": (
            "Edge cameras estimate real-time queue length and trigger the agent to page"
            " nearby associates when service levels dip. Notifications include predicted"
            " wait times and suggested lanes to open."
        ),
    },
    {
        "id": "experiential-retail-reporting",
        "title": "Experiential Retail Reporting",
        "content": (
            "An apparel flagship blends IoT dwell metrics with post-visit surveys."
            " The analytics copilot assembles weekly recaps with highlight reels,"
            " sentiment clusters, and recommended activations for the next drop."
        ),
    },
)


def _build_documents() -> list[Document]:
    return [
        Document(
            id=item["id"],
            content=item["content"],
            metadata={
                "title": item["title"],
            },
        )
        for item in RAW_DOCUMENTS
    ]


def _embed_documents(documents: list[Document]) -> list[Document]:
    embedder = CohereEmbedder(model=DOC_MODEL, input_type="search_document")
    embedder.embed_documents(documents)
    return documents


def _embed_query() -> list[float]:
    query_embedder = CohereEmbedder(model=QUERY_MODEL, input_type="search_query")
    return query_embedder.embed_text(QUERY_TEXT)["embedding"]


def _ensure_cohere_config() -> None:
    if not os.environ.get("COHERE_API_KEY"):
        raise RuntimeError("COHERE_API_KEY must be set to run this example")


def _prepare_dataset() -> tuple[list[Document], list[float]]:
    _ensure_cohere_config()
    docs = _build_documents()
    _embed_documents(docs)
    query_embedding = _embed_query()
    return docs, query_embedding


def _populate(store, documents: Iterable[Document]) -> None:
    """Replace store contents with the provided sample documents."""
    store.write_documents([doc.model_copy(deep=True) for doc in documents])
    print(f"Wrote {len(documents)} documents")


def _display_results(title: str, docs: list[Document]) -> None:
    print(f"\n{title}")
    if not docs:
        print("  (no hits)")
        return
    for doc in docs:
        print(f"  - {doc.id} | score={doc.score}")


def run_weaviate_example(documents: list[Document], query_embedding: list[float]) -> None:
    print("\n=== Weaviate ===")
    if not os.environ.get("WEAVIATE_URL"):
        print("Skipping Weaviate example: set WEAVIATE_URL and WEAVIATE_API_KEY")
        return

    index_name = os.environ.get("WEAVIATE_CLASS", "SampleRetriever")
    vector_store = WeaviateVectorStore(index_name=index_name, create_if_not_exist=True)

    _populate(vector_store, documents)

    retriever = WeaviateDocumentRetriever(vector_store=vector_store, top_k=5)

    baseline = retriever.run(query_embedding=query_embedding)["documents"]
    _display_results("All matches", baseline)

    filtered = retriever.run(query_embedding=query_embedding, similarity_threshold=0.7)["documents"]
    _display_results("Filtered with certainty >= 0.7", filtered)


def run_pinecone_example(documents: list[Document], query_embedding: list[float]) -> None:
    print("\n=== Pinecone ===")
    if not os.environ.get("PINECONE_API_KEY"):
        print("Skipping Pinecone example: set PINECONE_API_KEY")
        return

    index_name = os.environ.get("PINECONE_INDEX", "sample-retriever")

    try:
        vector_store = PineconeVectorStore(
            index_name=index_name,
            namespace=os.environ.get("PINECONE_NAMESPACE", "sample"),
            create_if_not_exist=True,
            index_type="serverless",
            cloud="aws",
            region="us-east-1",
        )
    except Exception as exc:  # pragma: no cover - environment specific
        print(f"Skipping Pinecone example: {exc}")
        return

    _populate(vector_store, documents)

    retriever = PineconeDocumentRetriever(vector_store=vector_store, top_k=5)

    baseline = retriever.run(query_embedding=query_embedding)["documents"]
    _display_results("All matches", baseline)

    filtered = retriever.run(query_embedding=query_embedding, similarity_threshold=0.5)["documents"]
    _display_results("Filtered with similarity >= 0.5", filtered)


def main() -> None:
    try:
        documents, query_embedding = _prepare_dataset()
    except RuntimeError as exc:
        print(f"Skipping retriever demos: {exc}")
        return

    print("Query:", QUERY_TEXT)
    run_weaviate_example(documents, query_embedding)
    run_pinecone_example(documents, query_embedding)


if __name__ == "__main__":
    main()
