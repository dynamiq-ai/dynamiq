import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory.long_term.base import LongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact

if TYPE_CHECKING:
    from pinecone import Pinecone as PineconeClient

_METADATA_JSON_KEY = "metadata_json"


def _scope_to_filter(scope: dict[str, str]) -> dict | None:
    """Translate a scope dict into a Pinecone metadata filter."""
    if not scope:
        return None
    if len(scope) == 1:
        (key, value), = scope.items()
        return {key: {"$eq": value}}
    return {"$and": [{key: {"$eq": value}} for key, value in scope.items()]}


def _fact_to_metadata(fact: Fact) -> dict[str, Any]:
    return {
        "fact_id": fact.id,
        "content": fact.content,
        "hash": fact.hash,
        "user_id": fact.user_id,
        _METADATA_JSON_KEY: json.dumps(fact.metadata or {}),
        "created_at": fact.created_at.isoformat(),
        "updated_at": fact.updated_at.isoformat(),
    }


def _metadata_to_fact(meta: dict[str, Any]) -> Fact:
    raw_meta = meta.get(_METADATA_JSON_KEY) or "{}"
    return Fact(
        id=meta["fact_id"],
        content=meta["content"],
        hash=meta["hash"],
        user_id=meta["user_id"],
        metadata=json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta,
        created_at=datetime.fromisoformat(meta["created_at"]),
        updated_at=datetime.fromisoformat(meta["updated_at"]),
    )


class PineconeLongTermMemoryBackend(LongTermMemoryBackend):
    """Long-term memory backend backed by Pinecone.

    The Pinecone index must be pre-created out of band — the data-plane SDK
    has no idempotent ensure_index primitive.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "pinecone-long-term-memory-backend"
    connection: PineconeConnection = Field(default_factory=PineconeConnection)
    index_name: str = "user_facts"
    namespace: str = "default"
    dimension: int = 1536
    _LIST_PAGE_SIZE: int = 10_000

    _client: "PineconeClient | None" = PrivateAttr(default=None)
    _index: Any = PrivateAttr(default=None)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return super().to_dict_exclude_params | {"_client": True, "_index": True, "connection": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        data = super().to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)
        data["connection"] = self.connection.to_dict(
            for_tracing=for_tracing, include_secure_params=include_secure_params, **kwargs
        )
        return data

    def model_post_init(self, __context) -> None:
        self._client = self.connection.connect()
        self._index = self._client.Index(name=self.index_name)

    def insert(self, fact: Fact, embedding: list[float]) -> None:
        self._index.upsert(
            vectors=[{"id": fact.id, "values": list(embedding), "metadata": _fact_to_metadata(fact)}],
            namespace=self.namespace,
        )

    def get(self, fact_id: str) -> Fact | None:
        result = self._index.fetch(ids=[fact_id], namespace=self.namespace)
        vectors = result.get("vectors") if isinstance(result, dict) else getattr(result, "vectors", {})
        if not vectors or fact_id not in vectors:
            return None
        vec = vectors[fact_id]
        meta = vec["metadata"] if isinstance(vec, dict) else vec.metadata
        return _metadata_to_fact(meta)

    def get_by_hash(self, *, user_id: str, content_hash: str) -> Fact | None:
        result = self._index.query(
            vector=[0.0] * self.dimension,
            top_k=1,
            namespace=self.namespace,
            filter=_scope_to_filter({"user_id": user_id, "hash": content_hash}),
            include_metadata=True,
        )
        matches = result.get("matches") if isinstance(result, dict) else getattr(result, "matches", [])
        if not matches:
            return None
        match = matches[0]
        meta = match["metadata"] if isinstance(match, dict) else match.metadata
        return _metadata_to_fact(meta)

    def delete(self, fact_id: str) -> None:
        self._index.delete(ids=[fact_id], namespace=self.namespace)

    def update(
        self,
        fact_id: str,
        *,
        content: str,
        content_hash: str,
        embedding: list[float],
        metadata: dict,
        updated_at: datetime,
    ) -> None:
        existing = self.get(fact_id)
        if existing is None:
            return
        new_fact = existing.model_copy(
            update={"content": content, "hash": content_hash, "metadata": metadata, "updated_at": updated_at}
        )
        self._index.upsert(
            vectors=[{"id": fact_id, "values": list(embedding), "metadata": _fact_to_metadata(new_fact)}],
            namespace=self.namespace,
        )

    def search(
        self,
        *,
        query_embedding: list[float],
        scope: dict[str, str],
        limit: int,
    ) -> list[tuple[Fact, float]]:
        result = self._index.query(
            vector=list(query_embedding),
            top_k=limit,
            namespace=self.namespace,
            filter=_scope_to_filter(scope),
            include_metadata=True,
        )
        matches = result.get("matches") if isinstance(result, dict) else getattr(result, "matches", [])
        out: list[tuple[Fact, float]] = []
        for match in matches:
            meta = match["metadata"] if isinstance(match, dict) else match.metadata
            score = match["score"] if isinstance(match, dict) else match.score
            out.append((_metadata_to_fact(meta), float(score)))
        return out

    def list_by_scope(self, scope: dict[str, str], limit: int = 100) -> list[Fact]:
        # Pinecone's top_k must be >= 1.
        if limit <= 0:
            return []
        top_k = min(max(limit, 1), self._LIST_PAGE_SIZE)
        result = self._index.query(
            vector=[0.0] * self.dimension,
            top_k=top_k,
            namespace=self.namespace,
            filter=_scope_to_filter(scope),
            include_metadata=True,
        )
        matches = result.get("matches") if isinstance(result, dict) else getattr(result, "matches", [])
        return [
            _metadata_to_fact(match["metadata"] if isinstance(match, dict) else match.metadata) for match in matches
        ]

    def delete_scope(self, scope: dict[str, str]) -> int:
        if not scope:
            raise ValueError("delete_scope requires a non-empty scope")
        # Pinecone Serverless supports only delete-by-id, with no cursor — loop
        # query → delete-by-id until the page is empty.
        total = 0
        flt = _scope_to_filter(scope)
        while True:
            result = self._index.query(
                vector=[0.0] * self.dimension,
                top_k=self._LIST_PAGE_SIZE,
                namespace=self.namespace,
                filter=flt,
                include_metadata=False,
            )
            matches = result.get("matches") if isinstance(result, dict) else getattr(result, "matches", [])
            ids = [match["id"] if isinstance(match, dict) else match.id for match in matches]
            if not ids:
                break
            self._index.delete(ids=ids, namespace=self.namespace)
            total += len(ids)
        return total
