import uuid
from datetime import datetime
from typing import Any

from pydantic import ConfigDict, PrivateAttr
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory.long_term.base import LongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact

_UUID_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")


def _to_point_id(fact_id: str) -> str:
    """Map a fact_id to a deterministic Qdrant UUID (Qdrant requires UUID/uint ids)."""
    return uuid.uuid5(_UUID_NAMESPACE, fact_id).hex


def _scope_to_filter(scope: dict[str, str]) -> Filter | None:
    if not scope:
        return None
    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in scope.items()
        ]
    )


def _fact_to_payload(fact: Fact) -> dict:
    return {
        "fact_id": fact.id,
        "content": fact.content,
        "hash": fact.hash,
        "user_id": fact.user_id,
        "metadata": fact.metadata,
        "created_at": fact.created_at.isoformat(),
        "updated_at": fact.updated_at.isoformat(),
    }


def _payload_to_fact(payload: dict) -> Fact:
    return Fact(
        id=payload["fact_id"],
        content=payload["content"],
        hash=payload["hash"],
        user_id=payload["user_id"],
        metadata=payload.get("metadata", {}),
        created_at=datetime.fromisoformat(payload["created_at"]),
        updated_at=datetime.fromisoformat(payload["updated_at"]),
    )


class QdrantLongTermMemoryBackend(LongTermMemoryBackend):
    """Long-term memory backend backed by Qdrant."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "qdrant-long-term-memory-backend"
    connection: QdrantConnection
    collection_name: str = "user_facts"
    dimension: int = 1536

    _client: QdrantClient | None = PrivateAttr(default=None)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return super().to_dict_exclude_params | {"_client": True, "connection": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        data = super().to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)
        data["connection"] = self.connection.to_dict(
            for_tracing=for_tracing, include_secure_params=include_secure_params, **kwargs
        )
        return data

    def model_post_init(self, __context) -> None:
        self._client = self.connection.connect()

    def _ensure_storage(self) -> None:
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Create the facts collection and payload indexes if absent. Safe to call repeatedly."""
        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
            for key in ("user_id", "hash"):
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=key,
                    field_schema="keyword",
                )

    def recreate_collection(self) -> None:
        """Drop and re-create the facts collection. Test-only helper."""
        if self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)
        self.ensure_collection()

    def drop_collection(self) -> None:
        """Drop the facts collection if it exists. Test-only helper."""
        if self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)

    def insert(self, fact: Fact, embedding: list[float]) -> None:
        self._client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=_to_point_id(fact.id),
                    vector=list(embedding),
                    payload=_fact_to_payload(fact),
                )
            ],
        )

    def get(self, fact_id: str) -> Fact | None:
        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[_to_point_id(fact_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        return _payload_to_fact(results[0].payload)

    def get_by_hash(self, *, user_id: str, content_hash: str) -> Fact | None:
        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="hash", match=MatchValue(value=content_hash)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not points:
            return None
        return _payload_to_fact(points[0].payload)

    def delete(self, fact_id: str) -> None:
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[_to_point_id(fact_id)]),
        )

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
        self._client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=_to_point_id(fact_id),
                    vector=list(embedding),
                    payload=_fact_to_payload(new_fact),
                )
            ],
        )

    def search(
        self,
        *,
        query_embedding: list[float],
        scope: dict[str, str],
        limit: int,
    ) -> list[tuple[Fact, float]]:
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=list(query_embedding),
            query_filter=_scope_to_filter(scope),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [(_payload_to_fact(point.payload), float(point.score)) for point in results]

    def list_by_scope(self, scope: dict[str, str], limit: int = 100) -> list[Fact]:
        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=_scope_to_filter(scope),
            limit=limit,
            with_payload=True,
        )
        return [_payload_to_fact(p.payload) for p in points]

    def delete_scope(self, scope: dict[str, str]) -> int:
        if not scope:
            raise ValueError("delete_scope requires a non-empty scope")
        scope_filter = _scope_to_filter(scope)
        ids: list = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scope_filter,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(p.id for p in points)
            if offset is None:
                break
        if not ids:
            return 0
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )
        return len(ids)
