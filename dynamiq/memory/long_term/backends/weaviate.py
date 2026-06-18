import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.connections import Weaviate as WeaviateConnection
from dynamiq.memory.long_term.base import LongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact

if TYPE_CHECKING:
    from weaviate import WeaviateClient

_METADATA_JSON_KEY = "metadata_json"
_UUID_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")


def _to_weaviate_uuid(fact_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, fact_id))


def _fact_to_properties(fact: Fact) -> dict[str, Any]:
    return {
        "fact_id": fact.id,
        "content": fact.content,
        "hash": fact.hash,
        "user_id": fact.user_id,
        _METADATA_JSON_KEY: json.dumps(fact.metadata or {}),
        "created_at": fact.created_at.isoformat(),
        "updated_at": fact.updated_at.isoformat(),
    }


def _properties_to_fact(props: dict[str, Any]) -> Fact:
    raw_meta = props.get(_METADATA_JSON_KEY) or "{}"
    return Fact(
        id=props["fact_id"],
        content=props["content"],
        hash=props["hash"],
        user_id=props["user_id"],
        metadata=json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta,
        created_at=datetime.fromisoformat(props["created_at"]),
        updated_at=datetime.fromisoformat(props["updated_at"]),
    )


def _scope_to_filter(scope: dict[str, str]):
    """Translate a scope dict into a weaviate v4 `Filter` expression."""
    if not scope:
        return None
    from weaviate.classes.query import Filter

    items = list(scope.items())
    expr = Filter.by_property(items[0][0]).equal(items[0][1])
    for key, value in items[1:]:
        expr = expr & Filter.by_property(key).equal(value)
    return expr


def _id_in_filter(uuids: list[str]):
    """`Filter.by_id().contains_any(...)` factored out so tests can stub it."""
    from weaviate.classes.query import Filter

    return Filter.by_id().contains_any(uuids)


class WeaviateLongTermMemoryBackend(LongTermMemoryBackend):
    """Long-term memory backend backed by Weaviate (client v4)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "weaviate-long-term-memory-backend"
    connection: WeaviateConnection = Field(default_factory=WeaviateConnection)
    collection_name: str = "UserFacts"
    dimension: int = 1536
    _SCOPE_PAGE_SIZE: int = 10_000

    _client: "WeaviateClient | None" = PrivateAttr(default=None)

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

    @property
    def _collection(self):
        """Lazy collection proxy; fetched per access (local, no network)."""
        return self._client.collections.get(self.collection_name)

    def _ensure_storage(self) -> None:
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Create the facts collection if absent. Safe to call repeatedly."""
        from weaviate.classes.config import Configure, DataType, Property, VectorDistances

        if self._client.collections.exists(self.collection_name):
            return
        self._client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
            properties=[
                Property(name="fact_id", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="hash", data_type=DataType.TEXT),
                Property(name="user_id", data_type=DataType.TEXT),
                Property(name=_METADATA_JSON_KEY, data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.TEXT),
                Property(name="updated_at", data_type=DataType.TEXT),
            ],
        )

    def recreate_collection(self) -> None:
        """Drop and re-create the facts collection. Test-only helper."""
        if self._client.collections.exists(self.collection_name):
            self._client.collections.delete(self.collection_name)
        self.ensure_collection()

    def drop_collection(self) -> None:
        """Drop the facts collection if it exists. Test-only helper."""
        if self._client.collections.exists(self.collection_name):
            self._client.collections.delete(self.collection_name)

    def insert(self, fact: Fact, embedding: list[float]) -> None:
        self._collection.data.insert(
            uuid=_to_weaviate_uuid(fact.id),
            properties=_fact_to_properties(fact),
            vector=list(embedding),
        )

    def get(self, fact_id: str) -> Fact | None:
        obj = self._collection.query.fetch_object_by_id(uuid=_to_weaviate_uuid(fact_id))
        if obj is None:
            return None
        return _properties_to_fact(obj.properties)

    def get_by_hash(self, *, user_id: str, content_hash: str) -> Fact | None:
        result = self._collection.query.fetch_objects(
            filters=_scope_to_filter({"user_id": user_id, "hash": content_hash}),
            limit=1,
        )
        objects = getattr(result, "objects", []) or []
        if not objects:
            return None
        return _properties_to_fact(objects[0].properties)

    def delete(self, fact_id: str) -> None:
        self._collection.data.delete_by_id(uuid=_to_weaviate_uuid(fact_id))

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
        self._collection.data.replace(
            uuid=_to_weaviate_uuid(fact_id),
            properties=_fact_to_properties(new_fact),
            vector=list(embedding),
        )

    def search(
        self,
        *,
        query_embedding: list[float],
        scope: dict[str, str],
        limit: int,
    ) -> list[tuple[Fact, float]]:
        if not scope:
            raise ValueError("search requires a non-empty scope")
        if limit <= 0:
            return []
        from weaviate.classes.query import MetadataQuery

        result = self._collection.query.near_vector(
            near_vector=list(query_embedding),
            limit=limit,
            filters=_scope_to_filter(scope),
            return_metadata=MetadataQuery(distance=True),
        )
        objects = getattr(result, "objects", []) or []
        out: list[tuple[Fact, float]] = []
        for obj in objects:
            distance = getattr(obj.metadata, "distance", None) if obj.metadata is not None else None
            # Convert cosine distance to similarity to match Qdrant/Pinecone (`1.0 = best`).
            score = 1.0 - float(distance) if distance is not None else 0.0
            out.append((_properties_to_fact(obj.properties), score))
        return out

    def list_by_scope(self, scope: dict[str, str], limit: int = 100) -> list[Fact]:
        if not scope:
            raise ValueError("list_by_scope requires a non-empty scope")
        if limit <= 0:
            return []
        result = self._collection.query.fetch_objects(
            filters=_scope_to_filter(scope),
            limit=limit,
        )
        objects = getattr(result, "objects", []) or []
        return [_properties_to_fact(obj.properties) for obj in objects]

    def delete_scope(self, scope: dict[str, str]) -> int:
        if not scope:
            raise ValueError("delete_scope requires a non-empty scope")
        flt = _scope_to_filter(scope)
        total = 0
        while True:
            result = self._collection.query.fetch_objects(filters=flt, limit=self._SCOPE_PAGE_SIZE)
            objects = getattr(result, "objects", []) or []
            if not objects:
                break
            uuids = [str(o.uuid) for o in objects]
            self._collection.data.delete_many(where=_id_in_filter(uuids))
            total += len(uuids)
        return total
