import logging
import uuid
from typing import Union

from qdrant_client.http import models as rest

from dynamiq.types import Document

logger = logging.getLogger(__name__)

DENSE_VECTORS_NAME = "text-dense"
SPARSE_VECTORS_NAME = "text-sparse"

QdrantPoint = Union[rest.ScoredPoint, rest.Record]


def convert_dynamiq_documents_to_qdrant_points(
    documents: list[Document],
    *,
    use_sparse_embeddings: bool,
    content_key: str | None = None,
) -> list[rest.PointStruct]:
    points = []
    for document in documents:
        payload = document.to_dict()
        payload[content_key] = payload.pop("content", "")
        if use_sparse_embeddings:
            vector = {}

            dense_vector = payload.pop("embedding", None)
            if dense_vector is not None:
                vector[DENSE_VECTORS_NAME] = dense_vector

            sparse_vector = payload.pop("sparse_embedding", None)
            if sparse_vector is not None:
                sparse_vector_instance = rest.SparseVector(**sparse_vector)
                vector[SPARSE_VECTORS_NAME] = sparse_vector_instance

        else:
            vector = payload.pop("embedding") or {}
        _id = convert_id(payload.get("id"))

        point = rest.PointStruct(
            payload=payload,
            vector=vector,
            id=_id,
        )
        points.append(point)
    return points


def convert_id(_id: str) -> str:
    """
    Converts any string into a UUID-like format in a deterministic way.

    Qdrant does not accept any string as an id, so an internal id has to be
    generated for each point. This is a deterministic way of doing so.
    """
    UUID_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")
    return uuid.uuid5(UUID_NAMESPACE, _id).hex


def convert_qdrant_point_to_dynamiq_document(
    point: QdrantPoint, use_sparse_embeddings: bool = False, content_key: str | None = None
) -> Document:
    payload = {**point.payload}
    payload["score"] = point.score if hasattr(point, "score") else None
    payload["content"] = payload.pop(content_key, "")

    if not use_sparse_embeddings:
        payload["embedding"] = point.vector if hasattr(point, "vector") else None
    elif hasattr(point, "vector") and point.vector is not None:
        payload["embedding"] = point.vector.get(DENSE_VECTORS_NAME)

        if SPARSE_VECTORS_NAME in point.vector:
            parse_vector_dict = {
                "indices": point.vector[SPARSE_VECTORS_NAME].indices,
                "values": point.vector[SPARSE_VECTORS_NAME].values,
            }
            payload["sparse_embedding"] = parse_vector_dict

    return Document(**payload)
