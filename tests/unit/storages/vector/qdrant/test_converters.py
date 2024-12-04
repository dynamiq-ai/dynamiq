import numpy as np
from qdrant_client.http import models as rest

from dynamiq.storages.vector.qdrant.converters import convert_id, convert_qdrant_point_to_dynamiq_document


def test_convert_id_is_deterministic():
    first_id = convert_id("new-test-id")
    second_id = convert_id("new-test-id")
    assert first_id == second_id


def test_point_to_document_reverts_proper_structure_from_record_without_sparse():

    point = rest.Record(
        id="a1b2c3d4-5678-90ab-cdef-1234567890ab",
        payload={
            "id": "new-id",
            "content": "New content",
            "metadata": {
                "new_field": 42,
            },
        },
        vector=[0.5, 0.5, 0.5, 0.5],
    )
    document = convert_qdrant_point_to_dynamiq_document(point, content_key="content")
    assert "new-id" == document.id
    assert "New content" == document.content
    assert {"new_field": 42} == document.metadata
    assert 0.0 == np.sum(np.array([0.5, 0.5, 0.5, 0.5]) - document.embedding)
