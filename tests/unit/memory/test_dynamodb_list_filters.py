"""DynamoDB server-side filters must treat list values as membership.

`_apply_filters_client_side_messages` (and InMemory) interpret a list filter value as
"metadata value is one of these". The server-side FilterExpression used `eq(list)`,
which compares the stored attribute to the whole list and so matched nothing.
"""

from types import SimpleNamespace

from dynamiq.memory.backends.dynamo_db import DynamoDB


def _build(filters):
    # The method only reads `metadata_attribute_name`, so no AWS client is needed.
    return DynamoDB._build_filter_expression_attr(SimpleNamespace(metadata_attribute_name="metadata"), filters)


def test_list_filter_uses_membership():
    expr = _build({"kind": ["human", "assistant"]})
    assert expr.get_expression()["operator"] == "IN"


def test_scalar_filter_still_uses_equality():
    expr = _build({"kind": "human"})
    assert expr.get_expression()["operator"] == "="


def test_empty_list_filter_is_left_to_the_client_side_filter():
    # boto3 would build "IN ()", which DynamoDB rejects as a syntax error.
    expr = _build({"kind": [], "user_id": "u1"})
    assert expr.get_expression()["operator"] == "="

    backend = SimpleNamespace(metadata_attribute_name="metadata")
    message = SimpleNamespace(metadata={"kind": "human"})
    assert DynamoDB._apply_filters_client_side_messages(backend, [message], {"kind": []}) == []
