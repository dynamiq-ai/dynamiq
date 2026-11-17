from unittest.mock import patch

from dynamiq.connections import Weaviate
from dynamiq.connections.connections import WeaviateDeploymentType


def _custom(**kwargs):
    return Weaviate(
        deployment_type=WeaviateDeploymentType.CUSTOM,
        http_host="weaviate.internal",
        http_port=8080,
        grpc_host="weaviate.internal",
        grpc_port=50051,
        **kwargs,
    )


def test_a_custom_deployment_uses_tls_and_the_api_key_by_default():
    with patch("weaviate.connect_to_custom") as connect:
        _custom(api_key="secret").connect()

    kwargs = connect.call_args.kwargs
    assert kwargs["http_secure"] is True and kwargs["grpc_secure"] is True
    assert kwargs["auth_credentials"] is not None


def test_a_self_hosted_deployment_connects_over_plain_http_without_a_key():
    with patch("weaviate.connect_to_custom") as connect:
        _custom(api_key="", secure=False).connect()

    kwargs = connect.call_args.kwargs
    assert kwargs["http_secure"] is False and kwargs["grpc_secure"] is False
    assert kwargs["auth_credentials"] is None
