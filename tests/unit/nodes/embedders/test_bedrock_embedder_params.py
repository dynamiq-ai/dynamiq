import pytest

from dynamiq.components.embedders.bedrock import BedrockEmbedder
from dynamiq.connections import AWS
from dynamiq.nodes.embedders import BedrockDocumentEmbedder, BedrockTextEmbedder

COHERE_V4 = "cohere.embed-v4:0"
TITAN = "amazon.titan-embed-text-v2:0"


@pytest.fixture
def connection():
    return AWS(access_key_id="key", secret_access_key="secret", region_name="us-east-1")


@pytest.mark.parametrize(
    ("node_cls", "attr", "input_type"),
    [
        (BedrockDocumentEmbedder, "document_embedder", "search_document"),
        (BedrockTextEmbedder, "text_embedder", "search_query"),
    ],
)
def test_cohere_params_reach_embed_call(connection, node_cls, attr, input_type):
    node = node_cls(connection=connection, model=COHERE_V4, dimensions=1024)
    node.init_components()

    params = getattr(node, attr).embed_params

    assert params["input_type"] == input_type
    assert params["truncate"] == "NONE"
    assert params["dimensions"] == 1024


def test_explicit_input_type_overrides_default(connection):
    node = BedrockTextEmbedder(connection=connection, model=COHERE_V4, input_type="clustering")
    node.init_components()

    assert node.text_embedder.embed_params["input_type"] == "clustering"


def test_cohere_params_not_sent_to_titan(connection):
    node = BedrockDocumentEmbedder(connection=connection, model=TITAN, dimensions=512)
    node.init_components()

    params = node.document_embedder.embed_params

    assert not {"input_type", "truncate", "dimensions"} & params.keys()


def test_component_without_node_fields_keeps_litellm_default(connection):
    params = BedrockEmbedder(connection=connection, model=COHERE_V4).embed_params

    assert params["input_type"] is None
    assert "dimensions" not in params


def test_new_fields_round_trip_through_dict(connection):
    node = BedrockDocumentEmbedder(connection=connection, model=COHERE_V4, input_type="clustering", dimensions=512)

    data = node.to_dict()

    assert data["input_type"] == "clustering"
    assert data["truncate"] == "NONE"
    assert data["dimensions"] == 512


def test_dimensions_not_sent_to_cohere_v3(connection):
    node = BedrockTextEmbedder(connection=connection, model="cohere.embed-english-v3", dimensions=512)
    node.init_components()

    params = node.text_embedder.embed_params

    assert params["input_type"] == "search_query"
    assert "dimensions" not in params


def test_cohere_truncate_skips_client_side_truncation(connection):
    node = BedrockDocumentEmbedder(connection=connection, model=COHERE_V4)
    node.init_components()
    long_text = "word " * 20000

    assert node.document_embedder._apply_text_truncation(long_text) == long_text


def test_titan_keeps_client_side_truncation(connection):
    node = BedrockDocumentEmbedder(connection=connection, model=TITAN)
    node.init_components()
    long_text = "word " * 20000

    assert len(node.document_embedder._apply_text_truncation(long_text)) < len(long_text)
