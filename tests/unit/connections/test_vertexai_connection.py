import logging

import pytest

from dynamiq.connections import VertexAI
from dynamiq.nodes.embedders.vertexai import VertexAIDocumentEmbedder

ENDPOINT_DNS = "999999.us-east1-123.prediction.vertexai.goog"
TOKEN_URI = "https://oauth2.googleapis.com/token"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Vertex/Google env vars would otherwise seed the connection defaults."""
    for key in list(__import__("os").environ):
        if key.startswith(("GOOGLE_CLOUD_", "VERTEXAI_")):
            monkeypatch.delenv(key, raising=False)


def make_connection(**overrides):
    params = {"vertex_project_id": "proj", "vertex_project_location": "us-east1"}
    params.update(overrides)
    return VertexAI(**params)


def service_account_fields():
    return {
        "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
        "client_email": "sa@proj.iam.gserviceaccount.com",
        "token_uri": TOKEN_URI,
    }


class TestApiBaseConstruction:
    def test_builds_api_base_from_endpoint_id_and_dns(self):
        connection = make_connection(vertex_endpoint_id="123", vertex_endpoint_dns=ENDPOINT_DNS)

        assert connection.api_base == (
            f"https://{ENDPOINT_DNS}/v1/projects/proj/locations/us-east1/endpoints/123"
        )

    def test_strips_scheme_and_trailing_slash_from_dns(self):
        connection = make_connection(
            vertex_endpoint_id="123", vertex_endpoint_dns=f"https://{ENDPOINT_DNS}/"
        )

        assert connection.api_base == (
            f"https://{ENDPOINT_DNS}/v1/projects/proj/locations/us-east1/endpoints/123"
        )

    def test_explicit_api_base_takes_precedence(self):
        connection = make_connection(
            vertex_api_base="https://custom.example/v1/foo/",
            vertex_endpoint_id="123",
            vertex_endpoint_dns=ENDPOINT_DNS,
        )

        assert connection.api_base == "https://custom.example/v1/foo"

    def test_no_endpoint_configuration_yields_no_api_base(self):
        assert make_connection().api_base is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"vertex_endpoint_id": "123"},
            {"vertex_endpoint_dns": ENDPOINT_DNS},
        ],
        ids=["id_only", "dns_only"],
    )
    def test_incomplete_endpoint_configuration_warns_and_yields_none(self, overrides, caplog):
        connection = make_connection(**overrides)

        with caplog.at_level(logging.WARNING):
            assert connection.api_base is None

        assert "incomplete self-deployed endpoint configuration" in caplog.text

    @pytest.mark.parametrize(
        "overrides",
        [
            {"vertex_endpoint_id": "   ", "vertex_endpoint_dns": ENDPOINT_DNS},
            {"vertex_endpoint_id": "123", "vertex_endpoint_dns": "   "},
        ],
        ids=["blank_id", "blank_dns"],
    )
    def test_blank_endpoint_fields_are_treated_as_unconfigured(self, overrides):
        """Whitespace must not produce a malformed URL such as `/endpoints/  `."""
        assert make_connection(**overrides).api_base is None

    @pytest.mark.parametrize("location", ["", "   "], ids=["empty", "blank"])
    def test_blank_location_yields_no_api_base(self, location):
        """A blank location would otherwise produce `/locations//`."""
        connection = make_connection(
            vertex_project_location=location,
            vertex_endpoint_id="123",
            vertex_endpoint_dns=ENDPOINT_DNS,
        )

        assert connection.api_base is None


class TestServiceAccountCredentials:
    def test_complete_service_account_is_reported_present(self):
        assert make_connection(**service_account_fields()).has_service_account_credentials is True

    def test_absent_service_account_is_reported_missing(self):
        assert make_connection().has_service_account_credentials is False

    @pytest.mark.parametrize("missing", ["private_key", "client_email", "token_uri"])
    def test_partial_service_account_is_reported_missing(self, missing):
        fields = service_account_fields()
        del fields[missing]

        assert make_connection(**fields).has_service_account_credentials is False

    def test_complete_service_account_is_passed_to_litellm(self):
        params = make_connection(**service_account_fields()).conn_params

        assert "vertex_credentials" in params

    def test_partial_service_account_falls_back_to_adc_with_warning(self, caplog):
        fields = service_account_fields()
        del fields["token_uri"]

        with caplog.at_level(logging.WARNING):
            params = make_connection(**fields).conn_params

        assert "vertex_credentials" not in params
        assert "incomplete service account credentials" in caplog.text

    def test_absent_service_account_falls_back_to_adc_silently(self, caplog):
        with caplog.at_level(logging.WARNING):
            params = make_connection().conn_params

        assert "vertex_credentials" not in params
        assert "incomplete service account credentials" not in caplog.text


class TestApiBaseIsScopedToCompletions:
    """A self-deployed endpoint is a chat endpoint - it must not reach embeddings or images."""

    def test_conn_params_excludes_api_base(self):
        connection = make_connection(vertex_endpoint_id="123", vertex_endpoint_dns=ENDPOINT_DNS)

        assert "api_base" not in connection.conn_params

    def test_completion_params_includes_api_base(self):
        connection = make_connection(vertex_endpoint_id="123", vertex_endpoint_dns=ENDPOINT_DNS)

        assert connection.completion_params["api_base"] == connection.api_base

    def test_completion_params_keeps_the_base_connection_params(self):
        connection = make_connection(vertex_endpoint_id="123", vertex_endpoint_dns=ENDPOINT_DNS)

        assert connection.completion_params["vertex_project"] == "proj"
        assert connection.completion_params["vertex_location"] == "us-east1"

    def test_completion_params_omits_api_base_when_not_configured(self):
        assert "api_base" not in make_connection().completion_params

    def test_completion_params_defaults_to_conn_params(self):
        """Every other provider must be unaffected by the new hook."""
        from dynamiq.connections import Cohere

        connection = Cohere(api_key="key")

        assert connection.completion_params == connection.conn_params

    def test_embedder_does_not_receive_the_chat_api_base(self):
        connection = make_connection(vertex_endpoint_id="123", vertex_endpoint_dns=ENDPOINT_DNS)

        embedder = VertexAIDocumentEmbedder(connection=connection)

        assert "api_base" not in embedder.document_embedder.embed_params
