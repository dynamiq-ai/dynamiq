import logging
from unittest.mock import patch

import pytest

from dynamiq.connections import GoogleDocumentAI

TOKEN_URI = "https://oauth2.googleapis.com/token"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Google env vars would otherwise seed the connection defaults."""
    for key in list(__import__("os").environ):
        if key.startswith(("GOOGLE_CLOUD_", "GOOGLE_DOCUMENT_AI_")):
            monkeypatch.delenv(key, raising=False)


def make_connection(**overrides):
    params = {"project_id": "proj"}
    params.update(overrides)
    return GoogleDocumentAI(**params)


def service_account_fields():
    return {
        "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
        "client_email": "sa@proj.iam.gserviceaccount.com",
        "token_uri": TOKEN_URI,
    }


class TestApiEndpoint:
    def test_endpoint_defaults_to_the_us_multi_region(self):
        assert make_connection().api_endpoint == "us-documentai.googleapis.com"

    @pytest.mark.parametrize("location", ["eu", "us-central1", "asia-southeast1"])
    def test_endpoint_is_scoped_to_the_configured_location(self, location):
        assert make_connection(location=location).api_endpoint == f"{location}-documentai.googleapis.com"

    def test_location_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_DOCUMENT_AI_LOCATION", "eu")

        assert make_connection().location == "eu"


class TestConnect:
    def test_client_is_built_against_the_location_endpoint(self):
        connection = make_connection(location="eu")

        with patch("google.cloud.documentai.DocumentProcessorServiceClient") as client_class:
            connection.connect()

        assert client_class.call_args.kwargs["client_options"].api_endpoint == "eu-documentai.googleapis.com"

    def test_complete_service_account_is_passed_as_credentials(self):
        connection = make_connection(**service_account_fields())

        with (
            patch("google.cloud.documentai.DocumentProcessorServiceClient") as client_class,
            patch("google.oauth2.service_account.Credentials.from_service_account_info") as from_info,
        ):
            connection.connect()

        from_info.assert_called_once()
        assert from_info.call_args.args[0]["client_email"] == "sa@proj.iam.gserviceaccount.com"
        assert client_class.call_args.kwargs["credentials"] is from_info.return_value

    def test_unset_fields_are_not_passed_as_nulls(self):
        """google-auth reads a null universe_domain as a non-default universe and switches to
        self-signed JWT auth, which the Document AI endpoint rejects."""
        connection = make_connection(**service_account_fields())

        with (
            patch("google.cloud.documentai.DocumentProcessorServiceClient"),
            patch("google.oauth2.service_account.Credentials.from_service_account_info") as from_info,
        ):
            connection.connect()

        info = from_info.call_args.args[0]
        assert "universe_domain" not in info
        assert all(value for value in info.values())

    def test_absent_service_account_falls_back_to_adc_silently(self, caplog):
        connection = make_connection()

        with patch("google.cloud.documentai.DocumentProcessorServiceClient") as client_class:
            with caplog.at_level(logging.WARNING):
                connection.connect()

        assert client_class.call_args.kwargs["credentials"] is None
        assert "incomplete service account credentials" not in caplog.text

    def test_partial_service_account_falls_back_to_adc_with_warning(self, caplog):
        """A partial service account makes google-auth raise instead of falling back."""
        fields = service_account_fields()
        del fields["token_uri"]
        connection = make_connection(**fields)

        with patch("google.cloud.documentai.DocumentProcessorServiceClient") as client_class:
            with caplog.at_level(logging.WARNING):
                connection.connect()

        assert client_class.call_args.kwargs["credentials"] is None
        assert "incomplete service account credentials" in caplog.text


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

    def test_project_id_alone_does_not_read_as_a_service_account(self, caplog):
        """Every Document AI connection sets project_id, so it must not trigger the warning."""
        with patch("google.cloud.documentai.DocumentProcessorServiceClient"):
            with caplog.at_level(logging.WARNING):
                make_connection(project_id="proj").connect()

        assert "incomplete service account credentials" not in caplog.text


class TestProcessorIsNotPartOfTheConnection:
    """One project/location pair hosts many processors, so the processor belongs to the node."""

    def test_connection_has_no_processor_fields(self):
        assert not hasattr(make_connection(), "processor_id")
        assert not hasattr(make_connection(), "processor_version")
