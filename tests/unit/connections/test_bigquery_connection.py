import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq.connections import BigQuery, GoogleCloud
from dynamiq.connections.managers import ConnectionManager

TOKEN_URI = "https://oauth2.googleapis.com/token"
KEY_EMAIL = "agent@sa-project.iam.gserviceaccount.com"


@pytest.fixture(scope="module")
def private_key():
    """A real, throwaway RSA key, so that google-auth builds credentials offline."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()


@pytest.fixture
def service_account_key(private_key):
    return {
        "project_id": "sa-project",
        "private_key_id": "key-id",
        "private_key": private_key,
        "client_email": KEY_EMAIL,
        "client_id": "1234567890",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": TOKEN_URI,
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/agent",
        "universe_domain": "googleapis.com",
    }


@pytest.fixture(autouse=True)
def ambient_credentials(monkeypatch, private_key):
    """Credentials of the runtime's own, which a BigQuery connection must never use.

    GoogleCloud reads its fields from GOOGLE_CLOUD_* variables, and google-auth looks up Application Default
    Credentials, which here fail the test.
    """
    monkeypatch.setenv("GOOGLE_CLOUD_PRIVATE_KEY", private_key)
    monkeypatch.setenv("GOOGLE_CLOUD_CLIENT_EMAIL", "runtime@platform.iam.gserviceaccount.com")
    monkeypatch.setenv("GOOGLE_CLOUD_TOKEN_URI", TOKEN_URI)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "platform-project")
    adc = MagicMock(side_effect=AssertionError("Application Default Credentials were looked up"))
    monkeypatch.setattr("google.auth.default", adc)
    return adc


def test_the_platform_json_builds_the_connection(service_account_key):
    payload = {
        "project": "analytics-prod",
        "location": "europe-west2",
        "maximum_bytes_billed": 1024**3,
        "service_account_json": service_account_key | {"impersonated_user_email": "someone@example.com"},
    }

    connection = BigQuery(id="bigquery-conn", **payload)

    assert connection.type == "dynamiq.connections.BigQuery"
    assert connection.model_dump(exclude={"id", "type"}) == payload | {"service_account_json": service_account_key}


def test_connects_as_the_service_account_in_its_key(service_account_key, ambient_credentials):
    connection = BigQuery(project="analytics-prod", location="EU", service_account_json=service_account_key)

    with patch("google.cloud.bigquery.Client") as client_class:
        client = connection.connect()

    assert client is client_class.return_value
    kwargs = client_class.call_args.kwargs
    assert kwargs["project"] == "analytics-prod"
    assert kwargs["location"] == "EU"
    assert kwargs["credentials"].service_account_email == KEY_EMAIL
    ambient_credentials.assert_not_called()


def test_the_client_never_looks_up_ambient_credentials(service_account_key, ambient_credentials):
    """The real client looks up Application Default Credentials and a default project when it is given none."""
    client = BigQuery(project="analytics-prod", service_account_json=service_account_key).connect()

    assert client.project == "analytics-prod"
    ambient_credentials.assert_not_called()
    client.close()


@pytest.mark.parametrize("missing", ["private_key", "client_email", "token_uri"])
def test_a_key_without_a_required_field_raises_a_connection_error(service_account_key, missing):
    """The GOOGLE_CLOUD_* variables that ambient_credentials sets would complete it, if anything read them."""
    del service_account_key[missing]
    connection = BigQuery(project="analytics-prod", service_account_json=service_account_key)

    with pytest.raises(ConnectionError, match=f"service account key has no {missing}"):
        connection.connect()


def test_an_empty_key_names_every_missing_field():
    with pytest.raises(ConnectionError, match="has no private_key, client_email, token_uri"):
        BigQuery(project="analytics-prod", service_account_json={}).connect()


def test_an_invalid_private_key_raises_a_connection_error(service_account_key):
    service_account_key["private_key"] = "-----BEGIN PRIVATE KEY-----\nc2VjcmV0\n-----END PRIVATE KEY-----\n"
    connection = BigQuery(project="analytics-prod", service_account_json=service_account_key)

    with pytest.raises(ConnectionError, match="service account key is not valid") as error:
        connection.connect()
    assert "c2VjcmV0" not in str(error.value)


def test_key_fields_never_come_from_the_environment():
    """The runtime's GOOGLE_CLOUD_* variables fill a GoogleCloud's missing fields, but never a BigQuery key's."""
    runtime_key = GoogleCloud(client_email=KEY_EMAIL)
    assigned = BigQuery(project="analytics-prod", service_account_json={})
    assigned.service_account_json = runtime_key

    for connection in (
        BigQuery(project="analytics-prod", service_account_json={"client_email": KEY_EMAIL}),
        BigQuery(project="analytics-prod", service_account_json=runtime_key),
        assigned,
    ):
        assert connection.service_account_json.client_email == KEY_EMAIL
        assert connection.service_account_json.private_key is None
        assert connection.model_dump()["service_account_json"]["token_uri"] is None


def test_the_key_can_be_the_text_of_its_json_file(service_account_key):
    from_text = BigQuery(project="analytics-prod", service_account_json=json.dumps(service_account_key))
    from_fields = BigQuery(project="analytics-prod", service_account_json=service_account_key)

    assert from_text.model_dump(exclude={"id"}) == from_fields.model_dump(exclude={"id"})


def test_reloading_the_connection_keeps_its_client_cache_key(service_account_key):
    """ConnectionManager keys its clients by the dump, and the runtime builds the connection anew for each run."""
    payload = {"id": "bigquery-conn", "project": "analytics-prod", "service_account_json": service_account_key}
    manager = ConnectionManager()

    first, second = BigQuery(**payload), BigQuery(**payload)
    reloaded = BigQuery(**first.model_dump())

    assert manager.get_connection_id(first) == manager.get_connection_id(second) == manager.get_connection_id(reloaded)


@pytest.mark.parametrize(
    "fields",
    [
        {"project": ""},
        {"maximum_bytes_billed": 0},
        {"maximum_bytes_billed": 10 * 1024**2 - 1},
    ],
    ids=["empty project", "no bytes", "less than BigQuery bills for a table"],
)
def test_invalid_settings_are_rejected(service_account_key, fields):
    with pytest.raises(ValidationError):
        BigQuery(**({"project": "analytics-prod", "service_account_json": service_account_key} | fields))


def test_the_lowest_bytes_limit_is_what_bigquery_bills_for_a_table(service_account_key):
    connection = BigQuery(
        project="analytics-prod", maximum_bytes_billed=10 * 1024**2, service_account_json=service_account_key
    )

    assert connection.maximum_bytes_billed == 10 * 1024**2


def test_a_validation_error_does_not_quote_the_key(service_account_key):
    """Pydantic quotes the start and the end of an input it rejects, here the key's text cut off in its private key."""
    text = json.dumps(service_account_key)
    cut_off = text[: text.index("-----BEGIN PRIVATE KEY-----") + 60]

    with pytest.raises(ValidationError) as error:
        BigQuery(project="analytics-prod", service_account_json=cut_off)

    assert cut_off[-20:] not in str(error.value)
