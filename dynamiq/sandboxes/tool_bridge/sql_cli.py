from datetime import timezone
from typing import Any

from google.auth.transport.requests import Request

from dynamiq.connections import AWSRedshift, BigQuery, DatabricksSQL, MySQL, PostgreSQL, Snowflake
from dynamiq.nodes.tools.sql_executor import BIGQUERY_DEFAULT_MAXIMUM_BYTES_BILLED

BIGQUERY_SCOPES = ["https://www.googleapis.com/auth/bigquery"]
CONNECT_TIMEOUT_SECONDS = 10

SQL_CLI_ENGINES: dict[type, str] = {
    PostgreSQL: "postgres",
    AWSRedshift: "redshift",
    MySQL: "mysql",
    Snowflake: "snowflake",
    DatabricksSQL: "databricks",
    BigQuery: "bigquery",
}


def sql_cli_engine(connection: Any) -> str | None:
    """The sql-cli engine that serves ``connection``, or None for a connection sql-cli cannot use."""
    return SQL_CLI_ENGINES.get(type(connection))


def sql_cli_connection(connection: Any) -> dict[str, Any]:
    """The connection payload sql-cli expects for ``connection``, as the platform serves it.

    TLS is never weaker than on the agent's own connection. BigQuery gets an access token that expires within the
    hour, minted from the service account key, which stays out of the sandbox, and the byte limit of the agent's tool,
    which sql-cli applies to its own jobs but the token does not enforce.

    Raises:
        ValueError: If sql-cli cannot use the connection.
        ConnectionError: If a BigQuery key is not valid or Google refuses it a token.
    """
    if isinstance(connection, (PostgreSQL, AWSRedshift)):
        return {
            "host": connection.host,
            "port": int(connection.port),
            "database": connection.database,
            "user": connection.user,
            "password": connection.password,
            "connect_timeout": CONNECT_TIMEOUT_SECONDS,
            # libpq's default, as on the agent's own connection; sql-cli's own default would refuse a server without
            # TLS that the agent's SQL tool reaches.
            "ssl": {"mode": "prefer"},
        }
    if isinstance(connection, MySQL):
        return {
            "host": connection.host,
            "port": int(connection.port),
            "database": connection.database,
            "user": connection.user,
            "password": connection.password,
            "connect_timeout": CONNECT_TIMEOUT_SECONDS,
            # sql-cli has no opportunistic TLS, which the agent's mysql-connector uses, so it is required. The
            # certificate must chain to a CA the sandbox trusts, which a private one such as RDS's does not; its host
            # name is not checked, as mysql-connector does not check it either.
            "ssl": {"verify_identity": False},
        }
    if isinstance(connection, Snowflake):
        return {
            "account": connection.account,
            "user": connection.user,
            "password": connection.password,
            "warehouse": connection.warehouse,
            "database": connection.database,
            "schema": connection.snowflake_schema,
        }
    if isinstance(connection, DatabricksSQL):
        return {
            "server_hostname": connection.server_hostname,
            "http_path": connection.http_path,
            "access_token": connection.access_token,
        }
    if isinstance(connection, BigQuery):
        credentials = connection.get_credentials(scopes=BIGQUERY_SCOPES)
        try:
            credentials.refresh(Request())
        except Exception as error:
            message = f"Google refused the BigQuery connection a token ({type(error).__name__})."
            raise ConnectionError(message) from error
        return {
            "project": connection.project,
            "location": connection.location,
            "maximum_bytes_billed": connection.maximum_bytes_billed or BIGQUERY_DEFAULT_MAXIMUM_BYTES_BILLED,
            "access_token": credentials.token,
            "expires_at": credentials.expiry.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    raise ValueError(f"sql-cli cannot use a {type(connection).__name__} connection.")
