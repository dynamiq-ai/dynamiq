import base64
import math
import re
import threading
import weakref
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.connections import AWSRedshift, BigQuery, DatabricksSQL, MySQL, PostgreSQL, Snowflake
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

# Runs share a cached connection, and DB-API connections such as MySQL's crash or corrupt results when several threads
# use one at once, so queries on a connection run one at a time.
_connection_locks: "weakref.WeakKeyDictionary[Any, threading.Lock]" = weakref.WeakKeyDictionary()
_connection_locks_guard = threading.Lock()


def _connection_lock(client: Any) -> threading.Lock:
    with _connection_locks_guard:
        return _connection_locks.setdefault(client, threading.Lock())


BIGQUERY_DEFAULT_MAXIMUM_BYTES_BILLED = 10 * 1024**3
_BIGQUERY_SELECT_STATEMENT_TYPE = "SELECT"
_BIGQUERY_LABEL_MAX_LENGTH = 63
_BYTES_PER_GB = 10**9
# Larger integers lose precision as JSON numbers, which are doubles in JavaScript.
_JSON_EXACT_INTEGER_LIMIT = 2**53

DESCRIPTION_SQL = """Executes SQL queries on multiple database systems including PostgreSQL,
MySQL, Snowflake, and AWS Redshift.

Key Capabilities:
- Full SQL operations: SELECT, INSERT, UPDATE, DELETE, DDL
- Multi-database support with automatic connection handling
- Complex query execution with joins, aggregations, subqueries
- Automatic result formatting and error handling

Usage Strategy:
- Use parameterized queries with placeholders instead of string concatenation
- Use specific columns instead of SELECT * for performance
- Include LIMIT clauses for large datasets
- Leverage database-specific features for optimization
- Handle transactions appropriately for data modifications

Parameter Guide:
- query: SQL statement with placeholders to execute (required)
- parameters: dict or list of values to bind to query placeholders (recommended for security)
- Database connection configured at tool initialization

Examples:
- {"query": "SELECT name, email FROM users WHERE status = %s LIMIT 10", "parameters": ["active"]}
- {"query": "SELECT department, COUNT(*) FROM employees GROUP BY department"}
- {"query": "INSERT INTO products (name, price) VALUES (%s, %s)", "parameters": ["Widget", 29.99]}"""

_DESCRIPTION_BIGQUERY_RULES = """Runs a query on Google BigQuery and returns the result rows.

Rules:
- Write BigQuery Standard SQL (GoogleSQL), and name each table in full, in backticks: `project.dataset.table`.
- A dry run first checks how many bytes the query would read, and a query over the limit is refused: select only
  the columns needed, filter on partition columns, and avoid SELECT *.
- Only the first rows of a large result come back: aggregate (COUNT, SUM, AVG with GROUP BY) and add LIMIT rather than
  reading raw rows.
"""
_DESCRIPTION_BIGQUERY_READ_ONLY = (
    "- Only SELECT statements run. Anything else is refused: INSERT, UPDATE, DELETE, MERGE, DDL, and scripts of\n"
    "  several statements.\n"
)
_DESCRIPTION_BIGQUERY_WRITES = "- Any statement runs, DML, DDL and scripts included.\n"
_DESCRIPTION_BIGQUERY_GUIDE = """
Exploring the data:
- Datasets: SELECT schema_name FROM `project`.`region-us`.INFORMATION_SCHEMA.SCHEMATA (with the datasets' region)
- Tables: SELECT table_name, table_type FROM `project.dataset`.INFORMATION_SCHEMA.TABLES
- Columns: SELECT column_name, data_type FROM `project.dataset`.INFORMATION_SCHEMA.COLUMNS WHERE table_name = 'orders'

Parameter Guide:
- query: the SQL to run (required)
- parameters: values to bind, a dict for @name placeholders or a list for ? placeholders. Bind strings, numbers,
  booleans and nulls; write lists into the query.

Examples:
- {"query": "SELECT status, COUNT(*) AS orders FROM `shop.sales.orders` GROUP BY status ORDER BY orders DESC"}
- {"query": "SELECT name FROM `shop.crm.users` WHERE country = @country LIMIT 50", "parameters": {"country": "DE"}}"""

DESCRIPTION_BIGQUERY = _DESCRIPTION_BIGQUERY_RULES + _DESCRIPTION_BIGQUERY_READ_ONLY + _DESCRIPTION_BIGQUERY_GUIDE
DESCRIPTION_BIGQUERY_WRITES_ALLOWED = (
    _DESCRIPTION_BIGQUERY_RULES + _DESCRIPTION_BIGQUERY_WRITES + _DESCRIPTION_BIGQUERY_GUIDE
)
_DEFAULT_DESCRIPTIONS = (DESCRIPTION_SQL, DESCRIPTION_BIGQUERY, DESCRIPTION_BIGQUERY_WRITES_ALLOWED)


class SQLInputSchema(BaseModel):
    query: str | None = Field(None, description="Parameter to provide a query that needs to be executed.")
    parameters: dict[str, Any] | list[Any] | None = Field(
        None,
        description="Parameters to bind to query placeholders for safe, parameterized execution.",
    )


class SQLExecutor(ConnectionNode):
    """
    A tool for SQL query execution.

    On BigQuery, a query runs only after a dry run shows that it is a SELECT, unless writes_allowed is set, and that it
    reads no more than the connection's maximum_bytes_billed, or 10 GiB when that is unset. The job bills at most that
    much, runs for at most query_timeout_seconds and returns at most max_rows rows.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        connection (PostgreSQL|MySQL|Snowflake|AWSRedshift|DatabricksSQL|BigQuery): The connection instance for the
            specified storage.
        query (Optional[str]): The SQL statement to execute.
        max_rows (int): BigQuery only: the most rows a result returns.
        query_timeout_seconds (float): BigQuery only: how long a query may run.
        writes_allowed (bool): BigQuery only: whether statements other than SELECT may run.
        input_schema (SQLInputSchema): The input schema for the tool.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.DATABASE_QUERY
    name: str = "sql-executor"
    description: str = DESCRIPTION_SQL
    connection: PostgreSQL | MySQL | Snowflake | AWSRedshift | DatabricksSQL | BigQuery
    query: str | None = None
    parameters: dict[str, Any] | list[Any] | None = None
    max_rows: int = Field(
        default=1000,
        gt=0,
        description="BigQuery only: the most rows a result returns. A result cut short says so.",
    )
    query_timeout_seconds: float = Field(
        default=60,
        gt=0,
        description="BigQuery only: how long a query may run before it is cancelled, in seconds.",
    )
    writes_allowed: bool = Field(
        default=False,
        description=(
            "BigQuery only: whether statements other than SELECT may run, such as DML, DDL and scripts. Without it, "
            "the dry run before each query must show a SELECT. The other databases run any statement."
        ),
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[SQLInputSchema]] = SQLInputSchema

    @model_validator(mode="after")
    def _describe_bigquery(self) -> "SQLExecutor":
        """Tell an agent BigQuery's rules, unless the node has a description of its own."""
        if isinstance(self.connection, BigQuery) and self.description in _DEFAULT_DESCRIPTIONS:
            self.description = DESCRIPTION_BIGQUERY_WRITES_ALLOWED if self.writes_allowed else DESCRIPTION_BIGQUERY
        return self

    def format_results(self, results: list[dict[str, Any]], query: str) -> str:
        """Format the retrieved results.

        Args:
            query (str): The executed SQL statement.
            results (list[dict[str,Any]]): List of execution results.

        Returns:
            str: Formatted content of the query result.
        """
        formatted_results = []
        if not results:
            return f'Query "{query}" executed successfully. No results returned.'
        for i, result in enumerate(results):
            formatted_result = f"Row {i + 1}\n"
            formatted_result += "\n".join(f"{key}: {value}" for key, value in result.items())
            formatted_results.append(formatted_result)
        return "\n\n".join(formatted_results)

    def execute(self, input_data: SQLInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:

        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.query or self.query
        params = input_data.parameters if input_data.parameters is not None else self.parameters
        if isinstance(self.connection, BigQuery):
            return self._execute_bigquery(query, params, config)
        try:
            if not query:
                raise ValueError("Query cannot be empty")
            with _connection_lock(self.client):
                cursor = self.client.cursor(
                    **(
                        self.connection.cursor_params
                        if not isinstance(self.connection, (PostgreSQL, AWSRedshift, DatabricksSQL))
                        else {}
                    )
                )
                if params is not None:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                output = cursor.fetchall() if cursor.description is not None else []
                cursor.close()
            if isinstance(self.connection, DatabricksSQL):
                output = [row.asDict(True) for row in output]
            if self.is_optimized_for_agents:
                output = self.format_results(output, query)
            return {"content": output}
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool {self.name} failed to execute query {query}. Error: {e}", recoverable=True
            )

    def _execute_bigquery(
        self, query: str | None, parameters: dict[str, Any] | list[Any] | None, config: RunnableConfig
    ) -> dict[str, Any]:
        """Runs a query on BigQuery once a dry run has checked it, within the byte, time and row limits.

        Agents get the rows as text in `content`; other callers get them as a list, with `columns`, `truncated`
        and `total_rows`. The BigQuery client is thread-safe, so unlike a DB-API connection it takes no lock.

        Raises:
            ToolExecutionException: If the query is refused or fails, with a message an agent can act on.
        """
        from google.cloud.bigquery import QueryJobConfig

        if not query:
            raise ToolExecutionException("BigQuery: the query is empty.", recoverable=True)
        query_parameters = _bigquery_query_parameters(parameters)
        maximum_bytes_billed = self.connection.maximum_bytes_billed or BIGQUERY_DEFAULT_MAXIMUM_BYTES_BILLED

        # A cached result would report no bytes processed.
        dry_run_config = QueryJobConfig(dry_run=True, use_query_cache=False, query_parameters=query_parameters)
        try:
            dry_run = self.client.query(query, job_config=dry_run_config, timeout=self.query_timeout_seconds)
        except Exception as e:
            raise self._bigquery_failure(e) from e
        self._check_bigquery_dry_run(dry_run, maximum_bytes_billed)

        job_config = QueryJobConfig(
            maximum_bytes_billed=maximum_bytes_billed,
            job_timeout_ms=math.ceil(self.query_timeout_seconds * 1000),
            labels=self._bigquery_labels(config),
            query_parameters=query_parameters,
        )
        job = None
        try:
            job = self.client.query(query, job_config=job_config, timeout=self.query_timeout_seconds)
            rows = job.result(timeout=self.query_timeout_seconds, max_results=self.max_rows)
            records = [{name: _json_safe(value) for name, value in row.items()} for row in rows]
        except TimeoutError as e:
            self._cancel_bigquery_job(job)
            raise ToolExecutionException(
                f"BigQuery: the query did not finish within {self.query_timeout_seconds:g} seconds and was cancelled. "
                "Make it read less: filter, select fewer columns or aggregate.",
                recoverable=True,
            ) from e
        except Exception as e:
            raise self._bigquery_failure(e) from e

        total_rows = rows.total_rows if rows.total_rows is not None else len(records)
        truncated = total_rows > len(records)
        if self.is_optimized_for_agents:
            content = self.format_results(records, query)
            if truncated:
                content += (
                    f"\n\nShowing the first {len(records)} of {total_rows} rows. "
                    "To cover the rest, aggregate or filter in the query."
                )
            return {"content": content}
        return {
            "content": records,
            "columns": _bigquery_columns(rows.schema),
            "truncated": truncated,
            "total_rows": total_rows,
        }

    def _check_bigquery_dry_run(self, dry_run: Any, maximum_bytes_billed: int) -> None:
        """Refuse a query that is not a SELECT, unless writes are allowed, or that would read more than the limit.

        Raises:
            ToolExecutionException: If the query is refused.
        """
        statement_type = dry_run.statement_type
        if not self.writes_allowed and statement_type != _BIGQUERY_SELECT_STATEMENT_TYPE:
            logger.info(f"Tool {self.name} - {self.id}: refused a BigQuery {statement_type} statement.")
            raise ToolExecutionException(
                f"BigQuery: only SELECT queries are allowed (got {statement_type or 'an unknown statement'})",
                recoverable=True,
            )
        estimate = dry_run.total_bytes_processed or 0
        if estimate > maximum_bytes_billed:
            logger.info(f"Tool {self.name} - {self.id}: refused a BigQuery query that would process {estimate} bytes.")
            raise ToolExecutionException(
                f"BigQuery: the query would process {estimate / _BYTES_PER_GB:,.3f} GB, more than the "
                f"{maximum_bytes_billed / _BYTES_PER_GB:,.3f} GB limit. Select fewer columns, filter on partition "
                "columns or aggregate.",
                recoverable=True,
            )

    def _bigquery_labels(self, config: RunnableConfig) -> dict[str, str]:
        """Label the job with the node and the run, which BigQuery's job history and billing then show."""
        labels = {"dynamiq_node": self.id, "dynamiq_run": config.run_id}
        return {key: _bigquery_label_value(value) for key, value in labels.items() if value}

    def _bigquery_failure(self, error: Exception) -> ToolExecutionException:
        logger.error(f"Tool {self.name} - {self.id}: BigQuery query failed. Error: {error}")
        return ToolExecutionException(f"BigQuery: {_bigquery_error_message(error)}", recoverable=True)

    def _cancel_bigquery_job(self, job: Any) -> None:
        """Cancel a job the client stopped waiting for: BigQuery's own job timeout may not stop it at once."""
        if job is None:
            return
        try:
            job.cancel(timeout=self.query_timeout_seconds)
        except Exception as e:
            logger.warning(f"Tool {self.name} - {self.id}: failed to cancel BigQuery job {job.job_id}. Error: {e}")


def _bigquery_query_parameters(parameters: dict[str, Any] | list[Any] | None) -> list[Any]:
    """Bind a dict to @name placeholders and a list to ? placeholders, typed by their Python values.

    Raises:
        ToolExecutionException: If a value has no BigQuery scalar type.
    """
    from google.cloud.bigquery import ScalarQueryParameter

    if isinstance(parameters, dict):
        return [
            ScalarQueryParameter(name, _bigquery_parameter_type(f"'{name}'", value), value)
            for name, value in parameters.items()
        ]
    return [
        ScalarQueryParameter(None, _bigquery_parameter_type(str(position), value), value)
        for position, value in enumerate(parameters or [], start=1)
    ]


def _bigquery_parameter_type(parameter: str, value: Any) -> str:
    """The BigQuery type of a parameter's value.

    Raises:
        ToolExecutionException: If the value has no BigQuery scalar type, such as a list or a dict.
    """
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT64"
    if isinstance(value, float):
        return "FLOAT64"
    if isinstance(value, Decimal):
        # BIGNUMERIC holds any NUMERIC value too, and NUMERIC columns compare with it.
        return "BIGNUMERIC"
    # A NULL needs a type as well, and STRING also coerces to the date and time types.
    if isinstance(value, str) or value is None:
        return "STRING"
    if isinstance(value, bytes):
        return "BYTES"
    if isinstance(value, datetime):
        return "DATETIME" if value.tzinfo is None else "TIMESTAMP"
    if isinstance(value, date):
        return "DATE"
    if isinstance(value, time):
        return "TIME"
    raise ToolExecutionException(
        f"BigQuery: parameter {parameter} is a {type(value).__name__}, which cannot be bound. Bind a string, number, "
        "boolean or null, and write lists into the query.",
        recoverable=True,
    )


def _bigquery_label_value(value: Any) -> str:
    """Fit a value to BigQuery's label rules: lowercase letters, digits, underscores and dashes, at most 63."""
    return re.sub(r"[^a-z0-9_-]", "_", str(value).lower())[:_BIGQUERY_LABEL_MAX_LENGTH]


def _bigquery_error_message(error: Exception) -> str:
    """BigQuery's message for a failed request or job, without the request URL and job footer that str() adds."""
    errors = getattr(error, "errors", None)
    if errors and isinstance(errors[0], dict) and errors[0].get("message"):
        return errors[0]["message"]
    return str(error)


def _bigquery_columns(schema: list[Any] | None) -> list[dict[str, str]]:
    """Each result column's name and BigQuery type, with a repeated column's type as ARRAY<type>."""
    return [
        {"name": field.name, "type": f"ARRAY<{field.field_type}>" if field.mode == "REPEATED" else field.field_type}
        for field in schema or []
    ]


def _json_safe(value: Any) -> Any:
    """Converts a BigQuery cell to a value JSON holds exactly.

    Numbers that a JSON number would round (integers beyond 2**53, Decimals a float cannot hold) become strings,
    as do non-finite floats, dates and times (ISO 8601) and bytes (base64). Structs and arrays are converted item
    by item.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value if abs(value) <= _JSON_EXACT_INTEGER_LIMIT else str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, Decimal):
        if not value.is_finite():
            return str(value)
        if value == value.to_integral_value():
            return _json_safe(int(value))
        number = float(value)
        return number if Decimal(repr(number)) == value else str(value)
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # Such as an INTERVAL, which the client reads as a dateutil relativedelta.
    return str(value)
