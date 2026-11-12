import concurrent.futures
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
import yaml
from google.api_core.exceptions import BadRequest
from google.cloud import bigquery
from google.cloud.bigquery import ScalarQueryParameter, SchemaField
from google.cloud.bigquery.table import Row

from dynamiq import Workflow
from dynamiq.connections import BigQuery, PostgreSQL
from dynamiq.flows import Flow
from dynamiq.nodes.tools.sql_executor import (
    BIGQUERY_DEFAULT_MAXIMUM_BYTES_BILLED,
    DESCRIPTION_BIGQUERY,
    DESCRIPTION_BIGQUERY_WRITES_ALLOWED,
    DESCRIPTION_SQL,
    SQLExecutor,
)
from dynamiq.runnables import RunnableConfig, RunnableStatus
from tests.helpers.agent_hooks import ALL_MODES, build_agent, final_step, run_scripted, tool_step

QUERY = "SELECT status, COUNT(*) AS orders FROM `shop.sales.orders` GROUP BY status"
SCHEMA = [SchemaField("status", "STRING"), SchemaField("orders", "INTEGER")]
RECORDS = [("paid", 3), ("refunded", 1)]


class Rows(list):
    """What job.result() returns: the rows fetched, with the result's schema and its total number of rows."""

    def __init__(self, rows, schema, total_rows):
        super().__init__(rows)
        self.schema = schema
        self.total_rows = total_rows


def make_rows(schema, records):
    field_to_index = {field.name: position for position, field in enumerate(schema)}
    return [Row(record, field_to_index) for record in records]


def answer(client, *, statement_type="SELECT", bytes_processed=2048, schema=SCHEMA, records=RECORDS):
    """Make the client answer the dry run with the statement type and the estimate, and the query with the rows."""
    rows = make_rows(schema, records)
    dry_run = MagicMock(statement_type=statement_type, total_bytes_processed=bytes_processed)
    job = MagicMock(job_id="job-1")
    job.result.side_effect = lambda timeout=None, max_results=None: Rows(rows[:max_results], schema, len(rows))
    client.query.side_effect = lambda query, job_config=None, **kwargs: dry_run if job_config.dry_run else job
    return job


def job_configs(client):
    """The job config of each query the tool sent: the dry run's, then the query's."""
    return [sent.kwargs["job_config"] for sent in client.query.call_args_list]


@pytest.fixture
def client(mocker):
    """The client that BigQuery.connect builds, with no network behind it."""
    client = mocker.create_autospec(bigquery.Client, instance=True)
    mocker.patch("google.cloud.bigquery.Client", return_value=client)
    mocker.patch("google.oauth2.service_account.Credentials.from_service_account_info")
    return client


@pytest.fixture
def connection():
    return BigQuery(
        id="bigquery-conn",
        project="analytics-prod",
        location="EU",
        service_account_json={
            "private_key": "-----BEGIN PRIVATE KEY-----\nkey\n-----END PRIVATE KEY-----\n",
            "client_email": "agent@sa-project.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token",
        },
    )


def test_a_select_runs_with_the_default_limits(client, connection):
    job = answer(client)
    tool = SQLExecutor(id="orders-tool", connection=connection)

    result = tool.run({"query": QUERY}, RunnableConfig(run_id="0b5e6a3c-run"))

    assert result.status == RunnableStatus.SUCCESS
    dry_run, query = job_configs(client)
    assert (dry_run.dry_run, dry_run.use_query_cache) == (True, False)
    assert query.dry_run is not True
    assert query.maximum_bytes_billed == BIGQUERY_DEFAULT_MAXIMUM_BYTES_BILLED == 10 * 1024**3
    assert int(query.job_timeout_ms) == 60_000
    assert query.labels == {"dynamiq_node": "orders-tool", "dynamiq_run": "0b5e6a3c-run"}
    job.result.assert_called_once_with(timeout=60, max_results=1000)


def test_the_connection_sets_the_bytes_limit(client, connection):
    answer(client)
    connection = connection.model_copy(update={"maximum_bytes_billed": 1024**3})

    SQLExecutor(connection=connection).run({"query": QUERY})

    assert job_configs(client)[-1].maximum_bytes_billed == 1024**3


def test_node_options_set_the_timeout_and_the_row_limit(client, connection):
    job = answer(client)

    SQLExecutor(connection=connection, max_rows=5, query_timeout_seconds=12.5).run({"query": QUERY})

    assert int(job_configs(client)[-1].job_timeout_ms) == 12_500
    job.result.assert_called_once_with(timeout=12.5, max_results=5)


def test_job_labels_keep_to_bigquery_rules(client, connection):
    answer(client)
    tool = SQLExecutor(id="Orders Tool/EU", connection=connection)

    tool.run({"query": QUERY}, RunnableConfig(run_id="Run:" + "x" * 70))

    assert job_configs(client)[-1].labels == {"dynamiq_node": "orders_tool_eu", "dynamiq_run": "run_" + "x" * 59}


@pytest.mark.parametrize(
    "statement_type",
    ["INSERT", "UPDATE", "DELETE", "MERGE", "CREATE_TABLE", "DROP_TABLE", "SCRIPT", "EXPORT_DATA", "CALL"],
)
def test_statements_other_than_select_are_refused_after_the_dry_run(client, connection, statement_type):
    answer(client, statement_type=statement_type)

    result = SQLExecutor(connection=connection).run({"query": "DELETE FROM `shop.sales.orders` WHERE TRUE"})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.message == f"BigQuery: only SELECT queries are allowed (got {statement_type})"
    assert result.error.recoverable
    assert len(job_configs(client)) == 1  # the dry run, and no query job


def test_writes_allowed_runs_other_statements(client, connection):
    answer(client, statement_type="INSERT", records=[])

    result = SQLExecutor(connection=connection, writes_allowed=True).run(
        {"query": "INSERT INTO `shop.sales.orders` (status) VALUES ('paid')"}
    )

    assert result.status == RunnableStatus.SUCCESS
    assert len(job_configs(client)) == 2


def test_a_query_over_the_bytes_limit_is_refused_before_it_runs(client, connection):
    answer(client, bytes_processed=12_345_678_901)

    result = SQLExecutor(connection=connection).run({"query": QUERY})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.message.startswith(
        "BigQuery: the query would process 12.346 GB, more than the 10.737 GB limit."
    )
    assert len(job_configs(client)) == 1


def test_the_connection_bytes_limit_also_refuses_a_query(client, connection):
    answer(client, bytes_processed=2 * 1024**3)
    connection = connection.model_copy(update={"maximum_bytes_billed": 1024**3})

    result = SQLExecutor(connection=connection).run({"query": QUERY})

    assert "more than the 1.074 GB limit" in result.error.message
    assert len(job_configs(client)) == 1


def test_cells_become_json_safe(client, connection):
    schema = [
        SchemaField("amount", "NUMERIC"),
        SchemaField("units", "NUMERIC"),
        SchemaField("day", "DATE"),
        SchemaField("at", "TIMESTAMP"),
        SchemaField("local", "DATETIME"),
        SchemaField("clock", "TIME"),
        SchemaField("blob", "BYTES"),
        SchemaField("big", "INTEGER"),
        SchemaField("largest_exact", "INTEGER"),
        SchemaField("ratio", "FLOAT"),
        SchemaField("tags", "STRING", mode="REPEATED"),
        SchemaField("item", "RECORD", fields=[SchemaField("price", "NUMERIC"), SchemaField("seen", "DATE")]),
        SchemaField("payload", "JSON"),
    ]
    record = (
        Decimal("12.50"),
        Decimal("3"),
        date(2026, 11, 13),
        datetime(2026, 11, 13, 9, 30, tzinfo=timezone.utc),
        datetime(2026, 11, 13, 9, 30),
        datetime(2026, 11, 13, 9, 30, 15).time(),
        b"\x00\xff",
        2**53 + 1,
        2**53,
        float("nan"),
        ["a", "b"],
        {"price": Decimal("1.25"), "seen": date(2026, 1, 2)},
        {"ids": [2**60], "ok": True},
    )
    answer(client, schema=schema, records=[record])

    result = SQLExecutor(connection=connection).run({"query": QUERY})

    assert json.loads(json.dumps(result.output, allow_nan=False))["content"] == [
        {
            "amount": 12.5,
            "units": 3,
            "day": "2026-11-13",
            "at": "2026-11-13T09:30:00+00:00",
            "local": "2026-11-13T09:30:00",
            "clock": "09:30:15",
            "blob": "AP8=",
            "big": "9007199254740993",
            "largest_exact": 9007199254740992,
            "ratio": "NaN",
            "tags": ["a", "b"],
            "item": {"price": 1.25, "seen": "2026-01-02"},
            "payload": {"ids": ["1152921504606846976"], "ok": True},
        }
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("0.1"), 0.1),
        (Decimal("1.50"), 1.5),
        (Decimal("12345678901234567.89"), "12345678901234567.89"),
        (Decimal("1234567890123.123456789"), "1234567890123.123456789"),
    ],
)
def test_decimals_a_float_would_round_come_back_as_strings(client, connection, value, expected):
    answer(client, schema=[SchemaField("amount", "BIGNUMERIC")], records=[(value,)])

    result = SQLExecutor(connection=connection).run({"query": QUERY})

    assert result.output["content"] == [{"amount": expected}]


def test_the_output_describes_the_result(client, connection):
    schema = [*SCHEMA, SchemaField("tags", "STRING", mode="REPEATED")]
    answer(client, schema=schema, records=[("paid", 3, ["eu"])])

    output = SQLExecutor(connection=connection).run({"query": QUERY}).output

    assert output == {
        "content": [{"status": "paid", "orders": 3, "tags": ["eu"]}],
        "columns": [
            {"name": "status", "type": "STRING"},
            {"name": "orders", "type": "INTEGER"},
            {"name": "tags", "type": "ARRAY<STRING>"},
        ],
        "truncated": False,
        "total_rows": 1,
    }


def test_a_result_cut_at_max_rows_says_so(client, connection):
    answer(client, records=[(f"status-{number}", number) for number in range(5)])

    output = SQLExecutor(connection=connection, max_rows=2).run({"query": QUERY}).output
    agent_output = SQLExecutor(connection=connection, max_rows=2, is_optimized_for_agents=True).run({"query": QUERY})

    assert output["content"] == [{"status": "status-0", "orders": 0}, {"status": "status-1", "orders": 1}]
    assert (output["truncated"], output["total_rows"]) == (True, 5)
    assert agent_output.output["content"].endswith(
        "\n\nShowing the first 2 of 5 rows. To cover the rest, aggregate or filter in the query."
    )


@pytest.mark.parametrize("is_optimized_for_agents", [False, True], ids=["workflow", "agent"])
def test_the_content_is_what_the_other_engines_return(client, connection, mocker, is_optimized_for_agents):
    answer(client)
    cursor = mocker.Mock()
    cursor.fetchall.return_value = [{"status": "paid", "orders": 3}, {"status": "refunded", "orders": 1}]
    mocker.patch("psycopg.connect", return_value=mocker.Mock(**{"cursor.return_value": cursor}))
    postgres = PostgreSQL(host="host", port=5432, database="db", user="user", password="password")

    bigquery_output = SQLExecutor(connection=connection, is_optimized_for_agents=is_optimized_for_agents).run(
        {"query": QUERY}
    )
    postgres_output = SQLExecutor(connection=postgres, is_optimized_for_agents=is_optimized_for_agents).run(
        {"query": QUERY}
    )

    assert bigquery_output.output["content"] == postgres_output.output["content"]
    if is_optimized_for_agents:
        assert bigquery_output.output.keys() == postgres_output.output.keys() == {"content"}


def test_dict_parameters_are_bound_by_name(client, connection):
    answer(client)
    parameters = {
        "country": "DE",
        "min_orders": 5,
        "share": 0.5,
        "active": True,
        "since": date(2026, 1, 1),
        "after": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "amount": Decimal("1.50"),
        "region": None,
    }

    SQLExecutor(connection=connection).run({"query": QUERY, "parameters": parameters})

    expected = [
        ScalarQueryParameter("country", "STRING", "DE"),
        ScalarQueryParameter("min_orders", "INT64", 5),
        ScalarQueryParameter("share", "FLOAT64", 0.5),
        ScalarQueryParameter("active", "BOOL", True),
        ScalarQueryParameter("since", "DATE", date(2026, 1, 1)),
        ScalarQueryParameter("after", "TIMESTAMP", datetime(2026, 1, 1, tzinfo=timezone.utc)),
        ScalarQueryParameter("amount", "BIGNUMERIC", Decimal("1.50")),
        ScalarQueryParameter("region", "STRING", None),
    ]
    assert [config.query_parameters for config in job_configs(client)] == [expected, expected]


def test_list_parameters_are_bound_by_position(client, connection):
    answer(client)

    SQLExecutor(connection=connection).run({"query": QUERY, "parameters": ["DE", 5]})

    expected = [ScalarQueryParameter(None, "STRING", "DE"), ScalarQueryParameter(None, "INT64", 5)]
    assert [config.query_parameters for config in job_configs(client)] == [expected, expected]


def test_a_parameter_without_a_bigquery_type_is_refused(client, connection):
    answer(client)

    result = SQLExecutor(connection=connection).run({"query": QUERY, "parameters": {"ids": [1, 2]}})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.message.startswith("BigQuery: parameter 'ids' is a list, which cannot be bound.")
    assert job_configs(client) == []


def test_an_empty_query_is_refused(client, connection):
    result = SQLExecutor(connection=connection).run({"query": ""})

    assert result.error.message == "BigQuery: the query is empty."
    assert job_configs(client) == []


def test_a_query_that_runs_too_long_is_cancelled(client, connection):
    job = answer(client)
    job.result.side_effect = concurrent.futures.TimeoutError()

    result = SQLExecutor(connection=connection, query_timeout_seconds=5).run({"query": QUERY})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.message.startswith("BigQuery: the query did not finish within 5 seconds and was cancelled.")
    job.cancel.assert_called_once()


def test_an_agent_gets_bigquery_s_own_error_message(client, connection):
    client.query.side_effect = BadRequest(
        "POST https://bigquery.googleapis.com/bigquery/v2/projects/analytics-prod/jobs: Syntax error at [1:1]",
        errors=[{"message": "Syntax error: Unexpected identifier at [1:1]", "reason": "invalidQuery"}],
    )

    result = SQLExecutor(connection=connection).run({"query": "SELEC 1"})

    assert result.error.message == "BigQuery: Syntax error: Unexpected identifier at [1:1]"
    assert result.error.recoverable


def test_queries_on_a_shared_client_run_at_once(client, connection):
    """The BigQuery client is thread-safe, so it does not take the lock that DB-API connections need."""
    job = answer(client)
    fetch = job.result.side_effect
    active = most_active = 0
    counter_lock = threading.Lock()

    def slow_fetch(**kwargs):
        nonlocal active, most_active
        with counter_lock:
            active += 1
            most_active = max(most_active, active)
        time.sleep(0.05)
        with counter_lock:
            active -= 1
        return fetch(**kwargs)

    job.result.side_effect = slow_fetch
    tool = SQLExecutor(connection=connection)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: tool.run({"query": QUERY}), range(8)))

    assert all(result.status == RunnableStatus.SUCCESS for result in results)
    assert most_active > 1


def test_the_description_tells_an_agent_bigquery_s_rules(client, connection):
    description = SQLExecutor(connection=connection).description

    assert description == DESCRIPTION_BIGQUERY
    for rule in ("BigQuery Standard SQL", "Only SELECT statements run", "GROUP BY", "LIMIT", "INFORMATION_SCHEMA"):
        assert rule in description
    assert SQLExecutor(connection=connection, writes_allowed=True).description == DESCRIPTION_BIGQUERY_WRITES_ALLOWED
    assert SQLExecutor(connection=connection, description="Query the shop.").description == "Query the shop."


@pytest.mark.parametrize("mode", ALL_MODES)
def test_an_agent_learns_the_rules_and_moves_on_from_a_refusal(client, connection, mode):
    answer(client)
    answer_select = client.query.side_effect

    def query(sql, job_config=None, **kwargs):
        if job_config.dry_run and sql.startswith("DELETE"):
            return MagicMock(statement_type="DELETE", total_bytes_processed=0)
        return answer_select(sql, job_config=job_config, **kwargs)

    client.query.side_effect = query
    tool = SQLExecutor(connection=connection)
    agent = build_agent(tools=[tool])
    name = agent.sanitize_tool_name(tool.name)
    steps = [tool_step(name, query="DELETE FROM `shop.sales.orders` WHERE TRUE"), tool_step(name, query=QUERY)]

    result, recorder = run_scripted(agent, mode, [*steps, final_step("3 orders are paid")])

    assert result.status == RunnableStatus.SUCCESS
    assert "BigQuery Standard SQL" in recorder.sent_text(0)
    assert "BigQuery: only SELECT queries are allowed (got DELETE)" in recorder.sent_text(1)
    assert "status: paid" in recorder.sent_text(2)


def test_the_other_engines_keep_their_description(mocker):
    mocker.patch("psycopg.connect")
    postgres = PostgreSQL(host="host", port=5432, database="db", user="user", password="password")

    assert SQLExecutor(connection=postgres).description == DESCRIPTION_SQL


def test_yaml_roundtrip(client, connection, tmp_path):
    connection = connection.model_copy(update={"maximum_bytes_billed": 1024**3})
    node = SQLExecutor(
        id="bigquery-node", connection=connection, max_rows=50, query_timeout_seconds=30, writes_allowed=True
    )
    workflow = Workflow(id="bigquery-workflow", flow=Flow(id="bigquery-flow", nodes=[node]))

    yaml_path = tmp_path / "bigquery_workflow.yaml"
    workflow.to_yaml_file(yaml_path)
    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
    roundtrip_path = tmp_path / "bigquery_workflow_roundtrip.yaml"
    loaded.to_yaml_file(roundtrip_path)
    roundtrip_node = Workflow.from_yaml_file(str(roundtrip_path), init_components=True).flow.nodes[0]

    stored, restored = (yaml.safe_load(path.read_text())["connections"] for path in (yaml_path, roundtrip_path))
    assert stored == restored
    assert stored["bigquery-conn"]["service_account_json"] == connection.model_dump()["service_account_json"]
    assert roundtrip_node.client is client
    assert roundtrip_node.connection.model_dump() == connection.model_dump()
    assert roundtrip_node.max_rows == 50
    assert roundtrip_node.query_timeout_seconds == 30
    assert roundtrip_node.writes_allowed is True
    assert roundtrip_node.description == DESCRIPTION_BIGQUERY_WRITES_ALLOWED
