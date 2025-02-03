import pytest

from dynamiq.connections import connections
from dynamiq.nodes.tools.sql_executor import SQLExecutor
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.fixture
def mock_fetchall_sql_response():
    """Mock response from select requests"""
    return [
        {"Description": "Row5Description", "Name": "Row5Name"},
        {"Description": "Row6Description", "Name": "Row6Name"},
        {"Description": "Row5Description", "Name": "Row5Name"},
        {"Description": "Row6Description", "Name": "Row6Name"},
    ]


@pytest.fixture
def mock_cursor_with_select(mocker, mock_fetchall_sql_response):
    mock_cursor = mocker.Mock()
    mock_cursor.fetchall.return_value = mock_fetchall_sql_response

    mock_connection = mocker.Mock()
    mock_connection.cursor.return_value = mock_cursor

    mocker.patch("mysql.connector.connect", return_value=mock_connection)
    mocker.patch("psycopg.connect", return_value=mock_connection)
    mocker.patch("snowflake.connector.connect", return_value=mock_connection)
    return mock_cursor


@pytest.mark.parametrize(
    "connection",
    [
        connections.PostgreSQL(host="test_host", port=5432, database="db", user="user", password="password"),
        connections.MySQL(host="test_host", database="db", user="user", password="password"),
        connections.Snowflake(
            user="user", password="password", database="db", account="account", warehouse="warehouse", schema="schema"
        ),
        connections.AWSRedshift(user="user", host="test_host", port=5439, database="db", password="password"),
    ],
)
def test_mysql_postgres_select_execute(mock_fetchall_sql_response, connection, mock_cursor_with_select):
    sql_tool = SQLExecutor(connection=connection)
    output = mock_fetchall_sql_response
    input_data = {"query": """select * from test1"""}

    result = sql_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["query"] == input_data["query"]

    mock_cursor_with_select.execute.assert_called_once_with(input_data["query"])
    mock_cursor_with_select.fetchall.assert_called_once()
    output_dump = result.output
    assert output_dump["content"] == output


@pytest.fixture
def mock_cursor_with_none_description(mocker):
    mock_cursor = mocker.Mock()
    mock_cursor.description = None

    mock_connection = mocker.Mock()
    mock_connection.cursor.return_value = mock_cursor

    mocker.patch("mysql.connector.connect", return_value=mock_connection)
    mocker.patch("psycopg.connect", return_value=mock_connection)
    mocker.patch("snowflake.connector.connect", return_value=mock_connection)
    return mock_cursor


@pytest.mark.parametrize(
    "connection",
    [
        connections.PostgreSQL(host="test_host", port=5432, database="db", user="user", password="password"),
        connections.MySQL(host="test_host", database="db", user="user", password="password"),
        connections.Snowflake(
            user="user", password="password", database="db", account="account", warehouse="warehouse", schema="schema"
        ),
        connections.AWSRedshift(user="user", host="test_host", port=5439, database="db", password="password"),
    ],
)
def test_non_select_queries_execution(mock_fetchall_sql_response, connection, mock_cursor_with_none_description):
    sql_tool = SQLExecutor(connection=connection)
    output = []
    input_data = {"query": """select * from test1"""}

    result = sql_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["query"] == input_data["query"]

    mock_cursor_with_none_description.execute.assert_called_once_with(input_data["query"])
    output_dump = result.output
    assert output_dump["content"] == output
