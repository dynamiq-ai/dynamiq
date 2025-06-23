import pytest
from databricks.sql.types import Row

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
def mock_fetchall_databricks_sql_response():
    """Mock response from databricks select requests"""
    return [
        Row(id=0),
        Row(id=1),
        Row(id=2),
        Row(id=3),
        Row(id=4),
        Row(id=5),
        Row(id=6),
        Row(id=7),
        Row(id=8),
        Row(id=9),
    ]


@pytest.fixture
def mock_databricks_sql_response():
    """Mock response from databricks select requests"""
    return [
        {"id": 0},
        {"id": 1},
        {"id": 2},
        {"id": 3},
        {"id": 4},
        {"id": 5},
        {"id": 6},
        {"id": 7},
        {"id": 8},
        {"id": 9},
    ]


@pytest.fixture
def mock_fetchall_sql_response_for_agents():
    """Mock response from select requests"""
    return (
        "Row 1\nDescription: Row5Description\nName: Row5Name\n\nRow 2\nDescription: Row6Description"
        "\nName: Row6Name\n\nRow 3\nDescription: Row5Description\nName: Row5Name\n\nRow 4"
        "\nDescription: Row6Description\nName: Row6Name"
    )


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
def test_select_execute(mock_fetchall_sql_response, connection, mock_cursor_with_select):
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
def mock_databricks_cursor_with_select(mocker, mock_fetchall_sql_response, mock_fetchall_databricks_sql_response):
    mock_databricks_cursor = mocker.Mock()
    mock_databricks_cursor.fetchall.return_value = mock_fetchall_databricks_sql_response

    mock_databricks_connection = mocker.Mock()
    mock_databricks_connection.cursor.return_value = mock_databricks_cursor

    mocker.patch("databricks.sql.connect", return_value=mock_databricks_connection)
    return mock_databricks_cursor


def test_databricks_select_execute(
    mock_fetchall_databricks_sql_response, mock_databricks_cursor_with_select, mock_databricks_sql_response
):
    connection = connections.DataBricksSQL(
        server_hostname="server_hostname", http_path="http_path", access_token="access_token"
    )
    sql_tool = SQLExecutor(connection=connection)
    output = mock_databricks_sql_response
    input_data = {"query": """select * from test1"""}

    result = sql_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["query"] == input_data["query"]

    mock_databricks_cursor_with_select.execute.assert_called_once_with(input_data["query"])
    mock_databricks_cursor_with_select.fetchall.assert_called_once()
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
    mocker.patch("databricks.sql.connect", return_value=mock_connection)
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
        connections.DataBricksSQL(
            server_hostname="server_hostname", http_path="http_path", access_token="access_token"
        ),
    ],
)
def test_non_select_queries_execution(connection, mock_cursor_with_none_description):
    sql_tool = SQLExecutor(connection=connection)
    output = []
    input_data = {"query": """delete * from test1"""}

    result = sql_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["query"] == input_data["query"]

    mock_cursor_with_none_description.execute.assert_called_once_with(input_data["query"])
    output_dump = result.output
    assert output_dump["content"] == output


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
def test_non_select_queries_execution_for_agents(connection, mock_cursor_with_none_description):
    sql_tool = SQLExecutor(connection=connection, is_optimized_for_agents=True)
    input_data = {"query": """delete * from test1"""}
    output = f'Query "{input_data["query"]}" executed successfully. No results returned.'

    result = sql_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["query"] == input_data["query"]

    mock_cursor_with_none_description.execute.assert_called_once_with(input_data["query"])
    output_dump = result.output
    assert output_dump["content"] == output


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
def test_select_execute_for_agents(mock_fetchall_sql_response_for_agents, connection, mock_cursor_with_select):
    sql_tool = SQLExecutor(connection=connection, is_optimized_for_agents=True)
    output = mock_fetchall_sql_response_for_agents
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
