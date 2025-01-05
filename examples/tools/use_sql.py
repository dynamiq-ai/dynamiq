from dynamiq.connections import connections
from dynamiq.nodes.tools import SqlExecutor


def basic_requests_snowflake_example():
    snowflake_connection = connections.SnowFlake()

    snowflake_executor = SqlExecutor(connection=snowflake_connection)
    snowflake_insert = {
        "query": """INSERT INTO test1 (Name, Desscription)
        VALUES ('Name1', 'Description1'), ('Name2', 'Description2');"""
    }
    snowflake_select = {"query": """select * from test1"""}
    snowflake_delete = {"query": """DELETE FROM test1 WHERE Name = 'Name1';"""}

    for query in [snowflake_insert, snowflake_select, snowflake_delete]:
        result = snowflake_executor.run(input_data=query)
        print("Query execution results:")
        print(result.output.get("result"))


def basic_requests_mysql_example():
    mysql_connection = connections.MySQL()

    mysql_executor = SqlExecutor(connection=mysql_connection)
    mysql_insert = {
        "query": """
    INSERT INTO test1 (`Name`, `Description`)
    VALUES
        ('Row1Name', 'Row1Description'),
        ('Row2Name', 'Row2Description');"""
    }
    mysql_select = {"query": """select * from test1"""}
    mysql_delete = {"query": """DELETE FROM test1 WHERE `Name` = 'Row1Name';"""}

    for query in [mysql_insert, mysql_select, mysql_delete]:
        result = mysql_executor.run(input_data=query)
        print("Query execution results:")
        print(result.output.get("result"))


if __name__ == "__main__":
    basic_requests_snowflake_example()
    basic_requests_mysql_example()
