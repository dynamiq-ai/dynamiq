import sqlite3
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import BaseConnection, ConnectionType
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class SQLiteConnection(BaseConnection):
    """
    A connection class for SQLite database.

    Attributes:
        db_path (str): Path to the SQLite database file.
    """

    db_path: str
    type: ConnectionType = ConnectionType.PostgreSQL

    def connect(self):
        """
        Establishes a connection to the SQLite database.

        Returns:
            sqlite3.Connection: The connection object.
        """
        return sqlite3.connect(self.db_path)

    @property
    def conn_params(self) -> dict:
        """
        Returns the parameters required for connection.

        Returns:
            dict: A dictionary containing the database path.
        """
        return {"db_path": self.db_path}


class DatabaseQueryInputSchema(BaseModel):
    query: str = Field(..., description="SQL query to execute on the database.")
    params: list[Any] = Field(default_factory=list, description="Parameters for the SQL query.")


class DatabaseQueryTool(ConnectionNode):
    """
    A tool for executing SQL queries on a database.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "ERP Database Query Tool"
    description: str = "Query ERP database to retrieve and manipulate data."
    connection: SQLiteConnection

    input_schema: ClassVar[type[DatabaseQueryInputSchema]] = DatabaseQueryInputSchema

    def execute(
        self,
        input_data: DatabaseQueryInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Executes the SQL query using the provided input data.

        Args:
            input_data (DatabaseQueryInputSchema): The input data containing the SQL query and parameters.
            config (RunnableConfig | None): Optional configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: The result of the SQL query execution.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.query
        params = input_data.params

        connection = None
        cursor = None
        try:
            connection = self.connection.connect()
            cursor = connection.cursor()

            cursor.execute(query, params)

            if query.strip().lower().startswith("select"):
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                results = self._format_results(results, columns)
            else:
                connection.commit()
                results = f"Query executed successfully. {cursor.rowcount} row(s) affected."

        except sqlite3.Error as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to execute query. Error: {e}")
            raise ToolExecutionException(f"Failed to execute query: {e}", recoverable=True)

        finally:
            if cursor is not None:
                cursor.close()

            if connection is not None:
                connection.close()

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(results)[:200]}")

        return {
            "content": results,
        }

    def _format_results(self, results: list[tuple], columns: list[str]) -> str:
        """
        Formats the query results into a readable string format.

        Args:
            results (list[tuple]): The rows returned by the SQL query.
            columns (list[str]): The column names of the query results.

        Returns:
            str: The formatted query results as a string.
        """
        formatted_rows = [", ".join(columns)]
        for row in results:
            formatted_rows.append(", ".join(map(str, row)))
        return "\n".join(formatted_rows)
