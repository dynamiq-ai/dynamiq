from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import AWSRedshift, MySQL, PostgreSQL, Snowflake
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class SQLInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a query that needs to be executed.")


class SqlExecutor(ConnectionNode):
    """
    A tool for SQL query execution.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        connection (PostgreSQL|MySQL|Snowflake|AWSRedshift): The connection instance for the specified storage.
        input_schema (SQLInputSchema): The input schema for the tool.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SQL Executor Tool"
    description: str = (
        "A tool for SQL query execution."
        "You can use this tool to execute the query, specified for PostgreSQL, MySQL, Snowflake, AWS Redshift."
    )
    connection: PostgreSQL | MySQL | Snowflake | AWSRedshift

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[SQLInputSchema]] = SQLInputSchema

    def execute(self, input_data, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.query
        try:
            cursor = self.client.cursor(
                **self.connection.cursor_params if not isinstance(self.connection, (PostgreSQL, AWSRedshift)) else {}
            )
            cursor.execute(query)
            output = cursor.fetchall() if cursor.description is not None else []
            cursor.close()
            return {"content": output}
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool {self.name} failed to execute query {query}. Error: {e}", recoverable=True
            )
