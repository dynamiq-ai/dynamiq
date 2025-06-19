from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import AWSRedshift, MySQL, PostgreSQL, Snowflake
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_SQL = """## Database Query Tool
### Purpose
Execute SQL queries against relational databases to retrieve, insert, update, or delete data.

### When to Use
- Retrieve specific data from database tables
- Perform data analysis and aggregation queries
- Execute database maintenance operations
- Generate reports from structured data
- Verify data integrity and relationships

### Key Capabilities
- Execute SELECT queries to retrieve data
- Perform INSERT, UPDATE, DELETE operations
- Run complex joins and aggregations
- Execute stored procedures and functions
- Support for PostgreSQL, MySQL, Snowflake, and AWS Redshift
- Automatic result formatting for readability
- Error handling with clear diagnostics

### Required Parameters
- query (string): The SQL statement to execute

### Usage Examples
#### Data Retrieval
```json
{
  "query": "SELECT name, email, created_at FROM users WHERE status = 'active' LIMIT 10"
}
```

#### Data Aggregation
```json
{
  "query": "SELECT department, COUNT(*) as employee_count, AVG(salary) as avg_salary FROM employees GROUP BY department"
}
```

#### Data Modification
```json
{
  "query": "UPDATE products SET price = price * 1.1 WHERE category = 'electronics'"
}
```

### Best Practices
1. Use specific column names instead of SELECT * for better performance
2. Include LIMIT clauses for large datasets to avoid overwhelming results
3. Use parameterized queries when possible to prevent SQL injection
4. Test queries carefully as modifications cannot be undone
5. Use transactions for multiple related operations
6. Check query execution time for performance optimization
"""  # noqa: E501


class SQLInputSchema(BaseModel):
    query: str | None = Field(None, description="Parameter to provide a query that needs to be executed.")


class SQLExecutor(ConnectionNode):
    """
    A tool for SQL query execution.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        connection (PostgreSQL|MySQL|Snowflake|AWSRedshift): The connection instance for the specified storage.
        query (Optional[str]): The SQL statement to execute.
        input_schema (SQLInputSchema): The input schema for the tool.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SQL Executor Tool"
    description: str = DESCRIPTION_SQL
    connection: PostgreSQL | MySQL | Snowflake | AWSRedshift
    query: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[SQLInputSchema]] = SQLInputSchema

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

    def execute(self, input_data, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.query or self.query
        try:
            if not query:
                raise ValueError("Query cannot be empty")
            cursor = self.client.cursor(
                self.connection.cursor_params if not isinstance(self.connection, (PostgreSQL, AWSRedshift)) else {}
            )
            cursor.execute(query)
            output = cursor.fetchall() if cursor.description is not None else []
            cursor.close()
            if self.is_optimized_for_agents:
                output = self.format_results(output, query)
            return {"content": output}
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool {self.name} failed to execute query {query}. Error: {e}", recoverable=True
            )
