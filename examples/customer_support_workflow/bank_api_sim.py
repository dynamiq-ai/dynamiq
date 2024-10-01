import os
from typing import Any, Literal

from pydantic import ConfigDict

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.node import ConnectionNode
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger

# Define system prompt as a constant
BANK_API_SYSTEM_PROMPT = """
You are an Internal Bank System Database.
You can receive different types of requests, and you can respond to them.
You know everything about the clients, customers, transactions, and accounts.
You should respond like it is a request to a real bank system API.
You should always return the JSON response, without any extra wording and tickmarks like ``` and ```json.
Example of answers can be:
{"account_id": 0101,
"balance": 1000$,
"last_transaction_id": 1010201,
"last_transaction_date": "2024-05-01",
}
Begin:
"""


class BankApiSim(ConnectionNode):
    """
    A simulation of an internal bank API using OpenAI's language model.

    This class provides a tool that simulates access to an internal bank system database.
    It uses OpenAI's GPT model to generate responses to queries about account information,
    transactions, and other banking-related data.
    """

    name: str = "InternalBankAPI"
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    description: str = (
        "A tool with access to the Internal Bank System Database. "
        "Input should be a dictionary with a key 'input' containing the information."
    )
    connection: OpenAIConnection | None = None
    llm: OpenAI | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        # If no client or connection is provided, create a default OpenAIConnection
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ) -> None:
        super().init_components(connection_manager)
        if self.llm is None:
            self.llm = OpenAI(
                connection=self.connection,
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
            )

    def _query_bank_api_sim(self, input_text: str) -> str:
        """
        Generate a simulated bank API response using the OpenAI language model.

        Args:
            input_text (str): The input query or request to the simulated bank API.

        Returns:
            str: A JSON-formatted string containing the simulated API response.

        Raises:
            ValueError: If the LLM fails to generate a result.
            TimeoutError: If the LLM call times out.
        """
        try:
            llm_result = self.llm.run(
                input_data={},
                prompt=Prompt(
                    messages=[
                        Message(content=BANK_API_SYSTEM_PROMPT, role="system"),
                        Message(content=input_text, role="user"),
                    ]
                ),
            )
        except TimeoutError:
            logger.error("LLM call timed out")
            raise

        if llm_result.status != RunnableStatus.SUCCESS:
            raise ValueError("Failed to get LLM result")

        output = llm_result.output["content"]
        logger.info(f"Input: {input_text}")
        logger.info(f"Output: {output}")
        return output

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the bank API simulation with the given input.

        Args:
            input_data (Dict[str, Any]): A dictionary containing the input query under the key 'input'.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the simulated API response under the key 'content',
                            or an error message under the key 'error' if an exception occurs.
        """

        if input_text := input_data.get("input"):
            try:
                result = self._query_bank_api_sim(str(input_text))
            except Exception as e:
                logger.error(f"Error executing BankApiSim: {str(e)}")
                return {"error": str(e)}
            return {"content": result}
        raise ToolExecutionException("Input data must contain an 'input' key with information.", recoverable=True)


if __name__ == "__main__":
    tool = BankApiSim()
    query = "Get the balance of account 0101"
    result = tool.run({"input": query})
    print(f"Results for query '{query}':\n{result.output['content']}")
