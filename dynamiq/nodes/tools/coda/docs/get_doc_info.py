from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCoda


class GetDocInfoInputSchema(BaseModel):
    """
    Input schema for the Get Info About a Doc node.
    """

    doc_id: str = Field(..., description="ID of the  doc to fetch metadata for.")


class GetDocInfo(BaseCoda):
    """
    Node to fetch metadata for a specific  doc.
    """

    name: str = "GetDocInfo"
    description: str = "Fetch metadata for a specific  doc."
    input_schema: ClassVar[type[GetDocInfoInputSchema]] = GetDocInfoInputSchema

    def execute(self, input_data: GetDocInfoInputSchema, config: RunnableConfig | None = None, **kwargs):
        """
        Execute the node to fetch doc metadata.

        Args:
            input_data (GetDocInfoInputSchema): Input schema data.
            config (RunnableConfig, optional): Execution configuration.

        Returns:
            dict: Metadata of the specified doc.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        api_url = self.connection.url + f"docs/{input_data.doc_id}"

        try:
            response = self.client.get(api_url, headers=self.connection.headers)
            response.raise_for_status()
            doc_info = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to fetch doc info. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to fetch metadata for the doc. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            return {
                "content": (
                    f"Doc Name: {doc_info['name']}\n"
                    f"Doc ID: {doc_info['id']}\n"
                    f"Owner: {doc_info['ownerName']} ({doc_info['owner']})\n"
                    f"Created At: {doc_info['createdAt']}\n"
                    f"Last Updated: {doc_info['updatedAt']}\n"
                )
            }

        return {"content": doc_info}
