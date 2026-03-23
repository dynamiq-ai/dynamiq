from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseAirtable


class ListBasesInput(BaseModel):
    """
    Input schema for listing all bases accessible by the token.
    Supports 'offset' to page through more than 1000 bases.
    """

    offset: str | None = Field(
        None, description="If specified, retrieves the next page of results using this offset token."
    )


class ListBases(BaseAirtable):
    """
    Node to list bases (GET /v0/meta/bases).
    """

    name: str = "ListBases"
    description: str = "Lists bases that the token can access."
    input_schema: ClassVar[type[ListBasesInput]] = ListBasesInput

    def execute(self, input_data: ListBasesInput, config: RunnableConfig = None, **kwargs):
        logger.info(f"Node {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        callbacks = config.callbacks if config else []
        self.run_on_node_execute_run(callbacks, **kwargs)

        url = f"{self.base_url}/meta/bases"
        params = {}
        if input_data.offset:
            params["offset"] = input_data.offset

        try:
            response = self.client.get(url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"{self.name} - {self.id}: failed to list bases. Error: {e}")
            raise ToolExecutionException(f"Failed to list bases: {str(e)}", recoverable=True)

        if self.is_optimized_for_agents:
            bases = data.get("bases", [])
            content = f"Found {len(bases)} base(s)."
            for base in bases:
                content += (
                    f"\n- {base.get('name', '<unknown>')}: "
                    f"{base.get('id', '<unknown>'), base.get('permissionLevel', '')}"
                )
            return {"content": content}
        else:
            return {"content": data}
