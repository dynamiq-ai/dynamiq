from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode, PageContentUnion


class InitialPage(BaseModel):
    """
    Structure for the 'initialPage' parameter.
    See: https://coda.io/developers/apis/v1#operation/createDoc
    """

    name: str | None = Field(default=None, description="Name of the page.")
    subtitle: str | None = Field(default=None, description="Subtitle of the page.")
    icon_name: str | None = Field(default=None, alias="iconName", description="Name of the icon to show on the page.")
    image_url: str | None = Field(default=None, alias="imageUrl", description="URL of the cover image to use.")
    parent_page_id: str | None = Field(
        default=None, alias="parentPageId", description="Parent page ID if creating a subpage."
    )
    page_content: PageContentUnion | None = Field(
        default=None,
        alias="pageContent",
        description="Actual page content, either textual/HTML/Markdown (canvas), embed, or syncPage.",
    )

    class Config:
        allow_population_by_field_name = True


class CodaCreateDocInputSchema(BaseModel):
    """
    Input schema for creating a new Coda doc.
    """

    title: str | None = Field(default="Untitled", description="Title of the new doc. Defaults to 'Untitled'.")
    source_doc: str | None = Field(
        default=None, alias="sourceDoc", description="Optional doc ID from which to create a copy."
    )
    timezone: str | None = Field(default=None, description="The timezone for the newly created doc.")
    folder_id: str | None = Field(
        default=None, alias="folderId", description="ID of the folder within which to create this doc."
    )
    initial_page: InitialPage | None = Field(
        default=None, alias="initialPage", description="Optional details of the initial page of the new doc."
    )

    class Config:
        allow_population_by_field_name = True


class CodaCreateDoc(BaseCodaNode):
    """
    Node to create a new Coda doc.
    """

    name: str = "CodaCreateDoc"
    description: str = "Create a new Coda doc with optional parameters."
    input_schema: ClassVar[type[CodaCreateDocInputSchema]] = CodaCreateDocInputSchema

    def execute(self, input_data: CodaCreateDocInputSchema, config: RunnableConfig | None = None, **kwargs):
        """
        Execute the node to create a new Coda doc.

        Args:
            input_data (CodaCreateDocInputSchema): Input schema data.
            config (RunnableConfig, optional): Execution configuration.

        Returns:
            dict: Metadata of the created doc.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        payload_dict = input_data.model_dump(exclude_none=True, by_alias=True)
        payload = self.dict_to_camel_case(payload_dict)

        api_url = self.connection.url + "docs"

        try:
            response = self.client.post(api_url, headers=self.connection.headers, json=payload)
            response.raise_for_status()
            created_doc = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to create doc. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to create a new doc. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            return {
                "content": (
                    f"New Doc Created:\n"
                    f"Title: {created_doc['name']}\n"
                    f"Doc ID: {created_doc['id']}\n"
                    f"Owner: {created_doc['ownerName']} ({created_doc['owner']})\n"
                    f"Link: {created_doc['browserLink']}\n"
                )
            }
        else:
            return {"content": created_doc}
