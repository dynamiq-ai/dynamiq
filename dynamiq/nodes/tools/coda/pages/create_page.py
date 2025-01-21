from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseCodaNode, PageContentUnion


class CodaCreatePageInputSchema(BaseModel):
    """
    Input schema for creating a new page in a Coda doc.
    """

    doc_id: str = Field(..., alias="docId", description="ID of the doc.")
    name: str | None = Field(None, description="Name of the page.")
    subtitle: str | None = Field(None, description="Subtitle of the page.")
    icon_name: str | None = Field(None, alias="iconName", description="Name of the icon.")
    image_url: str | None = Field(None, alias="imageUrl", description="URL of the cover image.")
    parent_page_id: str | None = Field(None, alias="parentPageId", description="ID of the parent page.")
    page_content: PageContentUnion | None = Field(
        None, alias="pageContent", description="Page content (canvas, embed, or syncPage)."
    )

    class Config:
        populate_by_name = True


class CodaCreatePage(BaseCodaNode):
    """
    Node to create a new page within a Coda doc.
    """

    name: str = "CodaCreatePage"
    description: str = "Create a new page in a Coda doc."
    input_schema: ClassVar[type[CodaCreatePageInputSchema]] = CodaCreatePageInputSchema

    def execute(self, input_data: CodaCreatePageInputSchema, config: RunnableConfig | None = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        doc_id = input_data.doc_id
        api_url = f"{self.connection.url}docs/{doc_id}/pages"

        payload_dict = input_data.model_dump(exclude_none=True, by_alias=True)
        payload = self.dict_to_camel_case(payload_dict)

        try:
            response = self.client.post(api_url, headers=self.connection.headers, json=payload)
            response.raise_for_status()
            created_page = response.json()  # The API returns details about the created page
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to create page. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to create a page. Error: {str(e)}.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            return {
                "content": (
                    f"New Page Created:\n"
                    f"Name: {created_page.get('name')}\n"
                    f"Page ID: {created_page.get('id')}\n"
                    f"Browser Link: {created_page.get('browserLink')}\n"
                )
            }
        else:
            return {"content": created_page}
