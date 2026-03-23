from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

from ..base import BaseAirtable


class ListViewsInput(BaseModel):
    """
    Input schema for listing views of a given base.
    """

    base_id: str = Field(..., description="The ID of the  base.")
    include_visible_field_ids: bool = Field(
        default=False,
        description="If true, appends `include=visibleFieldIds` to retrieve visible field IDs in grid views.",
    )


class ListViews(BaseAirtable):
    """
    Node to list basic information about all views in a given base.

    GET /v0/meta/bases/{baseId}/views
    """

    name: str = "ListViews"
    description: str = "Lists basic info for all views in a specified base."
    input_schema: ClassVar[type[ListViewsInput]] = ListViewsInput

    def execute(
        self,
        input_data: ListViewsInput,
        config: RunnableConfig = None,
        **kwargs,
    ):
        logger.info(f"Node {self.name} - {self.id} started with input:\n{input_data.model_dump()}")
        callbacks = config.callbacks if config else []
        self.run_on_node_execute_run(callbacks, **kwargs)

        base_id = input_data.base_id
        url = f"{self.base_url}/meta/bases/{base_id}/views"
        params = {}
        if input_data.include_visible_field_ids:
            params["include"] = "visibleFieldIds"

        try:
            response = self.client.get(url, headers=self.connection.headers, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"{self.name} - {self.id}: failed to list views. Error: {e}")
            raise ToolExecutionException(f"Failed to list views in base '{base_id}': {str(e)}", recoverable=True)

        if self.is_optimized_for_agents:
            views_list = data.get("views", [])
            lines = [f"Base {base_id} has {len(views_list)} view(s):"]
            for i, view in enumerate(views_list, start=1):
                view_id = view.get("id", "<no_id>")
                view_name = view.get("name", "<no_name>")
                view_type = view.get("type", "<no_type>")
                personal_user_id = view.get("personalForUserId")

                visible_field_ids = view.get("visibleFieldIds")

                line = f"{i}) '{view_name}' (type: {view_type}, id: {view_id})"
                if personal_user_id:
                    line += f", personal for user {personal_user_id}"
                if visible_field_ids:
                    line += f"\n   Visible fields: {', '.join(visible_field_ids)}"

                lines.append(line)

            summary_str = "\n".join(lines)
            return {
                "content": summary_str,
            }
        else:
            return {"content": data}
