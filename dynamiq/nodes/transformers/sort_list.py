import enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig


class SortOrder(str, enum.Enum):
    ASC = "ASC"
    DESC = "DESC"


class SortListInputSchema(BaseModel):
    input: list[Any] = Field(..., description="Parameter to provide list for sorting")
    field: str | None = Field(None, description="Parameter to provide field, that will be used for sorting objects")


class SortList(Node):
    group: Literal[NodeGroup.TRANSFORMERS] = NodeGroup.TRANSFORMERS
    name: str = "sort-list"
    description: str = "Node that returns sorted list"
    sort_by: SortOrder = SortOrder.DESC

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[SortListInputSchema]] = SortListInputSchema

    def execute(self, input_data: SortListInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Node for sorting list based on field and sort order.

        Args:
            input_data (SortListInputSchema): input data for the tool, which includes list for sorting.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing sorted list.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        input_list = input_data.input
        field = input_data.field
        try:
            reverse = self.sort_by == SortOrder.DESC
            if field:
                return {
                    "output": sorted(
                        input_list,
                        key=lambda item: item.get(field) if isinstance(item, dict) else getattr(item, field, None),
                        reverse=reverse,
                    )
                }
            else:
                return {"output": sorted(input_list, reverse=reverse)}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing filtering. \nError details: {e}")
