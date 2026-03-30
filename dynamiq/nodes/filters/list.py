from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.operators import Choice
from dynamiq.nodes.types import ChoiceCondition
from dynamiq.runnables import RunnableConfig


class ListFilterInputSchema(BaseModel):
    input: list[Any] = Field(..., description="Parameter to provide list for filtering")


class ListFilter(Node):
    group: Literal[NodeGroup.FILTERS] = NodeGroup.FILTERS
    name: str = "list-filter"
    description: str = "Node that returns filtered list"
    filters: ChoiceCondition

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[ListFilterInputSchema]] = ListFilterInputSchema

    def execute(self, input_data: ListFilterInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Returns the list of filtered items.

        Args:
            input_data (ListFilterInputSchema): input data for the tool, which includes list for filtering.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing filtered items.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        input_list = input_data.input
        filters = self.filters
        try:
            result = [item for item in input_list if Choice.evaluate(filters, item)]
            return {"output": result}
        except Exception as e:
            raise ValueError(f"Encountered an error while performing filtering. \nError details: {e}")
