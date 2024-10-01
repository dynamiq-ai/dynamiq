from typing import Literal, Any

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.operators import Pass


class Input(Pass):
    """
    A utility node representing the input of workflow.

    This class inherits from the Pass operator and is used to mark the beginning of a sequence of
    operations. It is typically used in workflow definitions or process models.

    Attributes:
        group (Literal[NodeGroup.UTILS]): The group the node belongs to, set to UTILS.
        schema (dict[str, Any] | None): The JSON schema for the input data.
    """

    name: str | None = "Start"
    group: Literal[NodeGroup.UTILS] = NodeGroup.UTILS
    schema: dict[str, Any] | None = None


class Output(Pass):
    """
    A utility node representing the output of workflow.

    This class inherits from the Pass operator and is used to mark the conclusion of a sequence of
    operations. It is typically used in workflow definitions or process models.

    Attributes:
        group (Literal[NodeGroup.UTILS]): The group the node belongs to, set to UTILS.
        schema (dict[str, Any] | None): The JSON schema for the output data.
    """

    name: str | None = "End"
    group: Literal[NodeGroup.UTILS] = NodeGroup.UTILS
    schema: dict[str, Any] | None = None
