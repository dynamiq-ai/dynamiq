from typing import ClassVar

from dynamiq.connections import Airtable
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode


class BaseAirtable(ConnectionNode):
    """
    Base class for all Airtable nodes with convenience utilities.
    Inherits from `ConnectionNode` to handle `connection`.
    """

    group: ClassVar[NodeGroup] = NodeGroup.TOOLS
    connection: Airtable
    is_optimized_for_agents: bool = True

    @property
    def base_url(self) -> str:
        """Return the base URL for Airtable from the connection."""
        return self.connection.url
