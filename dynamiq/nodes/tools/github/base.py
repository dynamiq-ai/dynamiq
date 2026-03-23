from typing import ClassVar

from dynamiq.connections import GitHub
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode


class BaseGitHub(ConnectionNode):
    """
    Base class for all GitHub nodes with common utilities and methods.
    """

    group: ClassVar[NodeGroup] = NodeGroup.TOOLS
    connection: GitHub
    is_optimized_for_agents: bool = True
