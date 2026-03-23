from typing import ClassVar

from dynamiq.connections import MailerLite
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode


class BaseMailerLite(ConnectionNode):
    """
    Base class for all MailerLite nodes with shared utilities and logic.
    """

    group: ClassVar[NodeGroup] = NodeGroup.TOOLS
    connection: ClassVar[MailerLite] = MailerLite
    is_optimized_for_agents: bool = True
