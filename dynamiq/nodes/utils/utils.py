from typing import Any, ClassVar, Literal

from pydantic import Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.operators import Pass
from dynamiq.runnables import RunnableResult, RunnableStatus


class Input(Pass):
    """
    A utility node representing the input of workflow.

    This class inherits from the Pass operator and is used to mark the beginning of a sequence of
    operations. It is typically used in workflow definitions or process models.

    Attributes:
        group (Literal[NodeGroup.UTILS]): The group the node belongs to, set to UTILS.
        schema (dict[str, Any] | None): The JSON schema for the input data.
    """

    name: str | None = "start"
    group: Literal[NodeGroup.UTILS] = NodeGroup.UTILS
    json_schema: dict[str, Any] | None = Field(
        default=None,
        alias="schema",
        description="""Determines input parameters of workflow.
        Provide it in the properties field format. Example:
        "properties": {
            "query": {
                "type": "Any"
            },
            "files": {
                "type": "list[files]"
            }
        }
    """,
    )
    _json_schema_fields: ClassVar[list[str]] = ["json_schema"]


class Output(Pass):
    """
    A utility node representing the output of workflow.

    This class inherits from the Pass operator and is used to mark the conclusion of a sequence of
    operations. It is typically used in workflow definitions or process models.

    Attributes:
        group (Literal[NodeGroup.UTILS]): The group the node belongs to, set to UTILS.
        schema (dict[str, Any] | None): The JSON schema for the output data.
    """

    name: str | None = "end"
    group: Literal[NodeGroup.UTILS] = NodeGroup.UTILS
    json_schema: dict[str, Any] | None = Field(
        default=None,
        alias="schema",
        description="""Determines output parameters of workflow.
        Provide it in the properties field format. Example:
        "properties": {
            "query": {
                "type": "Any"
            }
        }
    """,
    )
    _json_schema_fields: ClassVar[list[str]] = ["json_schema"]

    def validate_depends(self, depends_result: dict[str, RunnableResult]) -> None:
        """Validate dependencies, tolerating branches that were not taken.

        A workflow output is a join point: when a Choice sends execution down one branch,
        the other branch is SKIPPED, and the default rule would skip the output node too.
        Here a SKIPPED dependency is ignored as long as at least one dependency succeeded.

        Only the dependency's own SKIP status is tolerated; per-option and per-condition
        gates are still evaluated on every dependency, so conditional branching is
        unaffected. With a single dependency there is no successful sibling, so skip
        propagation behaves exactly as before.
        """
        tolerate_skips = any(
            depends_result[dep.node.id].status == RunnableStatus.SUCCESS
            for dep in self.depends
            if dep.node.id in depends_result
        )

        for dep in self.depends:
            dep_result = depends_result.get(dep.node.id)
            if tolerate_skips and dep_result is not None and dep_result.status == RunnableStatus.SKIP:
                continue
            self._validate_dependency_status(depend=dep, depends_result=depends_result)
            if dep.condition:
                self._validate_dependency_condition(depend=dep, depends_result=depends_result)
            if dep.option:
                self._validate_dependency_option(depend=dep, depends_result=depends_result)
