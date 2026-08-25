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

    @staticmethod
    def _is_branch_gated(node, seen: set[str] | None = None) -> bool:
        """Whether `node` sits downstream of a Choice option or a dependency condition.

        Only such a node can legitimately be SKIPPED because a branch was not taken. A
        node with no gate anywhere in its ancestry always runs, so its success says
        nothing about which branch executed and must not license tolerating a skip.
        """
        seen = seen if seen is not None else set()
        if node.id in seen:
            return False
        seen.add(node.id)

        return any(
            dep.option or dep.condition or Output._is_branch_gated(dep.node, seen)
            for dep in getattr(node, "depends", [])
        )

    def validate_depends(self, depends_result: dict[str, RunnableResult]) -> None:
        """Validate dependencies, tolerating branches that were not taken.

        A workflow output is a join point: when a Choice sends execution down one branch,
        the other branch is SKIPPED, and the default rule would skip the output node too.
        Here a SKIPPED dependency is ignored as long as a *branch-gated* dependency
        succeeded -- i.e. some branch demonstrably ran.

        The scan is restricted to branch-gated dependencies on purpose. An Output node
        often also depends on an ungated node such as `input`, which always succeeds;
        counting it would make skip tolerance unconditional and let an Output whose every
        branch was skipped report success carrying nulls.

        Only the dependency's own SKIP status is tolerated; per-option and per-condition
        gates are still evaluated on every dependency, so conditional branching is
        unaffected. With a single dependency there is no successful sibling, so skip
        propagation behaves exactly as before.
        """
        gated = {dep.node.id for dep in self.depends if self._is_branch_gated(dep.node)}
        tolerate_skips = any(
            depends_result[node_id].status == RunnableStatus.SUCCESS for node_id in gated if node_id in depends_result
        )

        for dep in self.depends:
            dep_result = depends_result.get(dep.node.id)
            if (
                tolerate_skips
                and dep.node.id in gated
                and dep_result is not None
                and dep_result.status == RunnableStatus.SKIP
            ):
                continue
            self._validate_dependency_status(depend=dep, depends_result=depends_result)
            if dep.condition:
                self._validate_dependency_condition(depend=dep, depends_result=depends_result)
            if dep.option:
                self._validate_dependency_option(depend=dep, depends_result=depends_result)
