from typing import Any, ClassVar, Literal

from pydantic import Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.operators import Pass
from dynamiq.nodes.types import DependencyTrigger
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
    def _is_branch_gated(
        node, error_sources: frozenset[str] | set[str] = frozenset(), seen: set[str] | None = None
    ) -> bool:
        """Whether `node` sits on a branch: downstream of a Choice option, a dependency condition or an error edge.

        Only such a node can legitimately be SKIPPED because a branch was not taken. A
        node with no gate anywhere in its ancestry always runs, so its success says
        nothing about which branch executed and must not license tolerating a skip.
        A node with an error edge starts two branches, its success and its failure, so
        the node itself and everything after it count as gated too.
        """
        seen = seen if seen is not None else set()
        if node.id in seen:
            return False
        seen.add(node.id)
        if node.id in error_sources:
            return True

        return any(
            dep.option
            or dep.condition
            or dep.trigger == DependencyTrigger.FAILURE
            or Output._is_branch_gated(dep.node, error_sources, seen)
            for dep in getattr(node, "depends", [])
        )

    def _error_source_ids(self) -> set[str]:
        """Nodes upstream of this output that an error edge leaves: their failure is a branch, not a dead end."""
        sources = {dep.node.id for dep in self.depends if dep.trigger == DependencyTrigger.FAILURE}
        seen, stack = set(), [dep.node for dep in self.depends]
        while stack:
            node = stack.pop()
            if node.id in seen:
                continue
            seen.add(node.id)
            for dep in getattr(node, "depends", []):
                if dep.trigger == DependencyTrigger.FAILURE:
                    sources.add(dep.node.id)
                stack.append(dep.node)
        return sources

    def validate_depends(self, depends_result: dict[str, RunnableResult]) -> None:
        """Validate dependencies, tolerating branches that were not taken.

        A workflow output is a join point: when a Choice sends execution down one branch,
        the other branch is SKIPPED, and the default rule would skip the output node too.
        Here a SKIPPED dependency is ignored as long as a *branch-gated* dependency
        took its branch -- i.e. some branch demonstrably ran. An error edge branches the same
        way: when its source fails, the failure is tolerated here as long as a gated
        dependency, the handler's branch, succeeded. The output's own error edge is a branch
        too: it is taken when its source failed, and not taken when the source succeeded.

        The scan is restricted to branch-gated dependencies on purpose. An Output node
        often also depends on an ungated node such as `input`, which always succeeds;
        counting it would make skip tolerance unconditional and let an Output whose every
        branch was skipped report success carrying nulls.

        Only the dependency's own SKIP status, and an error source's FAILURE, are tolerated;
        per-option and per-condition gates are still evaluated on every dependency, so
        conditional branching is unaffected. With a single dependency there is no
        successful sibling, so skip propagation behaves exactly as before.
        """
        error_sources = self._error_source_ids()
        gated = {dep.node.id for dep in self.depends if self._is_branch_gated(dep.node, error_sources)}

        def branch_taken(dep) -> bool:
            result = depends_result.get(dep.node.id)
            if result is None or dep.node.id not in gated:
                return False
            if dep.trigger == DependencyTrigger.FAILURE:
                return result.status == RunnableStatus.FAILURE
            return result.status == RunnableStatus.SUCCESS

        def branch_not_taken(dep) -> bool:
            result = depends_result.get(dep.node.id)
            if result is None or dep.node.id not in gated:
                return False
            if dep.trigger == DependencyTrigger.FAILURE:
                return result.status in (RunnableStatus.SUCCESS, RunnableStatus.SKIP)
            return result.status == RunnableStatus.SKIP or (
                result.status == RunnableStatus.FAILURE and dep.node.id in error_sources
            )

        tolerate_skips = any(branch_taken(dep) for dep in self.depends)
        for dep in self.depends:
            if tolerate_skips and branch_not_taken(dep):
                continue
            self._validate_dependency_status(depend=dep, depends_result=depends_result)
            if dep.condition:
                self._validate_dependency_condition(depend=dep, depends_result=depends_result)
            if dep.option:
                self._validate_dependency_option(depend=dep, depends_result=depends_result)
