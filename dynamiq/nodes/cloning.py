"""Cloning a node for isolated execution, and carrying id-keyed run config onto the clone.

Both the agent (parallel tool calls, factory sub-agents) and the ``Map`` operator clone a
node per iteration and give the clone fresh ids, so two concurrent copies do not collide in
tracing or streaming. Anything in ``RunnableConfig`` keyed by node id — ``nodes_override``,
``mock.exclude`` — stops matching the moment those ids change, which silently drops the
caller's intent. These helpers keep the walk and the realignment in one place so a new
id-keyed config field only has to be handled once.
"""

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel

if TYPE_CHECKING:
    from dynamiq.runnables import RunnableConfig


def regenerate_node_ids(obj: Any, id_map: dict[str, set[str]] | None = None) -> Any:
    """Recursively assign new ids to a cloned node and its nested models, in place.

    Args:
        obj: The object to walk.
        id_map: Optional collector, populated with ``{old_id: {new_id, ...}}``. A single
            original can yield several clones — a tool reachable by two paths in a cloned
            subtree, say — so each old id maps to a set rather than the last writer.

    Returns:
        Any: ``obj``, with every nested ``id`` replaced.
    """
    if isinstance(obj, BaseModel):
        if hasattr(obj, "id"):
            previous_id = getattr(obj, "id")
            new_id = str(uuid4())
            setattr(obj, "id", new_id)
            if id_map is not None and isinstance(previous_id, str):
                id_map.setdefault(previous_id, set()).add(new_id)

        for field_name in getattr(obj, "model_fields", {}):
            value = getattr(obj, field_name)
            if isinstance(value, list):
                setattr(obj, field_name, [regenerate_node_ids(item, id_map) for item in value])
            elif isinstance(value, dict):
                setattr(obj, field_name, {k: regenerate_node_ids(v, id_map) for k, v in value.items()})
            else:
                setattr(obj, field_name, regenerate_node_ids(value, id_map))
        return obj
    if isinstance(obj, list):
        return [regenerate_node_ids(item, id_map) for item in obj]
    if isinstance(obj, dict):
        return {k: regenerate_node_ids(v, id_map) for k, v in obj.items()}
    return obj


def carry_mock_exclusions(config: "RunnableConfig", id_map: dict[str, set[str]]) -> "RunnableConfig":
    """Extend ``config.mock.exclude_ids`` to cover the clones of every excluded node.

    An id-based mock exclusion says "this specific node must really run". Cloning gives that
    node a new id, so without this the exclusion stops matching and the clone is mocked
    against the caller's wishes — the opposite of what was asked for, and silently.

    Args:
        config: The run config for the cloned execution.
        id_map: ``{old_id: {new_id, ...}}`` from :func:`regenerate_node_ids`.

    Returns:
        RunnableConfig: ``config`` unchanged when there is nothing to carry, otherwise a
        shallow copy whose ``mock`` covers the new ids.
    """
    run_mock = getattr(config, "mock", None)
    if not run_mock or not run_mock.exclude_ids or not id_map:
        return config

    carried: set[str] = set()
    for old_id in run_mock.exclude_ids & id_map.keys():
        carried |= id_map[old_id]
    if not carried - run_mock.exclude_ids:
        return config

    config = config.model_copy(deep=False)
    config.mock = run_mock.model_copy(update={"exclude_ids": run_mock.exclude_ids | carried})
    return config
