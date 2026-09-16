"""Cloning a node for isolated execution, and carrying id-keyed run config onto the clone.

Both the agent (parallel tool calls, factory sub-agents) and the ``Map`` operator clone a
node per iteration and give the clone fresh ids, so two concurrent copies do not collide in
tracing or streaming. Anything in ``RunnableConfig`` keyed by node id — ``nodes_override``,
``mock.exclude`` — stops matching the moment those ids change, which silently drops the
caller's intent. These helpers keep the walk and the realignment in one place so a new
id-keyed config field only has to be handled once.
"""

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel

if TYPE_CHECKING:
    from dynamiq.runnables import RunnableConfig


def regenerate_node_ids(obj: Any, id_map: dict[str, set[str]] | None = None) -> Any:
    """Recursively assign new ids to a cloned node and its nested models, in place.

    Transformer paths that address a node by id (``$.<id>.output``), such as the ones between the
    nodes of a flow a SubWorkflow holds, are rewritten to the new ids afterwards, so the clone keeps
    reading the outputs it read before; a dependency gated on a Choice option follows the option's
    new id the same way, so the gate keeps holding. Only node ids drive the path rewrite and only
    option ids the gates: a column or a rule may carry the same text as an input key without meaning it.

    Args:
        obj: The object to walk.
        id_map: Optional collector, populated with ``{old_id: {new_id, ...}}``. A single
            original can yield several clones — a tool reachable by two paths in a cloned
            subtree, say — so each old id maps to a set rather than the last writer.

    Returns:
        Any: ``obj``, with every nested ``id`` replaced.
    """
    if id_map is None:
        id_map = {}
    node_ids: dict[str, str] = {}
    option_ids: dict[str, str] = {}
    _regenerate_ids(obj, id_map, node_ids, option_ids, seen=set())
    _remap_transformer_paths(obj, node_ids)
    _remap_dependency_options(obj, option_ids)
    return obj


def _regenerate_ids(
    obj: Any, id_map: dict[str, set[str]], node_ids: dict[str, str], option_ids: dict[str, str], seen: set[int]
) -> Any:
    # Imported here: the node and operator modules import this one.
    from dynamiq.nodes.node import Node
    from dynamiq.nodes.operators.operators import ChoiceOption

    if isinstance(obj, BaseModel):
        # One object reachable by two paths, such as a node listed in a flow and named in another
        # node's dependencies, gets one new id.
        if id(obj) in seen:
            return obj
        seen.add(id(obj))
        if hasattr(obj, "id"):
            previous_id = getattr(obj, "id")
            new_id = str(uuid4())
            setattr(obj, "id", new_id)
            if isinstance(previous_id, str):
                id_map.setdefault(previous_id, set()).add(new_id)
                if isinstance(obj, Node):
                    node_ids[previous_id] = new_id
                elif isinstance(obj, ChoiceOption):
                    option_ids[previous_id] = new_id

        for field_name in getattr(obj, "model_fields", {}):
            value = getattr(obj, field_name)
            if isinstance(value, list):
                setattr(obj, field_name, [_regenerate_ids(item, id_map, node_ids, option_ids, seen) for item in value])
            elif isinstance(value, dict):
                setattr(
                    obj,
                    field_name,
                    {k: _regenerate_ids(v, id_map, node_ids, option_ids, seen) for k, v in value.items()},
                )
            else:
                setattr(obj, field_name, _regenerate_ids(value, id_map, node_ids, option_ids, seen))
        return obj
    if isinstance(obj, list):
        return [_regenerate_ids(item, id_map, node_ids, option_ids, seen) for item in obj]
    if isinstance(obj, dict):
        return {k: _regenerate_ids(v, id_map, node_ids, option_ids, seen) for k, v in obj.items()}
    return obj


def _remap_transformer_paths(obj: Any, renamed: dict[str, str]) -> None:
    # Imported here: the node module is the one that imports this package's operators.
    from dynamiq.nodes.node import Transformer

    if not renamed:
        return
    # An id already quoted by an earlier pass, as under a Map inside a Map, is matched too.
    pattern = re.compile(r'\$\."?(' + "|".join(re.escape(old) for old in renamed) + r')"?(?=\.|$|\s|\|)')

    # Quoted, because a generated id may start with a digit or hold a dash, which a bare field cannot.
    def rename(path: Any) -> Any:
        return pattern.sub(lambda match: f'$."{renamed[match.group(1)]}"', path) if isinstance(path, str) else path

    for transformer in (model for model in _models(obj) if isinstance(model, Transformer)):
        transformer.path = rename(transformer.path)
        if transformer.selector:
            transformer.selector = {key: rename(value) for key, value in transformer.selector.items()}


def _remap_dependency_options(obj: Any, renamed: dict[str, str]) -> None:
    from dynamiq.nodes.node import NodeDependency

    # A gate is matched by string against the Choice's option ids, which were just renamed.
    for dependency in (model for model in _models(obj) if isinstance(model, NodeDependency)):
        if dependency.option in renamed:
            dependency.option = renamed[dependency.option]


def _models(obj: Any, seen: set[int] | None = None) -> Iterator[BaseModel]:
    """Every model reachable from ``obj`` through fields, lists and dicts, each once."""
    seen = set() if seen is None else seen
    if isinstance(obj, BaseModel):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        yield obj
        for field_name in type(obj).model_fields:
            yield from _models(getattr(obj, field_name, None), seen)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _models(item, seen)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _models(item, seen)


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
