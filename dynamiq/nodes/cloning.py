"""Cloning a node for isolated execution, and carrying id-keyed run config onto the clone.

Both the agent (parallel tool calls, factory sub-agents) and the ``Map`` operator clone a
node per iteration and give the clone fresh ids, so two concurrent copies do not collide in
tracing or streaming. Anything in ``RunnableConfig`` keyed by node id — ``nodes_override``,
``mock.exclude`` — stops matching the moment those ids change, which silently drops the
caller's intent. These helpers keep the walk and the realignment in one place so a new
id-keyed config field only has to be handled once.
"""

import re
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel

if TYPE_CHECKING:
    from dynamiq.runnables import RunnableConfig


def regenerate_node_ids(obj: Any, id_map: dict[str, set[str]] | None = None) -> Any:
    """Recursively assign new ids to a cloned node and its nested models, in place.

    Transformer paths that address a node by id (``$.<id>.output``), such as the ones between the
    nodes of a flow a SubWorkflow holds, are rewritten to the new ids afterwards, so the clone keeps
    reading the outputs it read before; a Choice option's condition naming a node follows it too, and
    a dependency gated on a Choice option follows the option's new id, so the gate keeps holding.
    An id is unique only where it is read, a node id within its flow and an option id within its
    Choice, so a path follows the ids of the flow holding the node that reads it and a gate the
    options of the Choice it depends on: two nested flows or two Choices that spell an id alike never
    rewrite each other's references, and the copied node's own paths, which read the flow around it
    rather than anything the copy carries, are left as written.
    Only node ids drive the path rewrite and only option ids the gates: a column or a rule may carry
    the same text as an input key without meaning it. An output transformer and a dependency's own
    condition read a result rather than the flow, so they are left as written. A rule, a row or a
    field keeps the id the user wrote, which findings and test coverage are keyed by.

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
    renamed: dict[int, tuple[str, str]] = {}
    _regenerate_ids(obj, id_map, renamed, seen=set())
    _remap_paths(obj, renamed)
    _remap_dependency_options(obj, renamed)
    return obj


def _regenerate_ids(obj: Any, id_map: dict[str, set[str]], renamed: dict[int, tuple[str, str]], seen: set[int]) -> Any:
    # Imported here: the node and operator modules import this one.
    from dynamiq.nodes.node import Node
    from dynamiq.nodes.operators.operators import ChoiceOption

    if isinstance(obj, BaseModel):
        # One object reachable by two paths, such as a node listed in a flow and named in another
        # node's dependencies, gets one new id.
        if id(obj) in seen:
            return obj
        seen.add(id(obj))
        # A rule, a row or a field keeps the id the user wrote, and holds nothing else to rename.
        if getattr(obj, "keeps_id", False):
            return obj
        if hasattr(obj, "id"):
            previous_id = getattr(obj, "id")
            new_id = str(uuid4())
            setattr(obj, "id", new_id)
            if isinstance(previous_id, str):
                id_map.setdefault(previous_id, set()).add(new_id)
                # Keyed by the object, not the old id: two Choices may each own an option called
                # `default`, and two nested flows a node called `start`.
                if isinstance(obj, (Node, ChoiceOption)):
                    renamed[id(obj)] = (previous_id, new_id)

        for field_name in getattr(obj, "model_fields", {}):
            value = getattr(obj, field_name)
            if isinstance(value, list):
                setattr(obj, field_name, [_regenerate_ids(item, id_map, renamed, seen) for item in value])
            elif isinstance(value, dict):
                setattr(obj, field_name, {k: _regenerate_ids(v, id_map, renamed, seen) for k, v in value.items()})
            else:
                setattr(obj, field_name, _regenerate_ids(value, id_map, renamed, seen))
        return obj
    if isinstance(obj, list):
        return [_regenerate_ids(item, id_map, renamed, seen) for item in obj]
    if isinstance(obj, dict):
        return {k: _regenerate_ids(v, id_map, renamed, seen) for k, v in obj.items()}
    return obj


def _renames(models: Iterable[Any], renamed: dict[int, tuple[str, str]]) -> dict[str, str]:
    """The old-to-new ids of the given models, for those the walk renamed."""
    return dict(pair for model in models if (pair := renamed.get(id(model))) is not None)


def _path_renamer(renamed: dict[str, str]) -> Callable[[Any], Any]:
    """A function rewriting `$.<old id>` or `$['<old id>']` at the head of a path to the new id.

    An id already quoted by an earlier pass, as under a Map inside a Map, is matched too. The dotted form
    quotes the new id because a generated id may start with a digit, which a bare field cannot; the bracket
    form, which shipped flows write as `$['splitter'].output`, keeps its brackets. In the dotted form the id
    ends where a character that cannot be part of an id follows, so `start` never matches `start-2` while
    `{{$.start}}` and `$.start['output']` are matched; the quotes come in pairs, so a path inside a quoted
    string keeps its closing quote.
    """
    alternatives = "|".join(re.escape(old) for old in renamed)
    pattern = re.compile(
        r'\$(?:\.(?:"(' + alternatives + r')"|(' + alternatives + r"))(?![A-Za-z0-9_-])"
        r"|\[([\"']?)(" + alternatives + r")\3\])"
    )

    def substitute(match: re.Match) -> str:
        dotted = match.group(1) or match.group(2)
        if dotted is not None:
            return f'$."{renamed[dotted]}"'
        return f"$['{renamed[match.group(4)]}']"

    def rename(path: Any) -> Any:
        return pattern.sub(substitute, path) if isinstance(path, str) else path

    return rename


def _remap_paths(obj: Any, renamed: dict[int, tuple[str, str]]) -> None:
    # Imported here: the flow, node and operator modules import this one.
    from dynamiq.flows.base import BaseFlow
    from dynamiq.nodes.node import Node
    from dynamiq.nodes.operators.operators import Choice

    models = list(_models(obj))
    # A node's input holds the results of its flow keyed by node id, and a Choice option's condition reads
    # that input, so both follow the ids of the flow holding the node. A node held by no flow in the copy,
    # the copied node itself above all, reads a flow the copy does not carry. A dependency's condition
    # reads the dependency's result (status, input, output, error) rather than the flow, so a node named
    # `output` or `status` must leave it alone.
    for flow in (model for model in models if isinstance(model, BaseFlow)):
        nodes = getattr(flow, "nodes", None) or []
        if not (scope := _renames(nodes, renamed)):
            continue
        rename = _path_renamer(scope)
        for node in nodes:
            _rename_transformer(node.input_transformer, rename)
            if isinstance(node, Choice):
                for option in node.options or []:
                    _rename_condition(option.condition, rename)
    # An output transformer selects from the node's own output, whose keys are not node ids, so a node
    # named like one of them must not pull its paths along. A SubWorkflow whose flow lacks a single
    # Output node is the exception: it returns every inner node's output keyed by the ids just renamed.
    for node in (model for model in models if isinstance(model, Node) and _outputs_by_node_id(model)):
        if scope := _renames(node.flow.nodes, renamed):
            _rename_transformer(node.output_transformer, _path_renamer(scope))


def _rename_transformer(transformer: Any, rename: Callable[[Any], Any]) -> None:
    if transformer is None:
        return
    transformer.path = rename(transformer.path)
    if transformer.selector:
        transformer.selector = {key: rename(value) for key, value in transformer.selector.items()}


def _outputs_by_node_id(node: Any) -> bool:
    from dynamiq.nodes.operators.sub_workflow import SubWorkflow
    from dynamiq.nodes.utils import Output

    if not isinstance(node, SubWorkflow) or node.flow is None:
        return False
    return sum(isinstance(inner, Output) for inner in node.flow.nodes) != 1


def _rename_condition(condition: Any, rename: Callable[[Any], Any]) -> None:
    if condition is None:
        return
    condition.variable = rename(condition.variable)
    for operand in condition.operands or []:
        _rename_condition(operand, rename)


def _remap_dependency_options(obj: Any, renamed: dict[int, tuple[str, str]]) -> None:
    from dynamiq.nodes.node import NodeDependency

    # A gate is matched by string against the option ids of the Choice it depends on, which were just
    # renamed; another Choice's option of the same name is not the one it names.
    for dependency in (model for model in _models(obj) if isinstance(model, NodeDependency)):
        if dependency.option is None:
            continue
        options = _renames(getattr(dependency.node, "options", None) or [], renamed)
        if dependency.option in options:
            dependency.option = options[dependency.option]


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
