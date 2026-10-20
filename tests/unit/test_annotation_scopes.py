"""From Python 3.14 (PEP 649) annotations are evaluated lazily, in a scope that sees the whole class body. A class that
defines a `type` property or a `list()` method therefore makes `type[...]` or `list[...]` in its own annotations name
that member instead of the builtin: pydantic fails to build the model, and `inspect.signature` on the method raises.
Before 3.14 the same code works, because an annotation was evaluated before the rest of the body ran, so this check
reads the source rather than relying on the interpreter running it. Only a name the class binds at or after the
annotation changes meaning; one bound before it, such as an inner Enum used as a field type, resolves the same way on
every version."""

import ast
from pathlib import Path

import pytest

ROOTS = [Path(__file__).parents[2] / "dynamiq", Path(__file__).parents[1]]


INNER_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def class_scope(cls: ast.ClassDef):
    """(node, statement) for every node evaluated in the class's own namespace, with the statement it belongs to: the
    class body and the blocks nested in it (`if`, `try`, `with`, loops, `match`), but not the inside of the functions,
    classes, lambdas and comprehensions defined there."""

    def visit(node: ast.AST, statement: ast.stmt):
        for child in ast.iter_child_nodes(node):
            child_statement = child if isinstance(child, ast.stmt) else statement
            yield child, child_statement
            if not isinstance(child, INNER_SCOPES):
                yield from visit(child, child_statement)

    for statement in cls.body:
        yield statement, statement
        if not isinstance(statement, INNER_SCOPES):
            yield from visit(statement, statement)


def bindings(cls: ast.ClassDef) -> dict[str, list[tuple[int, int]]]:
    """Each name the class body binds, with where each binding statement ends: the name is bound once it has run."""
    nodes = list(class_scope(cls))
    # `name: annotation` without a value declares the name but binds nothing.
    declared_only = {id(node.target) for node, _ in nodes if isinstance(node, ast.AnnAssign) and node.value is None}
    ends: dict[str, list[tuple[int, int]]] = {}
    for node, statement in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and id(node) not in declared_only:
            name = node.id
        elif isinstance(node, ast.alias):
            name = (node.asname or node.name).split(".")[0]
        elif isinstance(node, ast.ExceptHandler) and node.name:
            name = node.name
        else:
            continue
        ends.setdefault(name, []).append((statement.end_lineno, statement.end_col_offset))
    return ends


def annotations(cls: ast.ClassDef):
    for node, _ in class_scope(cls):
        if isinstance(node, ast.AnnAssign):
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                if arg is not None and arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns


def shadowed_annotation_names(path: Path) -> list[str]:
    found = []
    for cls in (node for node in ast.walk(ast.parse(path.read_text(), str(path))) if isinstance(node, ast.ClassDef)):
        bound = bindings(cls)
        for annotation in annotations(cls):
            start = (annotation.lineno, annotation.col_offset)
            found.extend(
                f"{path}:{node.lineno}: {cls.name}.{node.id}"
                for node in ast.walk(annotation)
                if isinstance(node, ast.Name) and any(end >= start for end in bound.get(node.id, ()))
            )
    return found


@pytest.mark.parametrize("root", ROOTS, ids=lambda root: root.name)
def test_no_class_annotation_names_a_member_of_its_own_class(root):
    found = [name for path in sorted(root.rglob("*.py")) for name in shadowed_annotation_names(path)]

    assert not found, (
        "From Python 3.14 these annotations name a member their class binds at or after them. Use builtins.<name> for "
        "a builtin, or rename the member:\n" + "\n".join(found)
    )


def test_the_check_catches_a_method_named_like_the_builtin_it_annotates(tmp_path):
    source = tmp_path / "store.py"
    source.write_text("class Store:\n    def list(self) -> list[str]:\n        return []\n")

    assert shadowed_annotation_names(source) == [f"{source}:2: Store.list"]


def test_the_check_looks_inside_blocks_in_the_class_body(tmp_path):
    source = tmp_path / "store.py"
    source.write_text(
        "import sys\n"
        "class Store:\n"
        "    if sys.version_info >= (3, 0):\n"
        "        def list(self) -> list[str]:\n"
        "            return []\n"
    )

    assert shadowed_annotation_names(source) == [f"{source}:4: Store.list"]


def test_a_name_only_declared_in_the_class_body_binds_nothing(tmp_path):
    source = tmp_path / "error.py"
    source.write_text("class Error:\n    type: type[Exception]\n")

    assert shadowed_annotation_names(source) == []


def test_a_name_bound_before_the_annotation_is_left_alone(tmp_path):
    source = tmp_path / "step.py"
    source.write_text("class Step:\n    class Kind:\n        pass\n    kind: Kind\n")

    assert shadowed_annotation_names(source) == []


def test_a_name_bound_again_after_the_annotation_is_flagged(tmp_path):
    source = tmp_path / "step.py"
    source.write_text("class Step:\n    class Kind:\n        pass\n    kind: Kind\n    Kind = int\n")

    assert shadowed_annotation_names(source) == [f"{source}:4: Step.Kind"]
