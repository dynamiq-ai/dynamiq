"""From Python 3.14 (PEP 649) annotations are evaluated lazily, in a scope that sees the whole class body. A class that
defines a `type` property or a `list()` method therefore makes `type[...]` or `list[...]` in its own annotations name
that member instead of the builtin: pydantic fails to build the model, and `inspect.signature` on the method raises.
Before 3.14 the same code works, so this check reads the source rather than relying on the interpreter running it."""

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
    """Every node evaluated in the class's own namespace: its body and the blocks nested in it (`if`, `try`, `with`,
    loops, `match`), but not the inside of the functions, classes, lambdas and comprehensions defined there."""

    def visit(node: ast.AST):
        for child in ast.iter_child_nodes(node):
            yield child
            if not isinstance(child, INNER_SCOPES):
                yield from visit(child)

    for statement in cls.body:
        yield statement
        if not isinstance(statement, INNER_SCOPES):
            yield from visit(statement)


def bound_names(cls: ast.ClassDef) -> set[str]:
    nodes = list(class_scope(cls))
    # `name: annotation` without a value declares the name but binds nothing.
    declared_only = {id(node.target) for node in nodes if isinstance(node, ast.AnnAssign) and node.value is None}
    names = set()
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and id(node) not in declared_only:
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def annotations(cls: ast.ClassDef):
    for node in class_scope(cls):
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
        bound = bound_names(cls)
        for annotation in annotations(cls):
            found.extend(
                f"{path}:{node.lineno}: {cls.name}.{node.id}"
                for node in ast.walk(annotation)
                if isinstance(node, ast.Name) and node.id in bound
            )
    return found


@pytest.mark.parametrize("root", ROOTS, ids=lambda root: root.name)
def test_no_class_annotation_names_a_member_of_its_own_class(root):
    found = [name for path in sorted(root.rglob("*.py")) for name in shadowed_annotation_names(path)]

    assert not found, "use builtins.<name>[...] for these annotations:\n" + "\n".join(found)


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
