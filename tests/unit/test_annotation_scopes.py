"""From Python 3.14 (PEP 649) annotations are evaluated lazily, in a scope that sees the whole class body. A class that
defines a `type` property or a `list()` method therefore makes `type[...]` or `list[...]` in its own annotations name
that member instead of the builtin: pydantic fails to build the model, and `inspect.signature` on the method raises.
Before 3.14 the same code works, so this check reads the source rather than relying on the interpreter running it."""

import ast
from pathlib import Path

import pytest

ROOTS = [Path(__file__).parents[2] / "dynamiq", Path(__file__).parents[1]]


def bound_names(cls: ast.ClassDef) -> set[str]:
    names = set()
    for statement in cls.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(statement.name)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                names.update(node.id for node in ast.walk(target) if isinstance(node, ast.Name))
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            if isinstance(statement.target, ast.Name):
                names.add(statement.target.id)
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in statement.names)
    return names


def annotations(cls: ast.ClassDef):
    for statement in cls.body:
        if isinstance(statement, ast.AnnAssign):
            yield statement.annotation
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = statement.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                if arg is not None and arg.annotation is not None:
                    yield arg.annotation
            if statement.returns is not None:
                yield statement.returns


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
