"""The packages export their members lazily, and every export resolves to what it named before."""

import ast
import importlib
import pkgutil
import re
import sys
import types
from importlib._bootstrap import _DeadlockError
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

import dynamiq
from dynamiq.connections import BaseConnection
from dynamiq.nodes import Node
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils.lazy_imports import (
    OPTIONAL_DEPENDENCIES,
    OptionalDependencyError,
    import_type,
    lazy_exports,
    optional_dependency_error,
)

REPO_ROOT = Path(dynamiq.__file__).parent.parent
LAZY_PACKAGES = sorted(
    ".".join(init.parent.relative_to(REPO_ROOT).parts)
    for init in (REPO_ROOT / "dynamiq").rglob("__init__.py")
    if "_LAZY_EXPORTS = {" in init.read_text()
)
MISSING_PACKAGE = re.compile(r"needs the '(?P<distribution>[^']+)' package")

# Type strings that existing workflows still use for classes that moved to dynamiq.nodes.knowledge_graphs.
LEGACY_TYPES = {
    "dynamiq.nodes.extractors.EntityExtractor": "KnowledgeGraphEntityExtractor",
    "dynamiq.nodes.extractors.KnowledgeGraphWriter": "KnowledgeGraphWriter",
    "dynamiq.nodes.extractors.Ontology": "Ontology",
    "dynamiq.nodes.extractors.Triple": "Triple",
    "dynamiq.nodes.retrievers.GraphRetriever": "KnowledgeGraphRetriever",
}


def _resolve(package: str, name: str):
    """Return the export, or None when it needs an extra that is not installed."""
    try:
        return getattr(importlib.import_module(package), name)
    except OptionalDependencyError as error:
        distribution = MISSING_PACKAGE.search(str(error))["distribution"]
        with pytest.raises(PackageNotFoundError):
            version(distribution)
        assert 'pip install "dynamiq[' in str(error)
        return None


def _type_checking_imports(package: str) -> dict[str, str]:
    """Map each name imported under ``if TYPE_CHECKING:`` in the package to its relative module."""
    tree = ast.parse((REPO_ROOT / Path(*package.split(".")) / "__init__.py").read_text())
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for statement in node.body:
                assert isinstance(statement, ast.ImportFrom)
                module = "." * statement.level + (statement.module or "")
                if statement.level == 0:
                    module = "." + statement.module.removeprefix(f"{package}.")
                for alias in statement.names:
                    imports[alias.asname or alias.name] = f"{module}:{alias.name}"
    return imports


def test_packages_with_lazy_exports_are_found():
    assert {"dynamiq", "dynamiq.nodes.llms", "dynamiq.nodes.tools", "dynamiq.storages.vector"} <= set(LAZY_PACKAGES)


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_lazy_exports_are_listed_in_all_and_type_checking(package):
    module = importlib.import_module(package)
    exports = module._LAZY_EXPORTS
    assert set(exports) <= set(module.__all__)
    assert set(exports) <= set(dir(module))

    relative = {}
    for name, target in exports.items():
        if target.startswith("."):
            module_name, _, attribute = target.partition(":")
            relative[name] = f"{module_name}:{attribute or name}"
    assert _type_checking_imports(package) == relative


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_lazy_exports_do_not_shadow_submodules(package):
    module = importlib.import_module(package)
    submodules = {info.name for info in pkgutil.iter_modules(module.__path__)}
    assert not submodules & set(module._LAZY_EXPORTS)


@pytest.mark.parametrize("package", [*LAZY_PACKAGES, "dynamiq.connections"])
def test_type_strings_resolve_to_their_class(package):
    module = importlib.import_module(package)
    for name in getattr(module, "_LAZY_EXPORTS", None) or [name for name in dir(module) if not name.startswith("_")]:
        cls = _resolve(package, name)
        # A class exported under another name, such as BaseAgent, is not what its own type string names.
        if not isinstance(cls, type) or not issubclass(cls, (Node, BaseConnection)) or cls.__name__ != name:
            continue
        type_string = f"{cls.__module__.rsplit('.', 1)[0]}.{cls.__name__}"
        assert WorkflowYAMLLoader.get_entity_by_type(type_string) is cls


@pytest.mark.parametrize(("legacy_type", "class_name"), LEGACY_TYPES.items())
def test_legacy_type_strings_resolve(legacy_type, class_name):
    assert WorkflowYAMLLoader.get_entity_by_type(legacy_type).__name__ == class_name


def test_every_module_imports_unless_an_extra_it_needs_is_missing():
    failures = []
    for module in pkgutil.walk_packages(dynamiq.__path__, "dynamiq."):
        try:
            importlib.import_module(module.name)
        except ImportError as error:
            if not optional_dependency_error(error, module.name):
                failures.append(f"{module.name}: {error!r}")
    assert not failures


def test_optional_dependencies_match_pyproject_extras():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]

    def names(requirements: list[str]) -> set[str]:
        return {re.match(r"[A-Za-z0-9_.-]+", requirement)[0].lower() for requirement in requirements}

    extras = project["optional-dependencies"]
    core = names(project["dependencies"])
    for distribution, extra in OPTIONAL_DEPENDENCIES.values():
        assert distribution.lower() in names(extras[extra])
        assert distribution.lower() not in core
    mapped = {distribution.lower() for distribution, _ in OPTIONAL_DEPENDENCIES.values()}
    in_extras = set().union(*(names(requirements) for extra, requirements in extras.items() if extra != "all"))
    # Never missing: python-dateutil comes with boto3, and pyOpenSSL only pins what snowflake-connector-python installs.
    assert in_extras - mapped == {"python-dateutil", "pyopenssl"}
    (everything,) = extras["all"]
    assert set(re.fullmatch(r"dynamiq\[(.*)\]", everything)[1].split(",")) == set(extras) - {"all", "cua", "monty"}


@pytest.fixture
def package(tmp_path, monkeypatch):
    """A throwaway package whose exports live in submodules written by the test."""
    monkeypatch.delenv("DYNAMIQ_EAGER_IMPORTS", raising=False)
    root = tmp_path / "lazy_pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("lazy_pkg")
    yield root, module
    for name in [name for name in sys.modules if name == "lazy_pkg" or name.startswith("lazy_pkg.")]:
        del sys.modules[name]


def test_export_is_imported_on_first_access_and_cached(package):
    root, module = package
    (root / "things.py").write_text("class Thing:\n    pass\n\nOther = Thing\n")
    lazy_exports("lazy_pkg", {"Thing": ".things", "Alias": ".things:Other"}, vars(module))

    assert "lazy_pkg.things" not in sys.modules
    assert "Thing" in dir(module)
    thing = module.Thing
    assert thing is sys.modules["lazy_pkg.things"].Thing
    assert vars(module)["Thing"] is thing
    assert module.Alias is thing


def test_unknown_export_raises_attribute_error(package):
    _, module = package
    lazy_exports("lazy_pkg", {}, vars(module))

    with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
        module.Missing


def test_missing_optional_dependency_names_the_extra(package, monkeypatch):
    root, module = package
    (root / "store.py").write_text("import qdrant_client\n\nclass Store:\n    pass\n")
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    lazy_exports("lazy_pkg", {"Store": ".store"}, vars(module))

    with pytest.raises(OptionalDependencyError) as error:
        module.Store
    assert str(error.value) == (
        "lazy_pkg.Store needs the 'qdrant-client' package, which is not installed. "
        'Install it with: pip install "dynamiq[vectorstores]"'
    )
    assert isinstance(error.value, ModuleNotFoundError)
    assert error.value.name == "qdrant_client"


def test_other_import_errors_are_not_reported_as_missing_extras(package):
    root, module = package
    (root / "broken.py").write_text("import a_module_that_does_not_exist\n")
    lazy_exports("lazy_pkg", {"Broken": ".broken"}, vars(module))

    with pytest.raises(ModuleNotFoundError) as error:
        module.Broken
    assert not isinstance(error.value, OptionalDependencyError)


def test_an_error_while_an_export_imports_is_not_taken_for_a_missing_name(package):
    root, module = package
    (root / "broken.py").write_text("import math\n\nmath.no_such_function\n")
    lazy_exports("lazy_pkg", {"Broken": ".broken"}, vars(module))

    with pytest.raises(ImportError, match="no attribute 'no_such_function'"):
        getattr(module, "Broken", None)


def test_eager_imports_resolve_every_export_at_package_import(package, monkeypatch):
    root, module = package
    # Child comes first and imports Base from the package, which has to resolve while the exports are imported.
    (root / "child.py").write_text("from lazy_pkg import Base\n\nclass Child(Base):\n    pass\n")
    (root / "base.py").write_text("class Base:\n    pass\n")
    (root / "store.py").write_text("import qdrant_client\n")
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    monkeypatch.setenv("DYNAMIQ_EAGER_IMPORTS", "1")

    lazy_exports("lazy_pkg", {"Child": ".child", "Base": ".base", "Store": ".store"}, vars(module))

    assert issubclass(vars(module)["Child"], vars(module)["Base"])
    assert "Store" not in vars(module)
    with pytest.raises(OptionalDependencyError):
        module.Store


def test_an_import_stopped_by_a_deadlock_error_is_retried(package, monkeypatch):
    root, module = package
    (root / "things.py").write_text("class Thing:\n    pass\n")
    import_module = importlib.import_module
    calls = []

    def deadlock_once(name, package=None):
        calls.append(name)
        if len(calls) == 1:
            raise _DeadlockError(f"deadlock detected by _ModuleLock({name!r})")
        return import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", deadlock_once)
    lazy_exports("lazy_pkg", {"Thing": ".things"}, vars(module))

    assert module.Thing is sys.modules["lazy_pkg.things"].Thing
    assert calls == [".things", ".things"]


def test_import_type_returns_the_named_object(monkeypatch):
    module = types.ModuleType("dynamiq_test_module")
    module.Thing = object()
    monkeypatch.setitem(sys.modules, "dynamiq_test_module", module)

    assert import_type("dynamiq_test_module.Thing") is module.Thing
    with pytest.raises(AttributeError):
        import_type("dynamiq_test_module.Missing")


def test_import_type_names_the_extra_a_module_needs(package, monkeypatch):
    root, _ = package
    (root / "store.py").write_text("import qdrant_client\n\nclass Store:\n    pass\n")
    monkeypatch.setitem(sys.modules, "qdrant_client", None)

    with pytest.raises(OptionalDependencyError, match="lazy_pkg.store.Store needs the 'qdrant-client' package"):
        import_type("lazy_pkg.store.Store")
