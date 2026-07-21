"""Back-compat guarantees for the knowledge-graph package move.

The KG nodes moved to ``dynamiq.nodes.knowledge_graphs``; the old locations
(``dynamiq.nodes.extractors`` / ``dynamiq.nodes.retrievers``) keep exposing the moved names
lazily via PEP 562 ``__getattr__`` to (a) avoid an import cycle at package-init time and
(b) still deserialize old serialized ``type`` strings.

These tests lock in the behavior that a lazy ``__getattr__`` alone would silently break:
``from ... import *`` only resolves names listed in ``__all__`` (a real per-name getattr that
fires ``__getattr__``); without ``__all__`` it reads ``__dict__`` and drops the moved names.
"""

import subprocess
import sys

import pytest

# (shim package, star-imported name, expected concrete class name in the new package)
MOVED = [
    ("dynamiq.nodes.extractors", "EntityExtractor", "KnowledgeGraphEntityExtractor"),
    ("dynamiq.nodes.extractors", "KnowledgeGraphWriter", "KnowledgeGraphWriter"),
    ("dynamiq.nodes.extractors", "Ontology", "Ontology"),
    ("dynamiq.nodes.extractors", "Triple", "Triple"),
    ("dynamiq.nodes.retrievers", "GraphRetriever", "KnowledgeGraphRetriever"),
]


def _star_import(module: str) -> dict:
    """Run ``from <module> import *`` and return the names it bound.

    ``import *`` is only legal at module scope, so it can't sit inside a test function directly --
    exec it into a throwaway namespace and inspect that instead.
    """
    ns: dict = {}
    exec(f"from {module} import *", ns)  # noqa: S102 - controlled, constant module name
    return ns


@pytest.mark.parametrize("module, name, _expected", MOVED)
def test_star_import_exposes_moved_name(module, name, _expected):
    """The regression guard: ``import *`` must still export each moved name."""
    assert name in _star_import(module), f"{name!r} missing from `from {module} import *`"


@pytest.mark.parametrize("module, name, expected_cls", MOVED)
def test_star_imported_name_is_the_real_class(module, name, expected_cls):
    """It resolves to the actual migrated class, not a stub/placeholder."""
    assert _star_import(module)[name].__name__ == expected_cls


@pytest.mark.parametrize("module, name, expected_cls", MOVED)
def test_direct_import_matches_star_import(module, name, expected_cls):
    """Direct attribute access (the deserialization path) resolves to the same object."""
    mod = __import__(module, fromlist=[name])
    assert getattr(mod, name) is _star_import(module)[name]


@pytest.mark.parametrize("module", ["dynamiq.nodes.extractors", "dynamiq.nodes.retrievers"])
def test_all_names_are_resolvable(module):
    """Every name promised in ``__all__`` must actually resolve (no dead entries)."""
    mod = __import__(module, fromlist=["__all__"])
    for name in mod.__all__:
        assert getattr(mod, name) is not None


@pytest.mark.parametrize("module", ["dynamiq.nodes.extractors", "dynamiq.nodes.retrievers"])
def test_cold_import_does_not_cycle(module):
    """A cold import of the shim (fresh interpreter) must not deadlock on the import cycle.

    Runs in a subprocess so ``sys.modules`` is empty -- an in-process import is a no-op once the
    package is already cached, so it can't catch a re-introduced eager cycle.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"cold `import {module}` failed:\n{result.stderr}"
