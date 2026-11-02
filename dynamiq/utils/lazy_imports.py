"""Deferred imports for dynamiq's integrations.

Serialized workflows name node classes as ``dynamiq.nodes.<package>.<Class>``, so every package exports each class it
holds. Importing those classes eagerly loads every integration's SDK, so the packages instead map their exports to
modules with :func:`lazy_exports`, and a module is imported when one of its names is first used. Reading a name whose
extra is not installed raises :class:`OptionalDependencyError`, including through ``hasattr``, ``getattr`` with a
default and ``from <package> import *``.
"""

import importlib
import importlib.util
import os
import sys
import threading
import time
from importlib._bootstrap import _DeadlockError
from types import ModuleType
from typing import Any

EAGER_IMPORTS_ENV_VAR = "DYNAMIQ_EAGER_IMPORTS"
_IMPORT_ATTEMPTS_ON_DEADLOCK = 3

# The optional dependencies, keyed by the module they provide, with the distribution and the extra that install them.
OPTIONAL_DEPENDENCIES: dict[str, tuple[str, str]] = {
    "bs4": ("beautifulsoup4", "files"),
    "inscriptis": ("inscriptis", "files"),
    "markdown": ("markdown", "files"),
    "pdf2image": ("pdf2image", "files"),
    "pdfplumber": ("pdfplumber", "files"),
    "reportlab": ("reportlab", "files"),
    "unstructured_client": ("unstructured-client", "files"),
    "matplotlib": ("matplotlib", "data"),
    "seaborn": ("seaborn", "data"),
    "chromadb": ("chromadb-client", "vectorstores"),
    "elasticsearch": ("elasticsearch", "vectorstores"),
    "opensearchpy": ("opensearch-py", "vectorstores"),
    "pgvector": ("pgvector", "vectorstores"),
    "pinecone": ("pinecone-client", "vectorstores"),
    "pymilvus": ("pymilvus", "vectorstores"),
    "qdrant_client": ("qdrant-client", "vectorstores"),
    "requests_aws4auth": ("requests-aws4auth", "vectorstores"),
    "turbopuffer": ("turbopuffer", "vectorstores"),
    "weaviate": ("weaviate-client", "vectorstores"),
    "databricks": ("databricks-sql-connector", "sql"),
    "mysql": ("mysql-connector-python", "sql"),
    "snowflake": ("snowflake-connector-python", "sql"),
    "browserbase": ("browserbase", "browser"),
    "playwright": ("playwright", "browser"),
    "stagehand": ("stagehand", "browser"),
    "steel": ("steel-sdk", "browser"),
    "daytona": ("daytona", "sandboxes"),
    "e2b": ("e2b", "sandboxes"),
    "e2b_code_interpreter": ("e2b-code-interpreter", "sandboxes"),
    "e2b_desktop": ("e2b-desktop", "sandboxes"),
    "google.api_core": ("google-cloud-aiplatform", "gcp"),
    "google.cloud.aiplatform": ("google-cloud-aiplatform", "gcp"),
    "google.cloud.aiplatform_v1": ("google-cloud-aiplatform", "gcp"),
    "google.cloud.documentai": ("google-cloud-documentai", "gcp"),
    "vertexai": ("google-cloud-aiplatform", "gcp"),
    "rapidfuzz": ("rapidfuzz", "evals"),
    "rouge_score": ("rouge-score", "evals"),
    "sacrebleu": ("sacrebleu", "evals"),
    "computer": ("cua-computer", "cua"),
    "pydantic_monty": ("pydantic-monty", "monty"),
}

# Serializes the first import of the module a serialized "type" string names: the YAML loader resolves node types
# from a thread pool, and threads that import modules importing each other can fail with a deadlock error. Never
# acquire it while a module is being imported, since a thread holding a module's import lock and waiting for this one
# deadlocks with a thread holding this one and importing that module.
TYPE_RESOLUTION_LOCK = threading.RLock()


class OptionalDependencyError(ModuleNotFoundError):
    """Raised when an integration needs an optional dependency that is not installed.

    It is a ``ModuleNotFoundError``, the error a missing dependency raised before the extras existed, so code that
    catches that keeps working.
    """


def optional_dependency_error(error: ImportError, owner: str) -> OptionalDependencyError | None:
    """Explain an import failure caused by an optional dependency that is not installed.

    Args:
        error: The error raised while importing the integration.
        owner: The name the caller asked for, shown in the message.

    Returns:
        An error naming the extra that installs the dependency, or None if the failure has another cause, such as a
        bug in an installed package.
    """
    missing = error.name or ""
    for module, (distribution, extra) in OPTIONAL_DEPENDENCIES.items():
        if missing == module or missing.startswith(f"{module}.") or module.startswith(f"{missing}."):
            if _is_installed(module):
                return None
            return OptionalDependencyError(
                f"{owner} needs the '{distribution}' package, which is not installed. "
                f'Install it with: pip install "dynamiq[{extra}]"',
                name=error.name,
            )
    return None


def _is_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # a parent package is missing
        return False
    except ValueError:  # the module is in sys.modules without a spec, as modules created at run time are
        return True


def eager_imports_enabled() -> bool:
    """Whether ``DYNAMIQ_EAGER_IMPORTS`` asks for every export to be imported when its package is."""
    return os.getenv(EAGER_IMPORTS_ENV_VAR, "").strip().lower() in ("1", "true")


def lazy_exports(package: str, exports: dict[str, str], namespace: dict[str, Any]) -> None:
    """Make a package import its exports on first use, by adding ``__getattr__`` and ``__dir__`` to it (PEP 562).

    Adding them here, rather than assigning them in the package, keeps them out of sight of type checkers, which
    then report a name the package does not export instead of typing it ``Any``.

    Args:
        package: The package's ``__name__``.
        exports: Maps each exported name to ``"<module>"`` or ``"<module>:<attribute>"``. A module that starts with
            "." is relative to the package, and the attribute defaults to the exported name.
        namespace: The package's ``globals()``. A resolved name is stored there, so later reads never reach
            ``__getattr__`` again.

    Raises:
        OptionalDependencyError: From ``__getattr__``, when an export needs an extra that is not installed.
    """

    def __getattr__(name: str) -> Any:
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module_name, _, attribute = target.partition(":")
        try:
            module = _import_module(module_name, package)
        except ImportError as error:
            if optional_error := optional_dependency_error(error, f"{package}.{name}"):
                raise optional_error from error
            raise
        except AttributeError as error:
            # Raised while the module ran. Raised from here, it would read as a name the package does not export.
            raise ImportError(f"Importing {package}.{name} failed: {error}") from error
        value = getattr(module, attribute or name)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(exports))

    # A star import reaches __getattr__ only for the names in the package's __all__, so __all__ lists every export.
    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__
    if eager_imports_enabled():
        for name in exports:
            try:
                __getattr__(name)
            except OptionalDependencyError:
                # Left lazy: it raises the same error when it is used.
                continue


def _import_module(name: str, package: str | None = None) -> ModuleType:
    """Import a module, again if Python stops the import with a deadlock error.

    Python raises that error, instead of waiting, in one of two threads that each hold an import lock the other
    needs, such as threads importing a package and one of its submodules at once. It undoes that thread's unfinished
    imports, so importing again once the other thread has finished normally succeeds.
    """
    for attempt in range(_IMPORT_ATTEMPTS_ON_DEADLOCK):
        try:
            return importlib.import_module(name, package)
        except _DeadlockError:
            if attempt == _IMPORT_ATTEMPTS_ON_DEADLOCK - 1:
                raise
            time.sleep(0.1 * (attempt + 1))  # an immediate retry often runs into the same imports again


def import_type(path: str) -> Any:
    """Return the object a serialized type string such as ``"dynamiq.nodes.llms.OpenAI"`` names.

    Args:
        path: The module and the attribute, joined by the last ".".

    Returns:
        The attribute.

    Raises:
        OptionalDependencyError: If the module needs an extra that is not installed.
        ImportError: If the module cannot be imported for another reason.
        AttributeError: If the module has no such attribute.
        ValueError: If the path names no module.
    """
    module_name, _, name = path.rpartition(".")
    module = sys.modules.get(module_name)
    if module is not None and name in vars(module):
        return vars(module)[name]
    with TYPE_RESOLUTION_LOCK:
        try:
            module = _import_module(module_name)
        except ImportError as error:
            if optional_error := optional_dependency_error(error, path):
                raise optional_error from error
            raise
        return getattr(module, name)
