"""Importing dynamiq loads no integration SDK, and what used to happen at import time still happens.

Each test runs in a fresh interpreter, since the test session has already imported most of dynamiq.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import dynamiq

# Modules that only the feature using them, or the first LLM call, may import.
DEFERRED_MODULES = [
    "litellm",
    "openai",
    "pandas",
    "matplotlib",
    "bs4",
    "markdown",
    "reportlab",
    "pdfplumber",
    "unstructured_client",
    "chromadb",
    "elasticsearch",
    "opensearchpy",
    "pgvector",
    "pinecone",
    "pymilvus",
    "qdrant_client",
    "turbopuffer",
    "weaviate",
    "databricks",
    "mysql",
    "snowflake",
    "browserbase",
    "playwright",
    "stagehand",
    "daytona",
    "e2b",
    "google.cloud.aiplatform",
    "google.cloud.documentai",
    "rapidfuzz",
    "sacrebleu",
    "rouge_score",
    "mcp",
    "uvicorn",
    "starlette",
]
# A model only dynamiq's registry knows, so litellm reports its context window only once the registry is synced.
REGISTRY_MODEL = "minimax/MiniMax-M2.7"
REGISTRY_MAX_INPUT_TOKENS = json.loads((Path(dynamiq.__file__).parent / "nodes/llms/model_registry.json").read_text())[
    REGISTRY_MODEL
]["max_input_tokens"]
# import dynamiq does not import its subpackages, so an import cycle between two of them breaks only a program
# that imports one of them first.
PACKAGES = sorted(init.parent.name for init in Path(dynamiq.__file__).parent.glob("*/__init__.py"))


def run_python(code: str, **env: str) -> list[str]:
    """Run code in a fresh interpreter and return the lines it prints."""
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "DYNAMIQ_EAGER_IMPORTS": "", **env},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_importing_the_core_loads_no_integration():
    output = run_python(
        f"""
        import sys

        import dynamiq
        from dynamiq import Workflow, connections
        from dynamiq.flows import Flow
        from dynamiq.nodes.agents import Agent
        from dynamiq.nodes.llms import Anthropic, Bedrock, Gemini, OpenAI
        from dynamiq.nodes.tools import HttpApiCall, MCPServer, Python
        from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

        print(sorted(module for module in {DEFERRED_MODULES!r} if module in sys.modules))
        """
    )
    assert output == ["[]"]


@pytest.mark.parametrize("eager", ["", "1"], ids=["lazy", "eager"])
@pytest.mark.parametrize("package", PACKAGES)
def test_each_package_imports_first(package, eager):
    run_python(f"import dynamiq.{package}", DYNAMIQ_EAGER_IMPORTS=eager)


def test_creating_an_llm_imports_litellm_with_the_registry_and_patches():
    output = run_python(
        f"""
        import sys

        from dynamiq import connections
        from dynamiq.nodes.llms import Anthropic

        print("litellm" in sys.modules)
        Anthropic(model="claude-sonnet-4-5", connection=connections.Anthropic(api_key="test"))

        import litellm
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        print(litellm.get_model_info(model="{REGISTRY_MODEL}")["max_input_tokens"])
        tool = {{"type": "function", "function": {{"name": "f", "parameters": {{"type": "object"}}, "strict": True}}}}
        print(AnthropicConfig()._map_tool_helper(tool)[0].get("strict"))
        """
    )
    assert output == ["False", str(REGISTRY_MAX_INPUT_TOKENS), "True"]


def test_litellm_imported_before_dynamiq_gets_the_registry_at_import():
    output = run_python(
        f"""
        import litellm

        import dynamiq.nodes.llms.registry

        print(litellm.get_model_info(model="{REGISTRY_MODEL}")["max_input_tokens"])
        """
    )
    assert output == [str(REGISTRY_MAX_INPUT_TOKENS)]


@pytest.mark.parametrize("first_import", ["", "import litellm"])
def test_get_litellm_knows_the_registry_models(first_import):
    output = run_python(
        f"""
        {first_import}
        from dynamiq.utils.litellm import get_litellm

        print(get_litellm().get_model_info(model="{REGISTRY_MODEL}")["max_input_tokens"])
        """
    )
    assert output == [str(REGISTRY_MAX_INPUT_TOKENS)]


def test_a_missing_sdk_is_reported_with_its_extra():
    output = run_python(
        """
        import sys

        sys.modules["qdrant_client"] = None

        from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader, WorkflowYAMLLoaderException

        # Workflows name a class through its package or its module.
        for entity_type in (
            "dynamiq.nodes.retrievers.QdrantDocumentRetriever",
            "dynamiq.nodes.writers.qdrant.QdrantDocumentWriter",
        ):
            try:
                WorkflowYAMLLoader.get_entity_by_type(entity_type)
            except WorkflowYAMLLoaderException as error:
                print(error)
        """
    )
    assert output == [
        "Entity 'dynamiq.nodes.retrievers.QdrantDocumentRetriever' is not available. "
        "dynamiq.nodes.retrievers.QdrantDocumentRetriever needs the 'qdrant-client' package, which is not installed. "
        'Install it with: pip install "dynamiq[vectorstores]"',
        "Entity 'dynamiq.nodes.writers.qdrant.QdrantDocumentWriter' is not available. "
        "dynamiq.nodes.writers.qdrant.QdrantDocumentWriter needs the 'qdrant-client' package, which is not installed. "
        'Install it with: pip install "dynamiq[vectorstores]"',
    ]


def test_eager_imports_env_var_restores_import_time_loading():
    output = run_python(
        """
        import sys

        import dynamiq.nodes.converters

        print("dynamiq.nodes.converters.pptx" in sys.modules, "litellm" in sys.modules)
        """,
        DYNAMIQ_EAGER_IMPORTS="1",
    )
    assert output == ["True True"]


def test_retry_check_does_not_import_litellm():
    output = run_python(
        """
        import sys

        from dynamiq.nodes.node import is_retryable_error

        print(is_retryable_error(ValueError("boom")), "litellm" in sys.modules)
        """
    )
    assert output == ["True False"]


def test_threads_resolve_node_types_concurrently_on_first_use():
    output = run_python(
        """
        import importlib
        import pathlib
        import threading

        import dynamiq
        from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader, WorkflowYAMLLoaderException
        from dynamiq.utils.lazy_imports import OptionalDependencyError

        root = pathlib.Path(dynamiq.__file__).parent
        packages = sorted(
            ".".join(init.parent.relative_to(root.parent).parts)
            for init in (root / "nodes").rglob("__init__.py")
            if "_LAZY_EXPORTS = {" in init.read_text()
        )
        types = [f"{package}.{name}" for package in packages for name in importlib.import_module(package)._LAZY_EXPORTS]
        threads_count = 16
        barrier = threading.Barrier(threads_count)
        errors = []

        def resolve(offset):
            barrier.wait()
            for index in range(len(types)):
                entity_type = types[(index * 7 + offset) % len(types)]
                try:
                    WorkflowYAMLLoader.get_entity_by_type(entity_type, {})
                except WorkflowYAMLLoaderException as error:
                    if not isinstance(error.__cause__, OptionalDependencyError):
                        errors.append(f"{entity_type}: {error!r}")
                except Exception as error:
                    errors.append(f"{entity_type}: {error!r}")

        threads = [threading.Thread(target=resolve, args=(offset * 11,)) for offset in range(threads_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print(len(types) > 150, errors[:3])
        """
    )
    assert output == ["True []"]
