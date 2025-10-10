import json
import re
from typing import Any, ClassVar, Iterator, Literal

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.embedders.base import TextEmbedder
from dynamiq.nodes.node import NodeDependency, NodeGroup, ensure_config
from dynamiq.nodes.retrievers.base import Retriever
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class VectorStoreRetrieverInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a query to retrieve documents.")
    alpha: float = Field(default=0.0, description="Parameter to provide alpha for hybrid retrieval.")
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Parameter to provide filters to apply for retrieving specific documents."
    )
    top_k: int | None = Field(default=None, description="Parameter to provide how many documents to retrieve.")
    similarity_threshold: float | None = Field(
        default=None,
        description="Parameter to provide minimal similarity "
        "or maximal distance score accepted for retrieved documents.",
    )


class VectorStoreRetriever(Node):
    """Node for retrieving relevant documents based on a query.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): Group for the node. Defaults to NodeGroup.TOOLS.
        name (str): Name of the tool. Defaults to "Retrieval Tool".
        description (str): Description of the tool.
        error_handling (ErrorHandling): Error handling configuration.
        text_embedder (TextEmbedder): Text embedder node.
        document_retriever (Retriever): Document retriever node.
        filters (dict[str, Any] | None): Filters for document retrieval.
        top_k (int): The maximum number of documents to return.
        alpha (float): The alpha parameter for hybrid retrieval.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "VectorStore Retriever"
    description: str = "A node for retrieving relevant documents based on a query."
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    text_embedder: TextEmbedder
    document_retriever: Retriever
    filters: dict[str, Any] = {}
    top_k: int | None = None
    alpha: float = 0.0
    similarity_threshold: float | None = None

    input_schema: ClassVar[type[VectorStoreRetrieverInputSchema]] = VectorStoreRetrieverInputSchema
    _EXCLUDED_METADATA_FIELDS: ClassVar[tuple[str, ...]] = (
        "embedding",
        "embeddings",
        "vector",
        "vectors",
    )
    _EXCLUDED_METADATA_TOKENS: ClassVar[tuple[str, ...]] = ("id", "hash")
    _EXPECTED_METADATA_KEYWORDS: ClassVar[tuple[str, ...]] = ("url", "link", "source", "file", "title")

    def __init__(self, **kwargs):
        """
        Initializes the VectorStoreRetriever with the given parameters.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class constructor.
        """
        super().__init__(**kwargs)
        self._run_depends = []

    def reset_run_state(self):
        """
        Reset the intermediate steps (run_depends) of the node.
        """
        self._run_depends = []

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """
        Initialize the components of the tool.

        Args:
            connection_manager (ConnectionManager, optional): connection manager. Defaults to ConnectionManager.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.text_embedder.is_postponed_component_init:
            self.text_embedder.init_components(connection_manager)
        if self.document_retriever.is_postponed_component_init:
            self.document_retriever.init_components(connection_manager)

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {"text_embedder": True, "document_retriever": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["text_embedder"] = self.text_embedder.to_dict(**kwargs)
        data["document_retriever"] = self.document_retriever.to_dict(**kwargs)
        return data

    def format_content(self, documents: list[Document], metadata_fields: list[str] | None = None) -> str:
        """Format the retrieved documents' metadata and content.

        Args:
            documents (list[Document]): List of retrieved documents.
            metadata_fields (list[str]): Metadata fields to include. If None, uses all metadata except embeddings.

        Returns:
            str: Formatted content of the documents.
        """
        formatted_docs: list[str] = []

        normalized_metadata_fields: list[str] | None = None
        include_score = False
        if metadata_fields is not None:
            seen_fields: set[str] = set()
            cleaned_fields: list[str] = []
            for field in metadata_fields:
                stripped = field.strip() if field else ""
                if not stripped:
                    continue
                lowered = stripped.lower()
                if lowered in seen_fields:
                    continue
                if lowered == "score":
                    include_score = True
                    seen_fields.add(lowered)
                    continue
                seen_fields.add(lowered)
                cleaned_fields.append(stripped)

            if cleaned_fields:
                normalized_metadata_fields = cleaned_fields
            elif include_score:
                normalized_metadata_fields = []

        for index, doc in enumerate(documents):
            metadata = doc.metadata or {}
            metadata_lines: list[str] = []

            if normalized_metadata_fields is not None:
                if include_score and doc.score is not None:
                    metadata_lines.append(self._format_metadata_line("Score", doc.score))
            else:
                if doc.score is not None:
                    metadata_lines.append(self._format_metadata_line("Score", doc.score))

            metadata_lines.extend(self._generate_metadata_lines(metadata, normalized_metadata_fields))

            metadata_block = "\n\n".join(metadata_lines) if metadata_lines else "No metadata available."
            content_block = doc.content or ""

            formatted_doc = (
                f"\n\n== Source {index + 1} ==\n\n"
                f"\n\n== Metadata ==\n{metadata_block}\n\n"
                f"\n\n== Content (Source {index + 1}) ==\n\n{content_block}"
            ).rstrip()
            formatted_docs.append(formatted_doc)

        return "\n\n".join(formatted_docs)

    @staticmethod
    def _prettify_field_name(field_name: str) -> str:
        if not field_name:
            return field_name

        cleaned = field_name.replace("_", " ").strip()
        lowered = cleaned.lower()
        if lowered.startswith("dynamiq"):
            cleaned = cleaned[len("dynamiq") :].lstrip(" -_/")

        prettified = cleaned.strip().title()
        return prettified or field_name

    @classmethod
    def _format_metadata_line(cls, field: str, value: Any) -> str:
        formatted_value = cls._stringify_metadata_value(value)
        return f"{field}: {formatted_value}"

    @classmethod
    def _stringify_metadata_value(cls, value: Any) -> str:
        if isinstance(value, (dict, list)):
            try:
                serialized = json.dumps(value, indent=2, sort_keys=True)
                return cls._postprocess_metadata_string(serialized)
            except (TypeError, ValueError):
                return cls._postprocess_metadata_string(str(value))
        return cls._postprocess_metadata_string(str(value))

    @staticmethod
    def _postprocess_metadata_string(value: str) -> str:
        if not value:
            return value

        if value.lower().startswith("dynamiq"):
            trimmed = value[7:]
            return trimmed.lstrip("/\\ ")
        return value

    def _resolve_metadata_path(
        self,
        metadata: dict[str, Any],
        field: str,
    ) -> tuple[Any | None, list[str]]:
        if not metadata:
            return None, []

        parts = field.split(".")
        current: Any = metadata
        actual_path: list[str] = []

        for part in parts:
            if not isinstance(current, dict):
                return None, []

            matching_key = next((key for key in current.keys() if key.lower() == part.lower()), None)
            if matching_key is None:
                return None, []

            actual_path.append(matching_key)
            current = current[matching_key]

        return current, actual_path

    def _iter_metadata_entries(
        self,
        metadata: dict[str, Any],
        requested_fields: list[str] | None,
    ) -> Iterator[tuple[list[str], list[str], Any]]:
        if not metadata:
            return

        if requested_fields is None:
            for key, value in metadata.items():
                if self._should_exclude_metadata_key(key):
                    continue
                yield from self._flatten_metadata(value, [self._prettify_field_name(key)], [key])
            return

        for field in requested_fields:
            value, path = self._resolve_metadata_path(metadata, field)
            if not path:
                continue

            if any(self._should_exclude_metadata_key(part) for part in path):
                continue

            label_parts = [self._prettify_field_name(part) for part in path]
            yield from self._flatten_metadata(value, label_parts, path)

    def _flatten_metadata(
        self,
        value: Any,
        label_parts: list[str],
        raw_parts: list[str],
    ) -> Iterator[tuple[list[str], list[str], Any]]:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if self._should_exclude_metadata_key(key):
                    continue
                yield from self._flatten_metadata(
                    nested_value,
                    label_parts + [self._prettify_field_name(key)],
                    raw_parts + [key],
                )
            return

        yield (label_parts or ["Metadata"]), raw_parts, value

    def _generate_metadata_lines(
        self,
        metadata: dict[str, Any],
        requested_fields: list[str] | None,
    ) -> list[str]:
        if not metadata:
            return []

        base_entries = list(self._iter_metadata_entries(metadata, requested_fields))

        expected_source_entries = (
            base_entries if requested_fields is None else list(self._iter_metadata_entries(metadata, None))
        )

        expected_entries: list[tuple[list[str], list[str], Any]] = []
        expected_paths: set[tuple[str, ...]] = set()
        for display_parts, raw_parts, value in expected_source_entries:
            if not self._contains_expected_keyword(raw_parts):
                continue
            path_key = self._normalize_raw_path(raw_parts)
            if path_key in expected_paths:
                continue
            expected_entries.append((display_parts, raw_parts, value))
            expected_paths.add(path_key)

        seen_paths: set[tuple[str, ...]] = set()
        lines_by_path: dict[tuple[str, ...], str] = {}
        general_lines: list[str] = []

        for display_parts, raw_parts, value in base_entries:
            path_key = self._normalize_raw_path(raw_parts)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            line = self._format_metadata_line(" - ".join(display_parts), value)
            lines_by_path[path_key] = line
            if path_key not in expected_paths:
                general_lines.append(line)

        expected_lines: list[str] = []
        for display_parts, raw_parts, value in expected_entries:
            path_key = self._normalize_raw_path(raw_parts)
            if path_key not in seen_paths:
                line = self._format_metadata_line(" - ".join(display_parts), value)
                lines_by_path[path_key] = line
                seen_paths.add(path_key)
            expected_lines.append(lines_by_path[path_key])

        return general_lines + expected_lines

    @staticmethod
    def _normalize_raw_path(raw_parts: list[str]) -> tuple[str, ...]:
        return tuple(part.lower() for part in raw_parts)

    @classmethod
    def _should_exclude_metadata_key(cls, key: str) -> bool:
        if not key:
            return False

        lowered_key = key.lower()
        if lowered_key in cls._EXCLUDED_METADATA_FIELDS:
            return True

        tokens = cls._tokenize_metadata_key(key)
        return any(token in cls._EXCLUDED_METADATA_TOKENS for token in tokens)

    @classmethod
    def _contains_expected_keyword(cls, raw_parts: list[str]) -> bool:
        for part in raw_parts:
            tokens = cls._tokenize_metadata_key(part)
            if any(token in cls._EXPECTED_METADATA_KEYWORDS for token in tokens):
                return True
        return False

    @staticmethod
    def _tokenize_metadata_key(key: str) -> list[str]:
        if not key:
            return []

        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
        normalized = re.sub(r"[^0-9a-zA-Z]+", "_", normalized)
        return [token for token in normalized.lower().split("_") if token]

    def execute(
        self, input_data: VectorStoreRetrieverInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute the retrieval tool.

        Args:
            input_data (dict[str, Any]): Input data for the tool.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Result of the retrieval.
        """

        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        filters = input_data.filters or self.filters
        top_k = input_data.top_k or self.top_k
        similarity_threshold = (
            input_data.similarity_threshold
            if input_data.similarity_threshold is not None
            else self.similarity_threshold
        )

        alpha = input_data.alpha or self.alpha
        query = input_data.query
        try:
            kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
            kwargs.pop("run_depends", None)
            text_embedder_output = self.text_embedder.run(
                input_data={"query": query}, run_depends=self._run_depends, config=config, **kwargs
            )
            self._run_depends = [NodeDependency(node=self.text_embedder).to_dict(for_tracing=True)]
            embedding = text_embedder_output.output.get("embedding")

            document_retriever_output = self.document_retriever.run(
                input_data={
                    "embedding": embedding,
                    **({"top_k": top_k} if top_k else {}),
                    "filters": filters,
                    "alpha": alpha,
                    **({"query": query} if alpha else {}),
                    **({"similarity_threshold": similarity_threshold} if similarity_threshold is not None else {}),
                },
                run_depends=self._run_depends,
                config=config,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.document_retriever).to_dict(for_tracing=True)]
            retrieved_documents = document_retriever_output.output.get("documents", [])
            logger.debug(f"Tool {self.name} - {self.id}: retrieved {len(retrieved_documents)} documents")

            result = self.format_content(retrieved_documents)
            logger.info(f"Tool {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")

            return {"content": result, "documents": retrieved_documents}
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: execution error: {str(e)}", exc_info=True)
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve data using the specified action. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )
