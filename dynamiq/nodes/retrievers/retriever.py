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
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger


class VectorStoreRetrieverInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a query to retrieve documents.")
    alpha: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Parameter to provide alpha for hybrid retrieval. 0 is keyword-only, 1 is semantic-only.",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter to provide filters to apply for retrieving specific documents.",
    )
    top_k: int | None = Field(default=None, description="Parameter to provide how many documents to retrieve.")
    similarity_threshold: float | None = Field(
        default=None,
        description="Post-retrieval score threshold. For Weaviate hybrid search this is a query-relative fused score, "
        "not an absolute semantic similarity; prefer max_vector_distance for semantic gating.",
    )
    max_vector_distance: float | None = Field(
        default=None,
        ge=0,
        description="Maximum vector distance for hybrid search; supported by Weaviate.",
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
        document_reranker (Node | None): Optional document_reranker node for reranking retrieved documents.
        filters (dict[str, Any] | None): Filters for document retrieval.
        top_k (int): The maximum number of documents to return.
        alpha (float): The alpha parameter for hybrid retrieval. 0 is keyword-only, 1 is semantic-only.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.SEMANTIC_SEARCH
    name: str = "vector-store-retriever"
    description: str = "A node for retrieving relevant documents based on a query."
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    text_embedder: TextEmbedder
    document_retriever: Retriever
    document_reranker: Node | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = None
    alpha: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Default alpha for hybrid retrieval. 0 is keyword-only, 1 is semantic-only.",
    )
    similarity_threshold: float | None = Field(
        default=None,
        description="Post-retrieval score threshold; hybrid fused scores are query-relative.",
    )
    max_vector_distance: float | None = Field(default=None, ge=0)
    agent_metadata_fields: list[str] | None = Field(
        default_factory=lambda: [
            "score",
            "title",
            "source_url",
            "url",
            "source",
        ],
        description="Metadata fields to include when this retriever is used as an agent tool.",
    )
    skip_empty_metadata: bool = Field(
        default=True,
        description="Whether to omit null and empty metadata values from formatted retrieval output.",
    )
    input_schema: ClassVar[type[VectorStoreRetrieverInputSchema]] = VectorStoreRetrieverInputSchema
    _METADATA_SCORE_PRECISION: ClassVar[int] = 3
    _EXCLUDED_METADATA_FIELDS: ClassVar[tuple[str, ...]] = (
        "embedding",
        "embeddings",
        "vector",
        "vectors",
    )
    _EXCLUDED_METADATA_TOKENS: ClassVar[tuple[str, ...]] = ("id", "hash")
    _EXPECTED_METADATA_KEYWORDS: ClassVar[tuple[str, ...]] = (
        "url",
        "link",
        "source",
        "file",
        "title",
    )

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
        if self.document_reranker and self.document_reranker.is_postponed_component_init:
            self.document_reranker.init_components(connection_manager)

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {
            "text_embedder": True,
            "document_retriever": True,
            "document_reranker": True,
        }

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["text_embedder"] = self.text_embedder.to_dict(**kwargs)
        data["document_retriever"] = self.document_retriever.to_dict(**kwargs)
        if self.document_reranker:
            data["document_reranker"] = self.document_reranker.to_dict(**kwargs)
        return data

    def format_content(
        self,
        documents: list[Document],
        metadata_fields: list[str] | None = None,
    ) -> str:
        """Format the retrieved documents' metadata and content.

        Args:
            documents (list[Document]): List of retrieved documents.
            metadata_fields (list[str]): Metadata fields to include. If None, uses all metadata except embeddings.

        Returns:
            str: Formatted content of the documents.
        """
        formatted_docs: list[str] = []

        requested_metadata_fields, include_score = self._normalize_metadata_fields(metadata_fields)

        for index, doc in enumerate(documents):
            metadata = doc.metadata or {}
            metadata_lines: list[str] = []

            if (requested_metadata_fields is None or include_score) and doc.score is not None:
                metadata_lines.append(self._format_metadata_line("Score", self._format_score(doc.score)))

            metadata_lines.extend(self._generate_metadata_lines(metadata, requested_metadata_fields))

            metadata_block = "\n".join(metadata_lines) if metadata_lines else "No metadata available."
            content_block = (doc.content or "").strip()

            formatted_doc = (
                f"--- Retrieved Source {index + 1} ---\n"
                f"Metadata:\n{metadata_block}\n\n"
                f"Content:\n{content_block}\n"
                f"--- End Source {index + 1} ---"
            ).rstrip()
            formatted_docs.append(formatted_doc)

        return "\n\n".join(formatted_docs)

    def _format_score(self, score: Any) -> Any:
        if not isinstance(score, (float, int)):
            return score
        return round(score, self._METADATA_SCORE_PRECISION)

    @staticmethod
    def _normalize_metadata_fields(metadata_fields: list[str] | None) -> tuple[list[str] | None, bool]:
        if metadata_fields is None:
            return None, False

        include_score = False
        seen_fields: set[str] = set()
        cleaned_fields: list[str] = []
        for field in metadata_fields:
            stripped = field.strip() if field else ""
            if not stripped:
                continue

            lowered = stripped.lower()
            if lowered in seen_fields:
                continue

            seen_fields.add(lowered)
            if lowered == "score":
                include_score = True
            else:
                cleaned_fields.append(stripped)

        return cleaned_fields, include_score

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

        if self.skip_empty_metadata and self._is_empty_metadata_value(value):
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
        base_entries.sort(key=lambda entry: not self._contains_expected_keyword(entry[1]))
        return [
            self._format_metadata_line(" - ".join(display_parts), value) for display_parts, _, value in base_entries
        ]

    @staticmethod
    def _is_empty_metadata_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    def _resolve_formatted_metadata_fields(self) -> list[str] | None:
        if self.is_optimized_for_agents:
            return self.agent_metadata_fields
        return None

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
        self,
        input_data: VectorStoreRetrieverInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
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
        max_vector_distance = (
            input_data.max_vector_distance if input_data.max_vector_distance is not None else self.max_vector_distance
        )

        alpha = input_data.alpha if input_data.alpha is not None else self.alpha
        query = input_data.query
        try:
            kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
            kwargs.pop("run_depends", None)
            check_cancellation(config)
            text_embedder_output = self.text_embedder.run(
                input_data={"query": query},
                run_depends=self._run_depends,
                config=config,
                **kwargs,
            )
            if text_embedder_output.status != RunnableStatus.SUCCESS:
                error = text_embedder_output.error.message if text_embedder_output.error else "unknown error"
                raise RuntimeError(f"Text embedder failed: {error}")
            self._run_depends = [NodeDependency(node=self.text_embedder).to_dict(for_tracing=True)]
            embedding = text_embedder_output.output.get("embedding")

            check_cancellation(config)
            document_retriever_output = self.document_retriever.run(
                input_data={
                    "embedding": embedding,
                    **({"top_k": top_k} if top_k else {}),
                    "filters": filters,
                    "alpha": alpha,
                    **({"query": query} if query is not None else {}),
                    **({"similarity_threshold": similarity_threshold} if similarity_threshold is not None else {}),
                    **({"max_vector_distance": max_vector_distance} if max_vector_distance is not None else {}),
                },
                run_depends=self._run_depends,
                config=config,
                **kwargs,
            )
            if document_retriever_output.status != RunnableStatus.SUCCESS:
                error = document_retriever_output.error.message if document_retriever_output.error else "unknown error"
                raise RuntimeError(f"Document retriever failed: {error}")
            self._run_depends = [NodeDependency(node=self.document_retriever).to_dict(for_tracing=True)]
            retrieved_documents = document_retriever_output.output.get("documents", [])
            logger.info(f"Tool {self.name} - {self.id}: retrieved {len(retrieved_documents)} documents")

            if self.document_reranker and retrieved_documents:
                docs_before_rerank = len(retrieved_documents)
                logger.info(
                    f"Tool {self.name} - {self.id}: Applying document_reranker '{self.document_reranker.name}' "
                    f"to {docs_before_rerank} documents"
                )
                check_cancellation(config)
                document_reranker_result = self.document_reranker.run(
                    input_data={"query": query, "documents": retrieved_documents},
                    run_depends=self._run_depends,
                    config=config,
                    **kwargs,
                )
                if document_reranker_result.status != RunnableStatus.SUCCESS:
                    error = (
                        document_reranker_result.error.message if document_reranker_result.error else "unknown error"
                    )
                    raise RuntimeError(f"Document reranker failed: {error}")
                self._run_depends = [NodeDependency(node=self.document_reranker).to_dict(for_tracing=True)]
                retrieved_documents = document_reranker_result.output.get("documents", [])
                logger.info(
                    f"Tool {self.name} - {self.id}: Document_reranker finished. "
                    f"Documents: {docs_before_rerank} -> {len(retrieved_documents)}"
                )

            result = self.format_content(
                retrieved_documents,
                metadata_fields=self._resolve_formatted_metadata_fields(),
            )
            logger.info(f"Tool {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")

            return {"content": result, "documents": retrieved_documents}
        except Exception as e:
            logger.error(
                f"Tool {self.name} - {self.id}: execution error: {str(e)}",
                exc_info=True,
            )
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve data using the specified action. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )
