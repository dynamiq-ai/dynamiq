from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.embedders.base import BaseEmbedder, InvalidEmbeddingError
from dynamiq.components.splitters.semantic import BreakpointThresholdType, SemanticSplitterComponent
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.embedders.base import TextEmbedder
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class SemanticSplitterInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Documents to split semantically.")


class SemanticSplitter(Node):
    """Semantic-similarity splitter.

    Splits text where consecutive sentence-group embeddings diverge above a
    configurable threshold. Re-uses any dynamiq :class:`TextEmbedder` node to
    produce embeddings.
    """

    group: Literal[NodeGroup.SPLITTERS] = NodeGroup.SPLITTERS
    name: str = "SemanticSplitter"
    description: str = "Splits text on semantic-similarity breakpoints."

    embedder: TextEmbedder | None = Field(default=None, description="Text embedder node used to embed sentence groups.")
    breakpoint_threshold_type: BreakpointThresholdType = Field(
        default=BreakpointThresholdType.PERCENTILE,
        description="Statistical method to compute breakpoint threshold.",
    )
    breakpoint_threshold_amount: float | None = Field(
        default=None,
        description="Threshold amount; defaults depend on threshold type.",
    )
    number_of_chunks: int | None = Field(
        default=None,
        description="If set, picks top-N largest distances as breakpoints (overrides threshold).",
    )
    buffer_size: int = Field(
        default=1,
        ge=0,
        description="Number of neighbour sentences in each embedding group.",
    )
    sentence_split_regex: str = Field(
        default=r"(?<=[.?!])\s+",
        description="Regex used to split text into sentences before grouping.",
    )
    min_chunk_size: int = Field(
        default=0,
        ge=0,
        description="Merge tail chunks shorter than this back into the previous chunk.",
    )

    splitter: Any | None = None
    input_schema: ClassVar[type[BaseModel]] = SemanticSplitterInputSchema

    @property
    def to_dict_exclude_params(self) -> dict[str, Any]:
        return super().to_dict_exclude_params | {"splitter": True, "embedder": True}

    def to_dict(
        self,
        include_secure_params: bool = False,
        for_tracing: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        data = super().to_dict(
            include_secure_params=include_secure_params,
            for_tracing=for_tracing,
            **kwargs,
        )
        if self.embedder is not None:
            data["embedder"] = self.embedder.to_dict(
                include_secure_params=include_secure_params,
                for_tracing=for_tracing,
                **kwargs,
            )
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.embedder is None:
            raise ValueError("SemanticSplitter requires an `embedder` (TextEmbedder) node.")
        self.embedder.init_components(connection_manager)
        if self.splitter is None:
            self.splitter = SemanticSplitterComponent(
                embed_fn=self._embed_batch,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_threshold_amount,
                number_of_chunks=self.number_of_chunks,
                buffer_size=self.buffer_size,
                sentence_split_regex=self.sentence_split_regex,
                min_chunk_size=self.min_chunk_size,
            )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.embedder is None or self.embedder.text_embedder is None:
            raise ValueError("SemanticSplitter requires an initialized `embedder`.")

        component = self.embedder.text_embedder
        texts_to_embed = [
            component._apply_text_truncation(f"{component.prefix}{text}{component.suffix}".replace("\n", " "))
            for text in texts
        ]
        embeddings, _ = component._embed_texts_batch(
            texts_to_embed=texts_to_embed,
            batch_size=component.batch_size,
        )
        try:
            for embedding in embeddings:
                BaseEmbedder.validate_embedding(embedding)
        except InvalidEmbeddingError as e:
            logger.error(f"Invalid embedding returned by model {component.model}: {str(e)}")
            raise ValueError(f"Invalid embedding returned by the model: {str(e)}")
        return embeddings

    def execute(
        self,
        input_data: SemanticSplitterInputSchema,
        config: RunnableConfig = None,
        **kwargs,
    ) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        documents = input_data.documents
        logger.debug(f"SemanticSplitter: splitting {len(documents)} documents.")
        output = self.splitter.run(documents=documents)
        return {"documents": output["documents"]}
