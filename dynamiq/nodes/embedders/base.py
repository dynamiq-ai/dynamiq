from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class DocumentEmbedderInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Parameter to provide documents to find embeddings for.")


class DocumentEmbedder(ConnectionNode):
    group: Literal[NodeGroup.EMBEDDERS] = NodeGroup.EMBEDDERS
    document_embedder: BaseEmbedder | None = None
    input_schema: ClassVar[type[DocumentEmbedderInputSchema]] = DocumentEmbedderInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_embedder": True}

    def execute(self, input_data: DocumentEmbedderInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Executes the document embedding process.

        This method takes input documents, computes their embeddings, and returns the result.

        Args:
            input_data (DocumentEmbedderInputSchema): An instance containing the documents to embed.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            The output from the document_embedder component, typically the computed embeddings.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        output = self.document_embedder.embed_documents(input_data.documents)
        logger.debug(f"{self.name} executed successfully.")

        return output


class TextEmbedderInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide query to find embeddings for.")


class TextEmbedder(ConnectionNode):
    group: Literal[NodeGroup.EMBEDDERS] = NodeGroup.EMBEDDERS
    text_embedder: BaseEmbedder | None = None
    input_schema: ClassVar[type[TextEmbedderInputSchema]] = TextEmbedderInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"text_embedder": True}

    def execute(self, input_data: TextEmbedderInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Execute the text embedding process.

        This method takes text input data, computes its embeddings, and returns the result.

        Args:
            input_data (TextEmbedderInputSchema): The input data containing the query to embed.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the embedding and the original query.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        output = self.text_embedder.embed_text(input_data.query)
        logger.debug(f"BedrockTextEmbedder: {output['meta']}")
        return {
            "embedding": output["embedding"],
            "query": input_data.query,
        }
