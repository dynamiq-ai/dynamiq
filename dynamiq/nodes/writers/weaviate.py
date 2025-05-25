from typing import Any

from dynamiq.connections import Weaviate
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.writers.base import Writer, WriterInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import WeaviateVectorStore
from dynamiq.storages.vector.dry_run import DryRunConfig
from dynamiq.storages.vector.dry_run_orchestrator import DryRunOrchestrator
from dynamiq.storages.vector.weaviate import WeaviateWriterVectorStoreParams
from dynamiq.utils.logger import logger


class WeaviateDocumentWriter(Writer, WeaviateWriterVectorStoreParams):
    """
    Document Writer Node using Weaviate Vector Store.

    This class represents a node for writing documents to a Weaviate Vector Store.

    Attributes:
        group (Literal[NodeGroup.WRITERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (Weaviate | None): The Weaviate connection.
        vector_store (WeaviateVectorStore | None): The Weaviate Vector Store instance.
    """

    name: str = "WeaviateDocumentWriter"
    connection: Weaviate | None = None
    vector_store: WeaviateVectorStore | None = None

    def __init__(self, **kwargs):
        """
        Initialize the WeaviateDocumentWriter.

        If neither vector_store nor connection is provided in kwargs, a default Weaviate connection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Weaviate()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return WeaviateVectorStore

    @property
    def vector_store_params(self):
        params = self.model_dump(include=set(WeaviateWriterVectorStoreParams.model_fields))

        if hasattr(self, "dry_run_config") and self.dry_run_config is not None:
            params["dry_run_config"] = self.dry_run_config

        params.update(
            {
                "connection": self.connection,
                "client": self.client,
            }
        )
        return params

    def execute(self, input_data: WriterInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the document writing operation.

        This method writes the input documents to the Weaviate Vector Store,
        with support for dry run operations.

        Args:
            input_data (WriterInputSchema): Input data containing the documents to be written.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the count of upserted documents and dry run results if applicable.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        active_dry_run = input_data.dry_run_config or getattr(self, "dry_run_config", None)

        if active_dry_run:
            return self._execute_dry_run(input_data, active_dry_run, config, **kwargs)
        else:
            return self._execute_normal(input_data, config, **kwargs)

    def _execute_dry_run(
        self, input_data: WriterInputSchema, dry_run_config: DryRunConfig, config: RunnableConfig, **kwargs
    ) -> dict[str, Any]:
        """Execute the operation in dry run mode.

        Args:
            input_data: Input data schema
            dry_run_config: Dry run configuration
            config: Runnable configuration
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Any]: Results including dry run information
        """
        logger.info(f"Executing Weaviate writer in dry run mode: {dry_run_config.mode}")

        orchestrator = DryRunOrchestrator(
            vector_store_cls=self.vector_store_cls,
            vector_store_params=self.vector_store_params,
            dry_run_config=dry_run_config,
        )

        dry_run_result = orchestrator.execute(input_data.documents, input_data.content_key)

        return {
            "dry_run_result": dry_run_result,
            "upserted_count": dry_run_result.documents_processed,
            "success": dry_run_result.success,
            "mode": dry_run_result.mode,
            "test_collection_name": dry_run_result.test_collection_name,
        }

    def _execute_normal(self, input_data: WriterInputSchema, config: RunnableConfig, **kwargs) -> dict[str, Any]:
        """Execute normal (non-dry-run) operation.

        Args:
            input_data: Input data schema
            config: Runnable configuration
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Any]: Standard execution results
        """
        documents = input_data.documents
        content_key = input_data.content_key

        upserted_count = self.vector_store.write_documents(documents, content_key=content_key)
        logger.debug(f"Upserted {upserted_count} documents to Weaviate Vector Store.")

        return {
            "upserted_count": upserted_count,
        }
