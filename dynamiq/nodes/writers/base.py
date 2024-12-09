from abc import ABC
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes.node import NodeGroup, VectorStoreNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector.base import BaseVectorStore
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class WriterInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Parameter to provide documents to write.")


class Writer(VectorStoreNode, ABC):

    group: Literal[NodeGroup.WRITERS] = NodeGroup.WRITERS
    vector_store: BaseVectorStore
    input_schema: ClassVar[type[WriterInputSchema]] = WriterInputSchema

    def execute(self, input_data: WriterInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Execute the document writing operation.

        This method writes the documents provided in the input_data to the Vector Store.

        Args:
            input_data (WriterInputSchema): A dictionary containing the input data.
                Expected to have a 'documents' key with the documents to be written.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the count of upserted documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = input_data.documents

        upserted_count = self.vector_store.write_documents(documents)
        logger.debug(f"Upserted {upserted_count} documents by {self.name}.")
        return {
            "upserted_count": upserted_count,
        }
