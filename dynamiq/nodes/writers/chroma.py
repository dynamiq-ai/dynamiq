from dynamiq.connections import Chroma
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.writers.base import Writer, WriterInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import ChromaVectorStore
from dynamiq.storages.vector.base import BaseWriterVectorStoreParams


class ChromaDocumentWriter(Writer, BaseWriterVectorStoreParams):
    """
    Document Writer Node using Chroma Vector Store.

    This class represents a node for writing documents to a Chroma Vector Store.

    Attributes:
        group (Literal[NodeGroup.WRITERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (Chroma | None): The connection to the Chroma Vector Store.
        vector_store (ChromaVectorStore | None): The Chroma Vector Store instance.
    """

    name: str = "ChromaDocumentWriter"
    connection: Chroma | None = None
    vector_store: ChromaVectorStore | None = None

    def __init__(self, **kwargs):
        """
        Initialize the ChromaDocumentWriter.

        If neither vector_store nor connection is provided in kwargs, a default Chroma connection will be created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Chroma()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return ChromaVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include={"index_name", "create_if_not_exist"}) | {
            "connection": self.connection,
            "client": self.client,
        }

    def execute(self, input_data: WriterInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Execute the document writing operation.

        This method writes the documents provided in the input_data to the Chroma Vector Store.

        Args:
            input_data (WriterInputSchema): An instance containing the input data.
                Expected to have a 'documents' key with the documents to be written.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the count of upserted documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = input_data.documents

        output = self.vector_store.write_documents(documents)
        return {
            "upserted_count": output,
        }
