from pydantic import BaseModel


class BaseVectorStoreParams(BaseModel):
    """Base parameters for vector store.

    Attributes:
        index_name (str): Name of the index. Defaults to "default".
    """
    index_name: str = "default"
    content_key: str = "content"


class BaseWriterVectorStoreParams(BaseVectorStoreParams):
    """Parameters for writer vector store.

    Attributes:
        create_if_not_exist (bool): Flag to create index if it does not exist. Defaults to True.
    """

    create_if_not_exist: bool = False
