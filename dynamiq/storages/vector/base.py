from abc import ABC, abstractmethod

from pydantic import BaseModel

from dynamiq.types import Document


class BaseVectorStore(ABC):

    @abstractmethod
    def count_documents() -> int:
        pass

    @abstractmethod
    def list_documents() -> list[Document]:
        pass

    @abstractmethod
    def write_documents() -> int:
        pass

    @abstractmethod
    def delete_documents() -> None:
        pass

    @abstractmethod
    def delete_documents_by_filters() -> None:
        pass

    @abstractmethod
    def delete_documents_by_file_id() -> None:
        pass


class BaseVectorStoreParams(BaseModel):
    """Base parameters for vector store.

    Attributes:
        index_name (str): Name of the index. Defaults to "default".
    """
    index_name: str = "default"


class BaseWriterVectorStoreParams(BaseVectorStoreParams):
    """Parameters for writer vector store.

    Attributes:
        create_if_not_exist (bool): Flag to create index if it does not exist. Defaults to True.
    """

    create_if_not_exist: bool = False
