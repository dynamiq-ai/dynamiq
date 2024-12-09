from abc import ABC, abstractmethod

from dynamiq.types import Document


class DocumentRetriever(ABC):
    """
    Document Retriever.
    """

    @abstractmethod
    def run() -> dict[str, list[Document]]:
        pass
