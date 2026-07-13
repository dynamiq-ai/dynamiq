from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from dynamiq.storages.vector.utils import create_file_id_filter, create_file_ids_filter
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.types import Document


class BaseVectorStoreParams(BaseModel):
    """Base parameters for vector store.

    Attributes:
        index_name (str): Name of the index. Defaults to "default".
        content_key (str): Key for content field. Defaults to "content".
    """
    index_name: str = "default"
    content_key: str = "content"


class BaseWriterVectorStoreParams(BaseVectorStoreParams):
    """Parameters for writer vector store.

    Attributes:
        create_if_not_exist (bool): Flag to create index if it does not exist. Defaults to True.
    """

    create_if_not_exist: bool = False


class BaseVectorStore(ABC):
    """Base class for all vector stores.

    This abstract class provides a consistent interface for all vector store implementations,
    including common methods for document deletion by file ID(s).
    """

    def get_documents_by_id(self, ids: list[str]) -> list["Document"]:
        """Fetch documents by their exact ids (not a similarity search).

        Backends that can address points by id override this; the rest inherit this default so callers
        can rely on a single contract.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support fetch-by-id.")

    @staticmethod
    def _normalize_document_ids(document_ids: "str | list[str]") -> list[str]:
        """Normalize a single id or a list of ids into a de-duplicated (order-preserving) list."""
        ids = [document_ids] if isinstance(document_ids, str) else list(document_ids)
        seen: set[str] = set()
        return [i for i in ids if not (i in seen or seen.add(i))]

    def replace_document_metadata(self, document_ids: "str | list[str]", metadata: dict[str, Any]) -> None:
        """Replace the metadata of one or more documents, identified by their ids.

        Each targeted document's existing metadata is fully replaced with ``metadata`` (not
        merged); content and embedding are preserved. ``document_ids`` may be a single id or a
        list of ids, in which case every listed document receives the same ``metadata``. Every
        backend overrides this with its native implementation.

        Args:
            document_ids (str | list[str]): The id, or list of ids, of the documents to update.
            metadata (dict[str, Any]): The new metadata to store in place of the existing one.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support replace_document_metadata.")

    @abstractmethod
    def delete_documents_by_filters(self, filters: dict[str, Any]) -> None:
        """
        Delete documents from the vector store based on the provided filters.

        Args:
            filters (dict[str, Any]): Filters to select documents to delete.
        """
        pass

    def delete_documents_by_file_id(self, file_id: str) -> None:
        """
        Delete documents from the vector store based on the provided file ID.
        File ID should be located in the metadata of the document.

        Args:
            file_id (str): The file ID to filter by.
        """
        filters = create_file_id_filter(file_id)
        self.delete_documents_by_filters(filters)

    def delete_documents_by_file_ids(self, file_ids: list[str], batch_size: int = 500) -> None:
        """
        Delete documents from the vector store based on the provided list of file IDs.
        File IDs should be located in the metadata of the documents.

        Args:
            file_ids (list[str]): The list of file IDs to filter by.
            batch_size (int): Maximum number of file IDs to process in a single batch. Defaults to 500.
        """
        if not file_ids:
            logger.warning("No file IDs provided. No documents will be deleted.")
            return

        if len(file_ids) > batch_size:
            for i in range(0, len(file_ids), batch_size):
                batch = file_ids[i : i + batch_size]
                filters = create_file_ids_filter(batch)
                self.delete_documents_by_filters(filters)
                logger.debug(f"Deleted documents batch {i//batch_size + 1} with {len(batch)} file IDs")
        else:
            filters = create_file_ids_filter(file_ids)
            self.delete_documents_by_filters(filters)
