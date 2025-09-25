from dynamiq.types.dry_run import DryRunConfig
from dynamiq.utils.logger import logger


class DryRunMixin:
    """Mixin class to add dry run functionality to vector stores.

    This mixin provides resource tracking and cleanup capabilities for vector stores
    operating in dry run mode. It tracks ingested documents and created collections
    to enable cleanup based on the DryRunConfig settings.
    """

    def __init__(self, dry_run_config: DryRunConfig | None = None):
        """Initialize the DryRunMixin.

        Args:
            dry_run_config: Configuration for dry run behavior. If None, default config is used.
        """
        self._dry_run_config = dry_run_config or DryRunConfig()
        self._tracked_documents: list[str] = []
        self._tracked_collection: str | None = None

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """Delete documents by their IDs.

        Args:
            document_ids: List of document IDs to delete.
            delete_all: Whether to delete all documents.
        """
        pass

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection by its name.

        Args:
            collection_name: Name of the collection to delete.
        """
        pass

    def _track_documents(self, document_ids: list[str]) -> None:
        """Track multiple documents for potential cleanup.

        Args:
            document_ids: List of document IDs to track.
        """
        self._tracked_documents.extend(document_ids)
        logger.debug(f"Tracked {len(document_ids)} documents")

    def _track_collection(self, collection_name: str) -> None:
        """Track a collection for potential cleanup.

        Args:
            collection_name: Name of the collection to track.
        """
        self._tracked_collection = collection_name
        logger.debug(f"Tracked collection: {collection_name}")

    def dry_run_cleanup(self, dry_run_config: DryRunConfig) -> None:
        """Clean up tracked resources based on configuration.

        Args:
            dry_run_config: Configuration for dry run behavior.
        """

        if dry_run_config.delete_documents and self._tracked_documents:
            try:
                self.delete_documents(list(self._tracked_documents))
                logger.debug(f"Cleaned up {len(self._tracked_documents)} tracked documents")
                self._tracked_documents = []
            except Exception as e:
                logger.error(f"Failed to clean up tracked documents: {e}")

        if dry_run_config.delete_collection and self._tracked_collection:
            try:
                self.delete_collection(self._tracked_collection)
                logger.debug(f"Cleaned up collection: {self._tracked_collection}")
                self._tracked_collection = None
            except Exception as e:
                logger.error(f"Failed to clean up collection {self._tracked_collection}: {e}")
