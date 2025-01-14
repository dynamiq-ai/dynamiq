"""Document store functionality for Elasticsearch."""

from typing import Any

from elasticsearch import NotFoundError
from elasticsearch.helpers import bulk

from dynamiq.types import Document
from dynamiq.utils.logger import logger


class ElasticsearchDocumentFeatures:
    """Document store features for Elasticsearch."""

    def update_document(
        self,
        id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Update an existing document.

        Args:
            id: Document ID
            content: New content (optional)
            metadata: New metadata (optional)
            embedding: New embedding vector (optional)
        """
        update_fields = {}
        if content is not None:
            update_fields[self.content_key] = content
        if metadata is not None:
            update_fields["metadata"] = metadata
        if embedding is not None:
            update_fields[self.embedding_key] = embedding

        if update_fields:
            self.client.update(index=self.index_name, id=id, body={"doc": update_fields}, refresh=True)

    def update_documents_batch(
        self,
        documents: list[Document],
        batch_size: int | None = None,
    ) -> int:
        """Update multiple documents in batches.

        Args:
            documents: List of documents to update
            batch_size: Size of batches for bulk operations

        Returns:
            Number of documents updated
        """
        batch_size = batch_size or self.write_batch_size
        total_updated = 0

        def generate_actions():
            for doc in documents:
                yield {
                    "_op_type": "update",
                    "_index": self.index_name,
                    "_id": doc.id,
                    "doc": {
                        self.content_key: doc.content,
                        "metadata": doc.metadata,
                        self.embedding_key: doc.embedding,
                    },
                }

        for i in range(0, len(documents), batch_size):
            batch = list(generate_actions())[i : i + batch_size]
            success, failed = bulk(self.client, batch, refresh=True, raise_on_error=True)
            total_updated += success

        return total_updated

    def create_alias(
        self,
        alias_name: str,
        index_names: list[str] | None = None,
    ) -> None:
        """Create an alias for one or more indices.

        Args:
            alias_name: Name of the alias
            index_names: List of indices to include in alias (defaults to current index)
        """
        index_names = index_names or [self.index_name]
        actions = []
        for index in index_names:
            actions.append({"add": {"index": index, "alias": alias_name}})
        self.client.indices.update_aliases({"actions": actions})

    def update_mapping(
        self,
        mapping: dict[str, Any],
        update_if_exists: bool = True,
    ) -> None:
        """Update the index mapping.

        Args:
            mapping: New mapping configuration
            update_if_exists: Whether to update existing fields
        """
        try:
            self.client.indices.put_mapping(
                index=self.index_name,
                body=mapping,
                ignore_conflicts=not update_if_exists,
            )
        except Exception as e:
            logger.error(f"Failed to update mapping: {str(e)}")
            raise

    def update_settings(
        self,
        settings: dict[str, Any],
    ) -> None:
        """Update the index settings.

        Args:
            settings: New settings configuration
        """
        try:
            # Close index before updating settings
            self.client.indices.close(index=self.index_name)

            # Update settings
            self.client.indices.put_settings(index=self.index_name, body=settings)

            # Reopen index
            self.client.indices.open(index=self.index_name)

        except Exception as e:
            # Ensure index is reopened even if update fails
            try:
                self.client.indices.open(index=self.index_name)
            except Exception as open_error:
                logger.warning(f"Failed to open index: {str(open_error)}")
            logger.error(f"Failed to update settings: {str(e)}")
            raise

    def get_document_by_id(
        self,
        id: str,
        include_embedding: bool = False,
    ) -> Document | None:
        """Get a single document by ID.

        Args:
            id: Document ID
            include_embedding: Whether to include the embedding vector

        Returns:
            Document if found, None otherwise
        """
        try:
            response = self.client.get(
                index=self.index_name,
                id=id,
                _source_excludes=([self.embedding_key] if not include_embedding else None),
            )

            return Document(
                id=response["_id"],
                content=response["_source"][self.content_key],
                metadata=response["_source"]["metadata"],
                embedding=response["_source"].get(self.embedding_key),
            )

        except NotFoundError:
            return None

    def get_document_count(
        self,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Get the number of documents in the index.

        Args:
            filters: Optional metadata filters

        Returns:
            Number of documents
        """
        query = {"match_all": {}}
        if filters:
            query = {"bool": {"must": [{"match": {f"metadata.{k}": v}} for k, v in filters.items()]}}

        return self.client.count(index=self.index_name, query=query)["count"]

    def get_field_statistics(
        self,
        field: str,
    ) -> dict[str, Any]:
        """Get statistics for a numeric field.

        Args:
            field: Field name (must be numeric)

        Returns:
            Dictionary with min, max, avg, sum
        """
        response = self.client.search(
            index=self.index_name,
            body={"size": 0, "aggs": {"stats": {"stats": {"field": field}}}},
        )

        return response["aggregations"]["stats"]
