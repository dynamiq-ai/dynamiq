import hashlib
from typing import Any

from dynamiq.cache.config import CacheConfig
from dynamiq.cache.managers import CacheManager


class WorkflowCacheManager(CacheManager):
    """Manager for caching workflow entity outputs.

    Attributes:
        config (CacheConfig): Cache configuration.
        serializer (Any): Serializer instance.
    """

    def __init__(
        self,
        config: CacheConfig,
        serializer: Any | None = None,
    ):
        """Initialize WorkflowCacheManager.

        Args:
            config (CacheConfig): Cache configuration.
            serializer (Any | None): Serializer instance.
        """
        super().__init__(
            config=config,
            serializer=serializer,
        )

    def get_entity_output(self, entity_id: str, input_data: Any) -> Any:
        """Retrieve cached entity output.

        Args:
            entity_id (str): Entity identifier.
            input_data (Any): Input data for the entity.

        Returns:
            Any: Cached output data.
        """
        key = self.get_key(entity_id=entity_id, input_data=input_data)
        return super().get(key=key)

    def set_entity_output(
        self, entity_id: str, input_data: Any, output_data: Any
    ) -> Any:
        """Cache entity output.

        Args:
            entity_id (str): Entity identifier.
            input_data (Any): Input data for the entity.
            output_data (Any): Output data to cache.

        Returns:
            Any: Result of cache set operation.
        """
        key = self.get_key(entity_id=entity_id, input_data=input_data)
        return super().set(key=key, value=output_data)

    def delete_entity_output(self, entity_id: str, input_data: Any) -> Any:
        """Delete cached entity output.

        Args:
            entity_id (str): Entity identifier.
            input_data (Any): Input data for the entity.

        Returns:
            Any: Result of cache delete operation.
        """
        key = self.get_key(entity_id=entity_id, input_data=input_data)
        return super().delete(key=key)

    def get_key(self, entity_id: str, input_data: Any) -> str:
        """Generate cache key for entity.

        Args:
            entity_id (str): Entity identifier.
            input_data (Any): Input data for the entity.

        Returns:
            str: Generated cache key.
        """
        input_data_hash = self.hash(self.serializer.dumps(input_data))
        return f"{entity_id}:{input_data_hash}"

    @staticmethod
    def hash(data: str) -> str:
        """Generate SHA-256 hash of data.

        Args:
            data (str): Data to hash.

        Returns:
            str: SHA-256 hash.
        """
        return hashlib.sha256(data.encode()).hexdigest()
