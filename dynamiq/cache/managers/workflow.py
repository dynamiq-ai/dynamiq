import hashlib
from typing import Any

from dynamiq.cache.config import CacheConfig
from dynamiq.cache.managers import CacheManager
from dynamiq.utils import format_value


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

    def get_entity_output(self, entity_id: str, input_data: dict, **kwargs) -> Any:
        """Retrieve cached entity output.

        Args:
            entity_id (str): Entity identifier.
            input_data (dict): Input data for the entity.
            kwargs (Any): Additional keyword arguments.

        Returns:
            Any: Cached output data.
        """
        key = self.get_key(entity_id=entity_id, input_data=input_data, **kwargs)
        return super().get(key=key)

    def set_entity_output(self, entity_id: str, input_data: dict, output_data: Any, **kwargs) -> Any:
        """Cache entity output.

        Args:
            entity_id (str): Entity identifier.
            input_data (dict): Input data for the entity.
            output_data (Any): Output data to cache.
            kwargs (Any): Additional keyword arguments.

        Returns:
            Any: Result of cache set operation.
        """
        key = self.get_key(entity_id=entity_id, input_data=input_data, **kwargs)
        return super().set(key=key, value=output_data)

    def delete_entity_output(self, entity_id: str, input_data: dict, **kwargs) -> Any:
        """Delete cached entity output.

        Args:
            entity_id (str): Entity identifier.
            input_data (dict): Input data for the entity.
            kwargs (Any): Additional keyword arguments.

        Returns:
            Any: Result of cache delete operation.
        """
        key = self.get_key(entity_id=entity_id, input_data=input_data, **kwargs)
        return super().delete(key=key)

    def get_key(self, entity_id: str, input_data: dict, **kwargs) -> str:
        """Generate cache key for entity.

        Args:
            entity_id (str): Entity identifier.
            input_data (dict): Input data for the entity.
            kwargs (Any): Additional keyword arguments.

        Returns:
            str: Generated cache key.
        """
        input_data_formatted = format_value(self._sort_dict(input_data))[0]
        input_data_hash = self.hash(self.serializer.dumps(input_data_formatted))
        kwargs_formatted = format_value(self._sort_dict(kwargs))[0]
        kwargs_hash = self.hash(self.serializer.dumps(kwargs_formatted))
        return f"{entity_id}:{input_data_hash}:{kwargs_hash}"

    @staticmethod
    def hash(data: str) -> str:
        """Generate SHA-256 hash of data.

        Args:
            data (str): Data to hash.

        Returns:
            str: SHA-256 hash.
        """
        return hashlib.sha256(data.encode()).hexdigest()

    def _sort_dict(self, d: dict) -> dict:
        """Recursively sort dictionary keys, including nested dictionaries.

        Args:
            d (dict): Dictionary to sort.

        Returns:
            dict: Sorted dictionary.
        """
        return {k: self._sort_dict(v) if isinstance(v, dict) else v for k, v in sorted(d.items())}
