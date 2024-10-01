from typing import Any

from dynamiq.cache.backends import BaseCache
from dynamiq.cache.config import RedisCacheConfig


class RedisCache(BaseCache):
    """Redis cache backend implementation."""

    @classmethod
    def from_config(cls, config: RedisCacheConfig):
        """Create RedisCache instance from configuration.

        Args:
            config (RedisCacheConfig): Redis cache configuration.

        Returns:
            RedisCache: Redis cache instance.
        """
        from redis import Redis

        return cls(client=Redis(**config.to_dict()))

    def get(self, key: str) -> Any:
        """Retrieve value from Redis cache.

        Args:
            key (str): Cache key.

        Returns:
            Any: Cached value.
        """
        return self.client.get(key)

    def set(self, key: str, value: dict, ttl: int | None = None) -> Any:
        """Set value in Redis cache.

        Args:
            key (str): Cache key.
            value (dict): Value to cache.
            ttl (int | None): Time-to-live for cache entry.

        Returns:
            Any: Result of cache set operation.
        """
        if ttl is None:
            return self.client.set(key, value)
        return self.client.setex(key, ttl, value)

    def delete(self, key: str) -> Any:
        """Delete value from Redis cache.

        Args:
            key (str): Cache key.

        Returns:
            Any: Result of cache delete operation.
        """
        return self.client.delete(key)
