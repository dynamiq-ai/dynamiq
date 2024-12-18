import enum
from typing import Literal
from pydantic import BaseModel

from dynamiq.connections import RedisConnection


class CacheBackend(str, enum.Enum):
    """Enumeration for cache backends."""
    Redis = "Redis"


class CacheConfig(BaseModel):
    """Configuration for cache settings.

    Attributes:
        backend (CacheBackend): The cache backend to use.
        namespace (str | None): Optional namespace for cache keys.
        ttl (int | None): Optional time-to-live for cache entries.
    """
    backend: CacheBackend
    namespace: str | None = None
    ttl: int | None = None

    def to_dict(self, **kwargs) -> dict:
        """Convert config to dictionary.

        Args:
            **kwargs: Additional arguments.

        Returns:
            dict: Configuration as dictionary.
        """
        return self.model_dump(**kwargs)


class RedisCacheConfig(CacheConfig, RedisConnection):
    """Configuration for Redis cache.

    Attributes:
        backend (Literal[CacheBackend.Redis]): The Redis cache backend.
    """
    backend: Literal[CacheBackend.Redis] = CacheBackend.Redis
