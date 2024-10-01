from abc import ABC, abstractmethod
from typing import TypeVar

from dynamiq.cache.config import CacheConfig

CacheClient = TypeVar("CacheClient")


class BaseCache(ABC):
    """Abstract base class for cache backends.

    Attributes:
        client (CacheClient): Cache client instance.
    """

    def __init__(self, client: CacheClient):
        """Initialize BaseCache.

        Args:
            client (CacheClient): Cache client instance.
        """
        self.client = client

    @classmethod
    def from_client(cls, client: CacheClient):
        """Create cache instance from client.

        Args:
            client (CacheClient): Cache client instance.

        Returns:
            BaseCache: Cache instance.
        """
        return cls(client=client)

    @classmethod
    @abstractmethod
    def from_config(cls, config: CacheConfig):
        """Create cache instance from configuration.

        Args:
            config (CacheConfig): Cache configuration.

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str):
        """Retrieve value from cache.

        Args:
            key (str): Cache key.

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: dict):
        """Set value in cache.

        Args:
            key (str): Cache key.
            value (dict): Value to cache.

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str):
        """Delete value from cache.

        Args:
            key (str): Cache key.

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError
