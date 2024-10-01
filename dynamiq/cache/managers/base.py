from typing import Any, Callable

from dynamiq.cache.backends import BaseCache, RedisCache
from dynamiq.cache.codecs import Base64Codec
from dynamiq.cache.config import CacheBackend, CacheConfig
from dynamiq.components.serializers import JsonSerializer


class CacheManager:
    """Manager for handling cache operations.

    Attributes:
        CACHE_BACKENDS_BY_TYPE (dict[CacheBackend, BaseCache]): Mapping of backends.
        cache_backend (BaseCache): Selected cache backend.
        cache (BaseCache): Cache instance.
        serializer (Any): Serializer instance.
        codec (Any): Codec instance.
        namespace (str | None): Cache namespace.
        ttl (int | None): Time-to-live for cache entries.
    """
    CACHE_BACKENDS_BY_TYPE: dict[CacheBackend, BaseCache] = {
        CacheBackend.Redis: RedisCache,
    }

    def __init__(
        self,
        config: CacheConfig,
        serializer: Any | None = None,
        codec: Any | None = None,
    ):
        """Initialize CacheManager.

        Args:
            config (CacheConfig): Cache configuration.
            serializer (Any | None): Serializer instance.
            codec (Any | None): Codec instance.
        """
        self.cache_backend = self.CACHE_BACKENDS_BY_TYPE.get(config.backend)
        self.cache = self.cache_backend.from_config(config)
        self.serializer = serializer or JsonSerializer()
        self.codec = codec or Base64Codec()
        self.namespace = config.namespace
        self.ttl = config.ttl

    def get(
        self,
        key: str,
        namespace: str | None = None,
        loads_func: Callable[[Any], Any] | None = None,
        decode_func: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Retrieve value from cache.

        Args:
            key (str): Cache key.
            namespace (str | None): Cache namespace.
            loads_func (Callable[[Any], Any] | None): Function to deserialize.
            decode_func (Callable[[Any], Any] | None): Function to decode.

        Returns:
            Any: Cached value.
        """
        loads = loads_func or self.serializer.loads
        decode = decode_func or self.codec.decode
        ns_key = self._get_key(key, namespace=self._get_namespace(namespace))

        if (res := self.cache.get(ns_key)) is not None:
            res = loads(decode(self.cache.get(key=ns_key)))

        return res

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        namespace: str | None = None,
        dumps_func: Callable[[Any], Any] | None = None,
        encode_func: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Set value in cache.

        Args:
            key (str): Cache key.
            value (Any): Value to cache.
            ttl (int | None): Time-to-live for cache entry.
            namespace (str | None): Cache namespace.
            dumps_func (Callable[[Any], Any] | None): Function to serialize.
            encode_func (Callable[[Any], Any] | None): Function to encode.

        Returns:
            Any: Result of cache set operation.
        """
        dumps = dumps_func or self.serializer.dumps
        encode = encode_func or self.codec.encode
        ns_key = self._get_key(key, namespace=self._get_namespace(namespace))
        ttl = ttl or self.ttl

        res = self.cache.set(key=ns_key, value=encode(dumps(value)), ttl=ttl)

        return res

    def delete(
        self,
        key: str,
        namespace: str | None = None,
    ) -> Any:
        """Delete value from cache.

        Args:
            key (str): Cache key.
            namespace (str | None): Cache namespace.

        Returns:
            Any: Result of cache delete operation.
        """
        ns_key = self._get_key(key, namespace=self._get_namespace(namespace))
        res = self.cache.delete(ns_key)

        return res

    def _get_namespace(self, namespace: str | None = None) -> str | None:
        """Get effective namespace.

        Args:
            namespace (str | None): Provided namespace.

        Returns:
            str | None: Effective namespace.
        """
        return namespace if namespace is not None else self.namespace

    @staticmethod
    def _get_key(key: str, namespace: str | None = None) -> str:
        """Construct cache key with namespace.

        Args:
            key (str): Cache key.
            namespace (str | None): Cache namespace.

        Returns:
            str: Namespaced cache key.
        """
        if namespace is not None:
            return f"{namespace}:{key}"
        return key
