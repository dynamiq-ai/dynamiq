from functools import wraps
from typing import Any, Callable

from dynamiq.cache import CacheConfig
from dynamiq.cache.managers import WorkflowCacheManager
from dynamiq.utils.logger import logger


def cache_wf_entity(
    entity_id: str,
    cache_enabled: bool = False,
    cache_manager_cls: type[WorkflowCacheManager] = WorkflowCacheManager,
    cache_config: CacheConfig | None = None,
) -> Callable:
    """Decorator to cache workflow entity outputs.

    Args:
        entity_id (str): Identifier for the entity.
        cache_enabled (bool): Flag to enable caching.
        cache_manager_cls (type[WorkflowCacheManager]): Cache manager class.
        cache_config (CacheConfig | None): Cache configuration.

    Returns:
        Callable: Wrapped function with caching.
    """
    def _cache(func: Callable) -> Callable:
        """Inner cache decorator.

        Args:
            func (Callable): Function to wrap.

        Returns:
            Callable: Wrapped function.
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, bool]:
            """Wrapper function to handle caching.

            Args:
                *args (Any): Positional arguments.
                **kwargs (Any): Keyword arguments.

            Returns:
                tuple[Any, bool]: Function output and cache status.
            """
            cache_manager = None
            from_cache = False
            input_data = kwargs.get("input_data", args[0] if args else None)
            if cache_enabled and cache_config:
                logger.debug(f"Entity_id {entity_id}: cache used")
                cache_manager = cache_manager_cls(config=cache_config)
                if output := cache_manager.get_entity_output(entity_id, input_data):
                    from_cache = True
                    return output, from_cache

            output = func(*args, **kwargs)

            if cache_manager:
                cache_manager.set_entity_output(entity_id, input_data, output)

            return output, from_cache

        return wrapper

    return _cache
