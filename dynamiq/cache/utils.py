from functools import wraps
from typing import Any, Callable

from pydantic import BaseModel

from dynamiq.cache import CacheConfig
from dynamiq.cache.managers import WorkflowCacheManager
from dynamiq.utils.logger import logger

FUNC_KWARGS_TO_REMOVE = (
    "input_data",
    "config",
    "run_id",
    "parent_run_id",
    "wf_run_id",
)


def cache_wf_entity(
    entity_id: str,
    cache_enabled: bool = False,
    cache_manager_cls: type[WorkflowCacheManager] = WorkflowCacheManager,
    cache_config: CacheConfig | None = None,
    func_kwargs_to_remove: tuple[str] = FUNC_KWARGS_TO_REMOVE,
) -> Callable:
    """Decorator to cache workflow entity outputs.

    Args:
        entity_id (str): Identifier for the entity.
        cache_enabled (bool): Flag to enable caching.
        cache_manager_cls (type[WorkflowCacheManager]): Cache manager class.
        cache_config (CacheConfig | None): Cache configuration.
        func_kwargs_to_remove (tuple[str]): List of params to remove from callable function kwargs.

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
            input_data = kwargs.pop("input_data", args[0] if args else {})
            input_data = dict(input_data) if isinstance(input_data, BaseModel) else input_data

            cleaned_kwargs = {k: v for k, v in kwargs.items() if k not in func_kwargs_to_remove}
            if cache_enabled and cache_config:
                logger.debug(f"Entity_id {entity_id}: cache used")
                cache_manager = cache_manager_cls(config=cache_config)
                if output := cache_manager.get_entity_output(
                    entity_id=entity_id, input_data=input_data, **cleaned_kwargs
                ):
                    from_cache = True
                    return output, from_cache

            output = func(*args, **kwargs)

            if cache_manager:
                cache_manager.set_entity_output(
                    entity_id=entity_id, input_data=input_data, output_data=output, **cleaned_kwargs
                )

            return output, from_cache

        return wrapper

    return _cache
