import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from dynamiq.cache.utils import cache_wf_entity_async


class TestCacheWfEntityAsync:
    @pytest.mark.asyncio
    async def test_cache_miss_calls_async_func(self):
        """On cache miss, the async wrapper should await the wrapped coroutine."""
        async def my_async_func(*args, **kwargs):
            return {"result": "computed"}

        cache = cache_wf_entity_async(
            entity_id="node-1",
            cache_enabled=False,
        )
        wrapped = cache(my_async_func)
        output, from_cache = await wrapped({"key": "val"}, config=None)

        assert output == {"result": "computed"}
        assert from_cache is False

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self):
        """On cache hit, should return cached output without calling the function."""
        mock_func = AsyncMock(return_value={"result": "computed"})

        mock_cache_manager = MagicMock()
        mock_cache_manager.get_entity_output.return_value = {"result": "cached"}

        with patch("dynamiq.cache.utils.WorkflowCacheManager", return_value=mock_cache_manager):
            cache = cache_wf_entity_async(
                entity_id="node-1",
                cache_enabled=True,
                cache_config=MagicMock(),
            )
            wrapped = cache(mock_func)
            output, from_cache = await wrapped({"key": "val"}, config=None)

        assert output == {"result": "cached"}
        assert from_cache is True
        mock_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_stores_result(self):
        """On cache miss with caching enabled, should store the result."""
        async def my_async_func(*args, **kwargs):
            return {"result": "computed"}

        mock_cache_manager = MagicMock()
        mock_cache_manager.get_entity_output.return_value = None

        with patch("dynamiq.cache.utils.WorkflowCacheManager", return_value=mock_cache_manager):
            cache = cache_wf_entity_async(
                entity_id="node-1",
                cache_enabled=True,
                cache_config=MagicMock(),
            )
            wrapped = cache(my_async_func)
            output, from_cache = await wrapped({"key": "val"}, config=None)

        assert output == {"result": "computed"}
        assert from_cache is False
        mock_cache_manager.set_entity_output.assert_called_once()
