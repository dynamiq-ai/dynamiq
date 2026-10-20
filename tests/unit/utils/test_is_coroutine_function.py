import asyncio
import functools
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamiq.utils import is_coroutine_function


async def async_function():
    pass


def sync_function():
    pass


class Client:
    async def aclose(self):
        pass

    def close(self):
        pass


def marked_like_cython():
    """A plain function carrying asyncio's coroutine marker, as Cython-compiled `async def` functions do."""


marked_like_cython._is_coroutine = asyncio.coroutines._is_coroutine


@pytest.mark.parametrize(
    "func",
    [async_function, Client().aclose, functools.partial(async_function), AsyncMock(), marked_like_cython],
    ids=["async def", "bound async method", "partial", "AsyncMock", "asyncio marker"],
)
def test_callables_that_return_a_coroutine_are_recognised(func):
    assert is_coroutine_function(func)


@pytest.mark.parametrize(
    "func", [sync_function, Client().close, MagicMock(), len], ids=["def", "bound method", "MagicMock", "builtin"]
)
def test_callables_that_return_a_value_are_not(func):
    assert not is_coroutine_function(func)
