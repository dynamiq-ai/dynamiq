import asyncio
import sys
import types

import pytest

from dynamiq.connections import CuaDesktop as CuaDesktopConnection
from dynamiq.nodes.tools.cua_desktop.cua_desktop import CuaDesktopTool


class FakeComputer:
    """Stands in for cua-computer's Computer. Its stop() is a plain function returning a coroutine, marked with
    asyncio's _is_coroutine flag the way legacy coroutine wrappers and Cython-compiled `async def` methods are."""

    def __init__(self, **kwargs):
        self.stopped = []

        async def stopping():
            self.stopped.append("awaited")

        def stop():
            return stopping()

        stop._is_coroutine = asyncio.coroutines._is_coroutine
        self.stop = stop


@pytest.fixture
def computer_module(monkeypatch):
    """cua-computer is an optional extra; the connection imports `computer.Computer` when it connects."""
    module = types.ModuleType("computer")
    module.Computer = FakeComputer
    monkeypatch.setitem(sys.modules, "computer", module)


def test_close_awaits_a_stop_method_marked_as_a_coroutine(computer_module):
    tool = CuaDesktopTool(connection=CuaDesktopConnection(api_key="test-key"))
    computer = FakeComputer()
    tool._computer = computer

    tool.close()

    assert computer.stopped == ["awaited"]
