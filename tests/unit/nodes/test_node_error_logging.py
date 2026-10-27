"""A node's execution error is logged with its traceback, so a failure seen only in a log can be traced."""

import pytest

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node
from dynamiq.runnables import RunnableConfig, RunnableStatus


class _Failing(Node):
    group: NodeGroup = NodeGroup.TOOLS
    name: str = "failing"

    def execute(self, input_data, config=None, **kwargs):
        raise ValueError("boom")

    async def execute_async(self, input_data, config=None, **kwargs):
        raise ValueError("boom")


def _execution_error_calls(logger):
    return [call for call in logger.error.call_args_list if "execution error: boom" in call.args[0]]


def test_sync_execution_error_is_logged_with_traceback(mocker):
    logger = mocker.patch("dynamiq.nodes.node.logger")

    result = _Failing().run(input_data={})

    assert result.status == RunnableStatus.FAILURE
    calls = _execution_error_calls(logger)
    assert len(calls) == 1
    assert calls[0].kwargs == {"exc_info": True}


@pytest.mark.asyncio
async def test_async_execution_error_is_logged_with_traceback(mocker):
    logger = mocker.patch("dynamiq.nodes.node.logger")

    result = await _Failing().run_async(input_data={}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.FAILURE
    calls = _execution_error_calls(logger)
    assert len(calls) == 1
    assert calls[0].kwargs == {"exc_info": True}
