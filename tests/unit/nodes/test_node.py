import asyncio

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


@pytest.fixture
def node_sync_result():
    return RunnableResult(
        status=RunnableStatus.SUCCESS, input={"input": "test_input"}, output={"output": "sync_output"}
    )


@pytest.fixture
def node_async_result():
    return RunnableResult(
        status=RunnableStatus.SUCCESS, input={"input": "test_input"}, output={"output": "async_output"}
    )


@pytest.fixture
def openai_node(mocker, node_sync_result, node_async_result):
    mocker.patch("dynamiq.nodes.llms.base.BaseLLM.run_sync", return_value=node_sync_result)
    mocker.patch("dynamiq.nodes.node.Node.run_async", return_value=node_async_result)
    yield OpenAI(model="gpt-4", connection=OpenAIConnection(api_key="test_api_key"))


def test_run_in_sync_context(openai_node, node_sync_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    result = openai_node.run(input_data, config)

    openai_node.run_sync.assert_called_once_with(input_data, config)
    openai_node.run_async.assert_not_called()
    assert result == node_sync_result


def test_run_in_sync_runtime_and_async_context(openai_node, node_async_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    async def run_async(*args, **kwargs):
        return await openai_node.run(*args, **kwargs)

    result = asyncio.run(run_async(input_data, config))

    openai_node.run_async.assert_called_once_with(input_data, config)
    openai_node.run_sync.assert_not_called()
    assert result == node_async_result


@pytest.mark.asyncio
async def test_run_in_async_context(openai_node, node_async_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    result = await openai_node.run(input_data, config)

    openai_node.run_async.assert_called_once_with(input_data, config)
    openai_node.run_sync.assert_not_called()
    assert result == node_async_result


@pytest.mark.asyncio
async def test_run_in_async_runtime_and_sync_context(openai_node, node_sync_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    result = await asyncio.to_thread(openai_node.run, input_data, config)

    openai_node.run_sync.assert_called_once_with(input_data, config)
    openai_node.run_async.assert_not_called()
    assert result == node_sync_result


def test_run_with_explicit_sync_flag(openai_node, node_sync_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    result = openai_node.run(input_data, config, is_async=False)

    openai_node.run_sync.assert_called_once_with(input_data, config)
    openai_node.run_async.assert_not_called()
    assert result == node_sync_result


@pytest.mark.asyncio
async def test_run_with_explicit_async_flag(openai_node, node_async_result):
    input_data = {"input": "test_input"}
    config = RunnableConfig()

    result = await openai_node.run(input_data, config, is_async=True)

    openai_node.run_async.assert_called_once_with(input_data, config)
    openai_node.run_sync.assert_not_called()
    assert result == node_async_result
