from io import BytesIO

import pytest

from dynamiq.cache import RedisCacheConfig
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunType
from dynamiq.nodes import CachingConfig, NodeGroup
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus

PROMPT = Prompt(messages=[Message(content="What is LLM?")])
PROMPT_SYSTEM = Prompt(messages=[Message(content="What is LLM?", role=MessageRole.SYSTEM)])
FILE_CONTENT = b"file content"
FILE = BytesIO(FILE_CONTENT)
FILE_DIFF = BytesIO(FILE_CONTENT + b"_")


@pytest.fixture()
def node_with_caching(openai_node):
    openai_node.caching = CachingConfig(enabled=True)
    return openai_node


@pytest.mark.parametrize(
    ("first_input_data", "first_kwargs", "second_input_data", "second_kwargs", "is_second_use_cache"),
    [
        # # Value of keys should count
        ({"a": 1, "b": 2}, {}, {"a": 1, "b": 2}, {}, True),
        ({"a": 1, "b": 2}, {}, {"b": 2, "a": 2}, {}, False),
        # # Order of keys should not count
        ({"a": 1, "b": 2}, {}, {"b": 2, "a": 1}, {}, True),
        # # FUNC_KWARGS_TO_REMOVE should not count
        ({"a": 1, "b": 2}, {"run_id": 1}, {"a": 1, "b": 2}, {"run_id": 2}, True),
        # # Value of kwargs should count
        ({"a": 1, "b": 2}, {"run_id": 1, "test": 1}, {"a": 1, "b": 2}, {"run_id": 2, "test": 1}, True),
        ({"a": 1, "b": 2}, {"run_id": 1, "test": 1}, {"a": 1, "b": 2}, {"run_id": 2, "test": 2}, False),
        # Different types should count
        ({"a": 1, "files": [FILE]}, {"prompt": PROMPT}, {"a": 1, "files": [FILE]}, {"prompt": PROMPT}, True),
        ({"a": 1, "files": [FILE]}, {"prompt": PROMPT}, {"a": 1, "files": [FILE_DIFF]}, {"prompt": PROMPT}, False),
        ({"a": 1, "files": [FILE]}, {"prompt": PROMPT}, {"a": 1, "files": [FILE]}, {"prompt": PROMPT_SYSTEM}, False),
    ],
)
def test_node_caching(
    node_with_caching,
    mock_redis,
    mock_redis_backend,
    mock_llm_response_text,
    mock_llm_executor,
    first_input_data,
    first_kwargs,
    second_input_data,
    second_kwargs,
    is_second_use_cache,
):
    # Run 1st time and fill cache
    cache_namespace = "dynamiq"
    cache_config = RedisCacheConfig(
        host="redis-test-sv", port=6379, db=0, namespace=cache_namespace
    )
    tracing = TracingCallbackHandler()
    result = node_with_caching.run(
        input_data=first_input_data,
        config=RunnableConfig(callbacks=[tracing], cache=cache_config),
        **first_kwargs,
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=first_input_data,
        output={"content": mock_llm_response_text},
    )
    assert result == expected_result
    assert mock_llm_executor.call_count == 1
    for run in tracing.runs.values():
        if (
            run.type == RunType.NODE
            and run.metadata["node"]["group"] == NodeGroup.LLMS.value
        ):
            assert not run.metadata["is_output_from_cache"]
            assert (
                len(
                    mock_redis.keys(f"{cache_namespace}:{run.metadata['node']['id']}:*")
                )
                == 1
            )

    # Run 2nd time to test cache
    mock_llm_executor.reset_mock()

    tracing = TracingCallbackHandler()
    result = node_with_caching.run(
        input_data=second_input_data,
        config=RunnableConfig(callbacks=[tracing], cache=cache_config),
        **second_kwargs,
    )

    expected_result.input = second_input_data
    if is_second_use_cache:
        llm_call_count = 0
        is_output_from_cache = True
        node_redis_keys = 1
    else:
        llm_call_count = 1
        is_output_from_cache = False
        node_redis_keys = 2

    assert result == expected_result
    assert mock_llm_executor.call_count == llm_call_count
    for run in tracing.runs.values():
        if (
            run.type == RunType.NODE
            and run.metadata["node"]["group"] == NodeGroup.LLMS.value
        ):
            assert bool(run.metadata["is_output_from_cache"]) is is_output_from_cache
            assert len(mock_redis.keys(f"{cache_namespace}:{run.metadata['node']['id']}:*")) == node_redis_keys
