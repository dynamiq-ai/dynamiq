import pytest

from dynamiq import Workflow, flows
from dynamiq.cache import RedisCacheConfig
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunType
from dynamiq.nodes import CachingConfig, NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


@pytest.fixture()
def node_with_caching(openai_node):
    openai_node.caching = CachingConfig(enabled=True)
    return openai_node


def test_node_caching(
    node_with_caching,
    mock_redis,
    mock_redis_backend,
    mock_llm_response_text,
    mock_llm_executor,
):
    # Run 1st time and fill cache
    cache_namespace = "dynamiq"
    cache_config = RedisCacheConfig(
        host="redis-test-sv", port=6379, db=0, namespace=cache_namespace
    )
    tracing = TracingCallbackHandler()
    input_data = {"a": 1, "b": 2}
    wf = Workflow(flow=flows.Flow(nodes=[node_with_caching]))
    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing], cache=cache_config),
    )

    expected_output = {
        node_with_caching.id: RunnableResult(
            status=RunnableStatus.SUCCESS,
            input=input_data,
            output={"content": mock_llm_response_text, "tool_calls": None},
        ).to_dict()
    }
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS, input=input_data, output=expected_output
    )
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
    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing], cache=cache_config),
    )

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS, input=input_data, output=expected_output
    )
    assert mock_llm_executor.call_count == 0
    for run in tracing.runs.values():
        if (
            run.type == RunType.NODE
            and run.metadata["node"]["group"] == NodeGroup.LLMS.value
        ):
            assert run.metadata["is_output_from_cache"]
            assert (
                len(
                    mock_redis.keys(f"{cache_namespace}:{run.metadata['node']['id']}:*")
                )
                == 1
            )
