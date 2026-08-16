"""Retries are for failures that waiting can fix.

A bad credential, a malformed request or an exhausted budget fails identically on every
attempt, so spending the retry budget on it only delays the error the caller already has.
Anything unrecognised keeps the previous behaviour and is still retried.
"""

import time
from typing import ClassVar

import httpx
import pytest
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    BudgetExceededError,
    ContextWindowExceededError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    Timeout,
)
from pydantic import BaseModel

from dynamiq.nodes.agents.exceptions import LLMContextWindowExceededError
from dynamiq.nodes.node import ErrorHandling, Node, NodeGroup, is_retryable_error
from dynamiq.runnables import RunnableConfig


def _llm_error(cls, **kw):
    """Some litellm errors require an httpx response; passing one suits them all."""
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
    return cls(message="x", model="m", llm_provider="openai", response=response, **kw)


class TestClassification:
    @pytest.mark.parametrize(
        "exc",
        [
            _llm_error(AuthenticationError),
            _llm_error(PermissionDeniedError),
            _llm_error(BadRequestError),
            _llm_error(NotFoundError),
            _llm_error(ContextWindowExceededError),
            BudgetExceededError(current_cost=10, max_budget=1),
            LLMContextWindowExceededError("prompt too long"),
        ],
    )
    def test_permanent_failures_are_not_retryable(self, exc):
        assert is_retryable_error(exc) is False

    @pytest.mark.parametrize(
        "exc",
        [
            _llm_error(RateLimitError),
            _llm_error(InternalServerError),
            Timeout(message="x", model="m", llm_provider="openai"),
            APIConnectionError(message="x", model="m", llm_provider="openai"),
            ConnectionError("socket died"),
            TimeoutError("too slow"),
            RuntimeError("something we have never seen"),
            ValueError("a plain error"),
        ],
    )
    def test_transient_and_unknown_failures_stay_retryable(self, exc):
        assert is_retryable_error(exc) is True

    def test_rate_limit_is_not_caught_by_the_bad_request_rule(self):
        """RateLimitError must not inherit its way into the non-retryable set."""
        assert not isinstance(_llm_error(RateLimitError), BadRequestError)
        assert is_retryable_error(_llm_error(RateLimitError)) is True


class _In(BaseModel):
    pass


def _failing_node(exc_factory, counter: list, **error_handling):
    class Failing(Node):
        group: NodeGroup = NodeGroup.TOOLS
        name: str = "failing"
        description: str = "always fails"
        input_schema: ClassVar[type[BaseModel]] = _In

        def execute(self, input_data, config: RunnableConfig = None, **kwargs):
            counter.append(1)
            raise exc_factory()

    settings = {"max_retries": 3, "retry_interval_seconds": 0.01, **error_handling}
    return Failing(error_handling=ErrorHandling(**settings))


class TestRetryBudget:
    def test_permanent_failure_costs_one_attempt(self):
        calls: list = []
        _failing_node(lambda: _llm_error(AuthenticationError), calls).run(input_data={})
        assert len(calls) == 1

    def test_transient_failure_still_uses_the_whole_budget(self):
        calls: list = []
        _failing_node(lambda: _llm_error(RateLimitError), calls).run(input_data={})
        assert len(calls) == 4  # the initial attempt plus max_retries

    def test_unknown_failure_still_uses_the_whole_budget(self):
        """Narrowing the budget must not change behaviour for errors we do not classify."""
        calls: list = []
        _failing_node(lambda: RuntimeError("mystery"), calls).run(input_data={})
        assert len(calls) == 4

    def test_result_is_still_a_failure_not_an_exception(self):
        calls: list = []
        result = _failing_node(lambda: _llm_error(AuthenticationError), calls).run(input_data={})
        assert result.status.value == "failure"
        assert result.error is not None


def _failing_async_node(exc_factory, counter: list, **error_handling):
    class FailingAsync(Node):
        group: NodeGroup = NodeGroup.TOOLS
        name: str = "failing-async"
        description: str = "always fails"
        input_schema: ClassVar[type[BaseModel]] = _In

        def execute(self, input_data, config: RunnableConfig = None, **kwargs):
            raise AssertionError("the async path must not fall back to the sync one")

        async def execute_async(self, input_data, config: RunnableConfig = None, **kwargs):
            counter.append(1)
            raise exc_factory()

    settings = {"max_retries": 3, "retry_interval_seconds": 0.01, **error_handling}
    return FailingAsync(error_handling=ErrorHandling(**settings))


class TestAsyncPathMatchesSync:
    """A node's retry behaviour must not depend on which loop happens to run it.

    ``execute_async_with_retry`` had none of the shaping the sync loop grew — it retried
    permanent failures and ignored ``max_retry_interval_seconds`` entirely, even though
    the setting is documented on ``ErrorHandling`` with no mention of a sync-only scope.
    """

    @pytest.mark.asyncio
    async def test_permanent_failure_costs_one_attempt(self):
        calls: list = []
        node = _failing_async_node(lambda: _llm_error(AuthenticationError), calls)
        with pytest.raises(AuthenticationError):
            await node.execute_async_with_retry(input_data={}, config=RunnableConfig())
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_transient_failure_still_uses_the_whole_budget(self):
        calls: list = []
        node = _failing_async_node(lambda: _llm_error(RateLimitError), calls)
        with pytest.raises(RateLimitError):
            await node.execute_async_with_retry(input_data={}, config=RunnableConfig())
        assert len(calls) == 4

    @pytest.mark.asyncio
    async def test_unknown_failure_still_uses_the_whole_budget(self):
        calls: list = []
        node = _failing_async_node(lambda: RuntimeError("mystery"), calls)
        with pytest.raises(RuntimeError):
            await node.execute_async_with_retry(input_data={}, config=RunnableConfig())
        assert len(calls) == 4

    @pytest.mark.asyncio
    async def test_interval_ceiling_caps_a_single_wait(self):
        calls: list = []
        node = _failing_async_node(
            lambda: _llm_error(RateLimitError),
            calls,
            retry_interval_seconds=0.4,
            backoff_rate=10.0,
            max_retry_interval_seconds=0.05,
        )
        started = time.time()
        with pytest.raises(RateLimitError):
            await node.execute_async_with_retry(input_data={}, config=RunnableConfig())
        # Uncapped this would wait 0.4 + 4 + 40 seconds.
        assert time.time() - started < 5


class TestBackoffShaping:
    def test_both_loops_share_one_backoff_implementation(self):
        """Two copies of the formula is how the async path drifted in the first place."""
        node = _failing_node(lambda: RuntimeError("x"), [], retry_interval_seconds=1, backoff_rate=2)
        assert 1.8 <= node._retry_backoff_seconds(1) <= 2.2  # 1 * 2**1, +/- jitter

    def test_jitter_actually_varies_the_wait(self):
        node = _failing_node(lambda: RuntimeError("x"), [], retry_interval_seconds=1)
        assert len({node._retry_backoff_seconds(0) for _ in range(20)}) > 1

    def test_ceiling_applies_before_jitter_not_after(self):
        node = _failing_node(lambda: RuntimeError("x"), [], retry_interval_seconds=100, max_retry_interval_seconds=10)
        assert all(node._retry_backoff_seconds(0) <= 10 * (1 + 0.1) for _ in range(20))

    def test_interval_ceiling_caps_a_single_wait(self):
        calls: list = []
        node = _failing_node(
            lambda: _llm_error(RateLimitError),
            calls,
            retry_interval_seconds=0.4,
            backoff_rate=10.0,
            max_retry_interval_seconds=0.05,
        )
        started = time.time()
        node.run(input_data={})
        # Uncapped this would wait 0.4 + 4 + 40 seconds.
        assert time.time() - started < 5

    def test_ceiling_is_off_by_default(self):
        assert ErrorHandling().max_retry_interval_seconds is None

    def test_defaults_are_unchanged(self):
        eh = ErrorHandling()
        assert (eh.max_retries, eh.retry_interval_seconds, eh.backoff_rate) == (0, 1, 1)
