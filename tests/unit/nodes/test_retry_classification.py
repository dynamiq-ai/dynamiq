"""Retries are spent only on failures that waiting can fix, and bounded when they are.

Every node type runs through ``execute_with_retry``, not only LLMs, so the classifier
recognises what it is certain about and leaves everything else alone.
"""

import pytest
from litellm.exceptions import AuthenticationError, BadRequestError, RateLimitError, Timeout

from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.node import RETRY_JITTER_RATIO, Node, is_retryable_error
from dynamiq.runnables import RunnableStatus


def _llm_error(cls):
    return cls(message="boom", model="gpt-4o-mini", llm_provider="openai")


def _node(raises=None, **error_handling):
    class _Probe(Node):
        group: NodeGroup = NodeGroup.TOOLS
        name: str = "probe"
        attempts: int = 0

        def execute(self, input_data, config=None, **kwargs):
            self.attempts += 1
            if raises is not None:
                raise raises
            return {"content": "ok"}

    return _Probe(error_handling=ErrorHandling(**error_handling))


class TestIsRetryableError:
    @pytest.mark.parametrize("exc", [AuthenticationError, BadRequestError])
    def test_permanent_failures_are_not_retried(self, exc):
        assert not is_retryable_error(_llm_error(exc))

    @pytest.mark.parametrize("exc", [RateLimitError, Timeout])
    def test_transient_failures_are_still_retried(self, exc):
        assert is_retryable_error(_llm_error(exc))

    def test_unrecognised_errors_keep_the_previous_behaviour(self):
        """Narrow the retry budget; never second-guess an error we do not understand."""
        assert is_retryable_error(RuntimeError("something else entirely"))


class TestRetryLoopHonoursTheClassification:
    def test_a_permanent_failure_stops_after_one_attempt(self):
        node = _node(raises=_llm_error(AuthenticationError), max_retries=3, retry_interval_seconds=0.01)

        result = node.run(input_data={})

        assert result.status == RunnableStatus.FAILURE
        assert node.attempts == 1, "an invalid credential fails the same way on every attempt"

    def test_a_transient_failure_uses_the_whole_budget(self):
        node = _node(raises=_llm_error(RateLimitError), max_retries=3, retry_interval_seconds=0.01)

        result = node.run(input_data={})

        assert result.status == RunnableStatus.FAILURE
        assert node.attempts == 4


class TestRetryBackoff:
    def test_growth_carries_jitter(self):
        """Deterministic backoff wakes every node that failed together at the same instant."""
        node = _node(retry_interval_seconds=1, backoff_rate=2)

        waits = [node._retry_backoff_seconds(2) for _ in range(20)]

        assert all(4 * (1 - RETRY_JITTER_RATIO) <= w <= 4 * (1 + RETRY_JITTER_RATIO) for w in waits)
        assert len(set(waits)) > 1

    def test_cap_is_a_hard_ceiling(self):
        """Applied after jitter, so the configured maximum is the real maximum."""
        node = _node(retry_interval_seconds=1, backoff_rate=3, max_retry_interval_seconds=20)

        assert all(node._retry_backoff_seconds(4) <= 20 for _ in range(20))

    def test_uncapped_by_default(self):
        node = _node(retry_interval_seconds=1, backoff_rate=3)

        assert node._retry_backoff_seconds(4) > 20
