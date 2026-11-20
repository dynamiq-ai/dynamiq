import asyncio
import logging

import pytest

from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import InputStreamingTimeoutError


class Lookup(Node):
    group: NodeGroup = NodeGroup.TOOLS
    name: str = "lookup"

    def execute(self, input_data: dict, config: RunnableConfig = None, **kwargs) -> dict:
        raise ValueError("upstream returned 400")


class Unreachable(Lookup):
    def ensure_client(self):
        raise ConnectionError("upstream returned 400")


@pytest.mark.parametrize("node_class", [Lookup, Unreachable], ids=["execution", "connection"])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_failed_node_run_is_logged_at_error_level_once(caplog, mode, node_class):
    """Error trackers count every error record, so a failure logged three times reads as three."""
    tool = node_class(error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0))
    with caplog.at_level(logging.INFO):
        if mode == "sync":
            result = tool.run(input_data={"query": "x"})
        else:
            result = asyncio.run(tool.run_async(input_data={"query": "x"}))

    assert result.status == RunnableStatus.FAILURE
    errors = [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(errors) == 1, errors
    assert "upstream returned 400" in errors[0]


class WaitsForAHuman(Lookup):
    def execute(self, input_data: dict, config: RunnableConfig = None, **kwargs) -> dict:
        raise InputStreamingTimeoutError(node_id=self.id, timeout=3)


def test_a_wait_for_human_input_that_times_out_is_not_an_error(caplog):
    with caplog.at_level(logging.INFO):
        result = WaitsForAHuman().run(input_data={})

    assert result.status == RunnableStatus.FAILURE
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
