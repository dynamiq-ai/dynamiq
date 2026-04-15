from unittest.mock import MagicMock

import pytest

from dynamiq.nodes.tools.context_manager import ContextManagerTool
from dynamiq.prompts import Message, MessageRole
from dynamiq.runnables import RunnableStatus


def _mock_tool(outputs):
    llm = MagicMock()
    llm.model = "gpt-4o-mini"
    llm.get_token_limit.return_value = 128_000
    llm.is_postponed_component_init = False
    llm.to_dict.return_value = {}
    llm.run.side_effect = [MagicMock(status=RunnableStatus.SUCCESS, output=out) for out in outputs]
    return ContextManagerTool(llm=llm, max_retries=3), llm


def test_retry_returns_summary_after_empty_attempts():
    tool, llm = _mock_tool([{"content": ""}, {"content": "  "}, {"content": "ok"}])
    assert tool._call_llm_for_summary([Message(role=MessageRole.USER, content="x")]) == "ok"
    assert llm.run.call_count == 3


def test_retry_raises_when_all_attempts_empty():
    tool, llm = _mock_tool([{"content": ""}] * 3)
    with pytest.raises(ValueError, match="failed to generate summary after 3 attempts"):
        tool._call_llm_for_summary([Message(role=MessageRole.USER, content="x")])
    assert llm.run.call_count == 3
