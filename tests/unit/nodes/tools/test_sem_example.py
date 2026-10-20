"""The optional Sem example validates its path and limits the selected tools."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from examples.components.tools.mcp_server_as_tool import use_sem


def test_missing_checkout(tmp_path):
    with pytest.raises(FileNotFoundError):
        use_sem.run_example(str(tmp_path / "missing"), "parse_config")


def test_checkout_must_be_directory(tmp_path):
    path = tmp_path / "file"
    path.touch()
    with pytest.raises(ValueError, match="directory"):
        use_sem.run_example(str(path), "parse_config")


@pytest.mark.parametrize("success", [True, False])
def test_tool_selection_and_result(tmp_path, success):
    tool = MagicMock()
    tool.name = "sem_find"
    tool.run.return_value = SimpleNamespace(
        status=use_sem.RunnableStatus.SUCCESS if success else use_sem.RunnableStatus.FAILURE,
        output={"content": "fixture code"},
        error="fixture failure",
    )
    with patch.object(use_sem, "MCPServer") as server:
        server.return_value.get_mcp_tools.return_value = [tool]
        if success:
            assert use_sem.run_example(str(tmp_path), "parse_config") == {"content": "fixture code"}
        else:
            with pytest.raises(RuntimeError, match="sem_find failed"):
                use_sem.run_example(str(tmp_path), "parse_config")
        assert server.call_args.kwargs["include_tools"] == ["sem_find", "sem_grep"]
        assert server.call_args.kwargs["connection"].env == {"SEM_REPO": str(tmp_path.resolve())}
        tool.run.assert_called_once_with(input_data={"query": "parse_config", "mode": "context", "token_budget": 1200})
