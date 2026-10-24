import pytest

from dynamiq.connections import ScaleSerp, ZenRows
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from dynamiq.runnables import RunnableStatus

# Nothing listens on port 9, so the request fails at once with an error that quotes the URL and its query.
UNREACHABLE = "http://127.0.0.1:9"


@pytest.mark.parametrize(
    ("tool", "input_data"),
    [
        (ScaleSerpTool(connection=ScaleSerp(api_key="scale-secret", url=UNREACHABLE)), {"query": "x"}),
        (ZenRowsTool(connection=ZenRows(api_key="scale-secret", url=UNREACHABLE)), {"url": "https://example.com"}),
    ],
)
def test_request_errors_leave_out_the_api_key(tool, input_data):
    result = tool.run(input_data=input_data)

    assert result.status == RunnableStatus.FAILURE
    assert "scale-secret" not in str(result.error.message)
