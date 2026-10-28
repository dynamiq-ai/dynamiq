import pytest

from dynamiq.connections import FXMacroData
from dynamiq.nodes.tools.fxmacrodata import FXMacroDataTool


@pytest.fixture
def announcements_payload():
    """Trimmed response from /v1/announcements/usd/inflation."""
    return {
        "currency": "USD",
        "indicator": "inflation",
        "name": "Inflation (CPI)",
        "source": "BLS",
        "pagination": {"limit": 2, "offset": 0, "returned_count": 2, "has_more": False},
        "data": [
            {
                "date": "2026-08-31",
                "val": 3.4,
                "announcement_datetime": 1789129800,
                "announcement_datetime_local": "2026-09-11T08:30:00-04:00",
                "source_url": "https://www.bls.gov/news.release/archives/cpi_09112026.htm",
            },
            {
                "date": "2026-07-31",
                "val": 3.4,
                "announcement_datetime": 1786537800,
                "announcement_datetime_local": None,
                "source_url": None,
            },
        ],
    }


def _mock_requests(mocker, payload, status_code=200):
    mock_response = mocker.Mock()
    mock_response.json.return_value = payload
    mock_response.status_code = status_code
    mock_response.text = ""
    return mocker.patch("requests.request", return_value=mock_response)


def test_connection_sends_key_header():
    connection = FXMacroData(api_key="test_key")
    assert connection.headers["X-API-Key"] == "test_key"
    assert connection.url == "https://api.fxmacrodata.com"


def test_connection_without_key_sends_no_key_header(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    connection = FXMacroData()
    assert connection.api_key is None
    assert "X-API-Key" not in connection.headers


def test_announcements(mocker, announcements_payload):
    mock_requests = _mock_requests(mocker, announcements_payload)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"currency": "USD", "indicator": "inflation", "limit": 2, "start_date": "2026-01-01"})

    mock_requests.assert_called_once()
    call_kwargs = mock_requests.call_args[1]
    assert call_kwargs["method"] == "GET"
    assert call_kwargs["url"] == "https://api.fxmacrodata.com/v1/announcements/usd/inflation"
    assert call_kwargs["headers"]["X-API-Key"] == "test_key"
    assert call_kwargs["params"] == {"limit": 2, "start_date": "2026-01-01"}

    content = result.output["content"]
    assert "2026-08-31: 3.4 (released 2026-09-11T08:30:00-04:00)" in content["result"]
    assert "2026-07-31: 3.4 (released release time unknown)" in content["result"]
    assert content["sources_with_url"] == [
        "[Inflation (CPI) 2026-08-31](https://www.bls.gov/news.release/archives/cpi_09112026.htm)"
    ]
    assert content["raw_response"] == announcements_payload


def test_announcements_optimized_for_agents(mocker, announcements_payload):
    _mock_requests(mocker, announcements_payload)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"), is_optimized_for_agents=True)

    result = tool.run({"currency": "usd", "indicator": "inflation"})

    content = result.output["content"]
    assert content.startswith("## Sources with URLs")
    assert "## FXMacroData announcements" in content
    assert "Inflation (CPI) (USD), source: BLS" in content


def test_announcements_requires_indicator(mocker):
    mock_requests = _mock_requests(mocker, {})
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"currency": "usd"})

    assert result.status.value == "failure"
    assert "indicator" in result.error.message
    mock_requests.assert_not_called()


def test_rejects_bad_currency_code(mocker):
    mock_requests = _mock_requests(mocker, {})
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"endpoint": "calendar", "currency": "us dollar"})

    assert result.status.value == "failure"
    assert "three-letter currency code" in result.error.message
    mock_requests.assert_not_called()


def test_calendar(mocker):
    payload = {
        "currency": "USD",
        "data": [
            {
                "release": "trade_balance",
                "name": "Trade Balance",
                "announcement_datetime_utc": "2026-10-06T12:30:00+00:00",
                "announcement_datetime_local": "2026-10-06T08:30:00-04:00",
                "event_importance": "medium",
            }
        ],
    }
    mock_requests = _mock_requests(mocker, payload)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"endpoint": "calendar", "currency": "usd", "indicator": "trade_balance", "limit": 5})

    call_kwargs = mock_requests.call_args[1]
    assert call_kwargs["url"] == "https://api.fxmacrodata.com/v1/calendar/usd"
    assert call_kwargs["params"] == {"indicator": "trade_balance"}
    assert (
        "2026-10-06T08:30:00-04:00: Trade Balance (trade_balance), importance medium"
        in result.output["content"]["result"]
    )


def test_data_catalogue(mocker):
    payload = {
        "gdp": {"name": "GDP", "unit": "USD bn", "source": "BEA"},
        "policy_rate": {"name": "Policy Rate"},
    }
    mock_requests = _mock_requests(mocker, payload)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"endpoint": "data_catalogue", "currency": "usd", "indicator": "gdp", "start_date": "2026-01-01"})

    call_kwargs = mock_requests.call_args[1]
    assert call_kwargs["url"] == "https://api.fxmacrodata.com/v1/data_catalogue/usd"
    assert call_kwargs["params"] == {}
    assert result.output["content"]["result"] == "gdp: GDP (USD bn, BEA)\npolicy_rate: Policy Rate"


def test_forex(mocker):
    payload = {
        "base": "EUR",
        "quote": "USD",
        "source": "Official central-bank reference rates",
        "data": [{"date": "2026-10-02", "val": 1.1712}],
    }
    mock_requests = _mock_requests(mocker, payload)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"endpoint": "forex", "currency": "eur", "quote": "USD", "limit": 1})

    call_kwargs = mock_requests.call_args[1]
    assert call_kwargs["url"] == "https://api.fxmacrodata.com/v1/forex/eur/usd"
    assert call_kwargs["params"] == {"limit": 1}
    assert "2026-10-02: 1.1712" in result.output["content"]["result"]


def test_forex_requires_quote(mocker):
    mock_requests = _mock_requests(mocker, {})
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"endpoint": "forex", "currency": "eur"})

    assert result.status.value == "failure"
    assert "'quote'" in result.error.message
    mock_requests.assert_not_called()


def test_node_defaults_are_used(mocker, announcements_payload):
    mock_requests = _mock_requests(mocker, announcements_payload)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"), currency="usd", indicator="inflation", limit=3)

    tool.run({})

    call_kwargs = mock_requests.call_args[1]
    assert call_kwargs["url"] == "https://api.fxmacrodata.com/v1/announcements/usd/inflation"
    assert call_kwargs["params"] == {"limit": 3}


def test_error_response_surfaces_detail(mocker):
    _mock_requests(
        mocker,
        {"detail": "This endpoint requires an Individual or Business API key.", "code": "api_key_required"},
        status_code=401,
    )
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"endpoint": "forex", "currency": "eur", "quote": "usd"})

    assert result.status.value == "failure"
    assert "HTTP 401" in result.error.message
    assert "requires an Individual or Business API key" in result.error.message


def test_key_is_stripped_and_redirects_are_not_followed(mocker, announcements_payload):
    mock_requests = _mock_requests(mocker, announcements_payload)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="  test_key\n"))

    tool.run({"currency": "usd", "indicator": "inflation"})

    call_kwargs = mock_requests.call_args[1]
    assert call_kwargs["headers"]["X-API-Key"] == "test_key"
    assert call_kwargs["allow_redirects"] is False
    assert call_kwargs["timeout"] == 30.0


def test_malformed_key_fails_without_echoing_it(mocker):
    mock_requests = _mock_requests(mocker, {})
    connection = FXMacroData(api_key="secret key")
    assert "X-API-Key" not in connection.headers
    tool = FXMacroDataTool(connection=connection)

    result = tool.run({"currency": "usd", "indicator": "inflation"})

    assert result.status.value == "failure"
    assert "secret" not in result.error.message
    mock_requests.assert_not_called()


def test_redirect_response_is_refused(mocker):
    _mock_requests(mocker, {}, status_code=302)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"endpoint": "calendar", "currency": "usd"})

    assert result.status.value == "failure"
    assert "HTTP 302: redirect refused" in result.error.message


def test_key_is_redacted_from_exception_text(mocker):
    mocker.patch("requests.request", side_effect=ValueError("Invalid header value b'secret-key-123'"))
    tool = FXMacroDataTool(connection=FXMacroData(api_key="secret-key-123"))

    result = tool.run({"endpoint": "calendar", "currency": "usd"})

    assert result.status.value == "failure"
    assert "secret-key-123" not in result.error.message
    assert "[redacted]" in result.error.message


def test_key_is_redacted_from_error_body(mocker):
    _mock_requests(mocker, {"detail": "Key secret-key-123 has no EUR access"}, status_code=403)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="secret-key-123"))

    result = tool.run({"currency": "eur", "indicator": "inflation"})

    assert "secret-key-123" not in result.error.message


def test_error_body_with_http_200_is_a_failure(mocker):
    _mock_requests(mocker, {"detail": "Quota exceeded"})
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"endpoint": "calendar", "currency": "usd"})

    assert result.status.value == "failure"
    assert "Quota exceeded" in result.error.message


def test_non_json_body_is_a_failure(mocker):
    mock_response = mocker.Mock(status_code=200, text="<html>gateway</html>")
    mock_response.json.side_effect = ValueError("not json")
    mocker.patch("requests.request", return_value=mock_response)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"endpoint": "calendar", "currency": "usd"})

    assert result.status.value == "failure"
    assert "not JSON" in result.error.message


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("calendar", []),
        ("calendar", {"currency": "USD"}),
        ("calendar", {"data": "x"}),
        ("forex", {"data": [1.17]}),
        ("data_catalogue", ["gdp"]),
    ],
)
def test_malformed_shapes_are_failures(mocker, endpoint, payload):
    _mock_requests(mocker, payload)
    tool = FXMacroDataTool(connection=FXMacroData(api_key="test_key"))

    result = tool.run({"endpoint": endpoint, "currency": "eur", "quote": "usd"})

    assert result.status.value == "failure"
    assert "Unexpected response shape" in result.error.message
