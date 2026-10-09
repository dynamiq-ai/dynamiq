from dynamiq.cli.client import ApiClient
from dynamiq.cli.config import Settings


def test_requests_name_the_cli(requests_mock):
    requests_mock.get("https://api.example.test/v1/projects", json={"data": []})

    ApiClient(Settings(api_host="https://api.example.test", api_key="pat")).get("/v1/projects")

    assert requests_mock.last_request.headers["User-Agent"].startswith("dynamiq-cli/")
    assert requests_mock.last_request.headers["Authorization"] == "Bearer pat"
