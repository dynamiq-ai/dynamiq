import enum
import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from dynamiq.connections import Http as HttpConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ActionParsingException, ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_HTTP = """## HTTP API Call Tool
### Overview
Make web requests to external APIs and services with support for various HTTP methods, payload formats, and response types.
### Capabilities
- Execute all standard HTTP methods (GET, POST, PUT, DELETE, PATCH)
- Configure headers, query parameters, and request body
- Handle multiple payload formats (raw or JSON) and response types
- Customize timeout and success response criteria
### When to Use
- Fetch data from external web services
- Submit data to external systems
- Interact with HTTP-based APIs including RESTful services
- Access authenticated third-party services
### Parameters
- **url** (string): Required endpoint URL if not configured in tool settings
- **data** (object, optional): Request body payload (JSON-serializable)
- **headers** (object, optional): HTTP request headers
- **params** (object, optional): URL query parameters
- **payload_type** (string, optional): Format of payload ("raw" or "json", default: "raw")
### Configuration
- **method** (string): HTTP method (GET, POST, PUT, DELETE, etc.)
- **timeout** (number): Request timeout in seconds (default: 30)
- **success_codes** (array): HTTP status codes considered successful (default: [200])
- **response_type** (string): Response format ("text", "raw", "json", default: "raw")
  - "text": Returns response as string
  - "raw": Returns raw response content as bytes
  - "json": Parses response as JSON object
  - Responses with "application/json" content-type automatically parse as JSON
### Examples
#### Basic GET Request
{"url": "https://api.example.com/data"}
#### POST with JSON Payload
{
  "url": "https://api.example.com/submit",
  "data": {"name": "John Doe", "email": "john@example.com"},
  "payload_type": "json",
  "headers": {
    "Content-Type": "application/json",
    "Authorization": "Bearer token123"
  }
}
#### GET with Query Parameters
{
  "url": "https://api.example.com/search",
  "params": {"q": "search term", "limit": 10}
}
### Best Practices
1. Set appropriate timeouts based on expected response times
2. Include authentication in headers rather than URLs
3. Use "json" response type for JSON APIs, "text" for HTML/plain text
4. The tool handles URL encoding for query parameters
"""  # noqa: E501


class ResponseType(str, enum.Enum):
    TEXT = "text"
    RAW = "raw"
    JSON = "json"


class RequestPayloadType(str, enum.Enum):
    RAW = "raw"
    JSON = "json"


class HttpApiCallInputSchema(BaseModel):
    data: dict = Field(default={}, description="Parameter to provide payload.")
    url: str = Field(default="", description="Parameter to provide endpoint url.")
    payload_type: RequestPayloadType = Field(default=None, description="Parameter to specify the type of payload data.")
    headers: dict = Field(default={}, description="Parameter to provide headers to the request.")
    params: dict = Field(default={}, description="Parameter to provide GET parameters in URL.")

    @field_validator("data", "headers", "params", mode="before")
    @classmethod
    def validate_dict_fields(cls, value: Any, field: str) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value or "{}")
            except json.JSONDecodeError as e:
                raise ActionParsingException(f"Invalid JSON string provided for '{field}'. Error: {e}")
        elif isinstance(value, dict):
            return value
        else:
            raise ActionParsingException(f"Expected a dictionary or a JSON string for '{field}'.")


class HttpApiCall(ConnectionNode):
    """
    A component for sending API requests using requests library.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group the node belongs to.
        connection (HttpConnection | None): The connection based on sending http requests.A new connection
            is created if none is provided.
        success_codes(list[int]): The list of codes when request is successful.
        timeout (float): The timeout in seconds.
        data(dict[str,Any]): The data to send as body of request.
        headers(dict[str,Any]): The headers of request.
        payload_type (dict[str, Any]): Parameter to specify the type of payload data.
        params(dict[str,Any]): The additional query params of request.
        response_type(ResponseType|str): The type of response content.
    """

    name: str = "Api Call Tool"
    description: str = DESCRIPTION_HTTP
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    connection: HttpConnection
    success_codes: list[int] = [200]
    timeout: float = 30
    payload_type: RequestPayloadType = RequestPayloadType.RAW
    data: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    url: str = ""
    response_type: ResponseType | str | None = ResponseType.RAW
    input_schema: ClassVar[type[HttpApiCallInputSchema]] = HttpApiCallInputSchema

    def execute(self, input_data: HttpApiCallInputSchema, config: RunnableConfig = None, **kwargs):
        """Execute the API call.

        This method takes input data and returns content of API call response.

        Args:
            input_data (dict[str, Any]): The input data containing(optionally) data, headers, payload_type,
                params for request.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
             dict: A dictionary with the following keys:
                - "content" (bytes|string|dict[str,Any]): Value containing the result of request.
                - "status_code" (int): The status code of the request.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n" f"{input_data.model_dump()}")

        data = self.connection.data | self.data | input_data.data
        payload_type = input_data.payload_type or self.payload_type
        extras = {"data": data} if payload_type == RequestPayloadType.RAW else {"json": data}
        url = input_data.url or self.url or self.connection.url
        if not url:
            raise ValueError("No url provided.")
        headers = input_data.headers
        params = input_data.params

        response = self.client.request(
            method=self.connection.method,
            url=url,
            headers=self.connection.headers | self.headers | headers,
            params=self.connection.params | self.params | params,
            timeout=self.timeout,
            **extras,
        )

        if response.status_code not in self.success_codes:
            raise ToolExecutionException(
                f"Request failed with unexpected status code: {response.status_code} and response: {response.text}. "
                f"Please analyze the error and take appropriate action."
            )

        response_type = self.response_type
        if "response_type" not in self.model_fields_set and response.headers.get("content-type") == "application/json":
            response_type = ResponseType.JSON

        if response_type == ResponseType.TEXT:
            content = response.text
        elif response_type == ResponseType.RAW:
            content = response.content
        elif response_type == ResponseType.JSON:
            content = response.json()
        else:
            allowed_types = [item.value for item in ResponseType]
            raise ValueError(
                f"Response type must be one of the following: {', '.join(allowed_types)}"
            )
        logger.info(f"Tool {self.name} - {self.id}: finished with RESULT:\n" f"{str(content)[:200]}...")
        return {"content": content, "status_code": response.status_code}
