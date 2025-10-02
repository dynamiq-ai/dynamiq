import enum
import io
import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dynamiq.connections import Http as HttpConnection
from dynamiq.connections import HTTPMethod
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ActionParsingException, ToolExecutionException
from dynamiq.nodes.agents.utils import FileMappedInput
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file.base import FileInfo
from dynamiq.utils.logger import logger

DESCRIPTION_HTTP = """Makes HTTP API requests with support for all methods and configurable parameters.

Key Capabilities:
- All HTTP methods: GET, POST, PUT, DELETE, PATCH
- Custom headers, authentication, and request/response handling
- JSON, form data, and file upload support
- Automatic response parsing and error handling

Usage Strategy:
- Use GET for data retrieval, POST for creation
- Include authentication headers for secured APIs
- Handle different content types with appropriate parameters
- Configure timeouts and retries for reliability

Parameter Guide:
- url: Target API endpoint (required)
- method: HTTP method (GET, POST, PUT, DELETE)
- headers: Custom headers including authentication
- body/files: Request payload for POST/PUT operations

Examples:
- {"url": "https://api.example.com/data", "method": "GET", "headers": {"Authorization": "Bearer token"}}
- {"url": "https://api.com/users", "method": "POST", "body": {"name": "John"}}
- {"url": "https://api.com/upload", "method": "PUT", "files": {"file1.txt": "content1", "file2.pdf": "content2"}}"""


class ResponseType(str, enum.Enum):
    TEXT = "text"
    RAW = "raw"
    JSON = "json"


class RequestPayloadType(str, enum.Enum):
    RAW = "raw"
    JSON = "json"


def handle_file_upload(input_data: dict[str, str | bytes] | FileMappedInput) -> dict[str, io.BytesIO]:
    """
    Handles file uploading and converts all inputs to BytesIO objects.

    Args:
        input_data: Dictionary mapping parameter names to file objects to upload or FileMappedInput object.

    Returns:
        dict[str, io.BytesIO]: Dictionary mapping parameter names to BytesIO objects.

    Raises:
        ValueError: If invalid file data type is provided.
    """
    files_data = {}
    if isinstance(input_data, FileMappedInput):
        files = input_data.input
        files_map = {getattr(f, "name", f"file_{id(f)}"): f for f in input_data.files}
    else:
        files = input_data
        files_map = {}

    for param_name, file in files.items():
        if isinstance(file, bytes):
            bytes_io = io.BytesIO(file)
            bytes_io.name = param_name
            files_data[param_name] = bytes_io
        elif isinstance(file, io.BytesIO):
            files_data[param_name] = file
        elif isinstance(file, FileInfo):
            bytes_io = io.BytesIO(file.content)
            bytes_io.name = file.name
            files_data[param_name] = bytes_io
        elif isinstance(file, str):
            if files_map:
                if file in files_map:
                    bytes_io = io.BytesIO(files_map[file].getvalue())
                    bytes_io.name = files_map[file].name
                    files_data[param_name] = bytes_io
                else:
                    raise ValueError(f"File {file} not found in files.")
            else:
                raise ValueError(
                    f"Error: Invalid file data type: {type(file)}. "
                    "If you want to use file path from files, provide FileMappedInput object."
                )
        else:
            raise ValueError(f"Error: Invalid file data type: {type(file)}. Expected bytes, BytesIO, or FileInfo.")

    return files_data


class HttpApiCallInputSchema(BaseModel):
    data: dict = Field(default={}, description="Parameter to provide payload.")
    url: str = Field(default="", description="Parameter to provide endpoint url.")
    payload_type: RequestPayloadType = Field(default=None, description="Parameter to specify the type of payload data.")
    headers: dict = Field(default={}, description="Parameter to provide headers to the request.")
    params: dict = Field(default={}, description="Parameter to provide GET parameters in URL.")
    files: dict[str, io.BytesIO] = Field(
        default={},
        description="Parameter to provide files to the request. Maps parameter names to file paths for file uploads. "
        "Provide strings for file IDs from files.",
        map_from_storage=True,
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    @field_validator("files", mode="before")
    @classmethod
    def files_validator(cls, input_data: dict[str, str | bytes] | FileMappedInput):
        """Validate and process files."""
        return handle_file_upload(input_data)


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
        url(str): The endpoint url for sending request
        method(str): The HTTP method for sending request.
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
    method: HTTPMethod | None = None
    response_type: ResponseType | str | None = ResponseType.RAW
    is_files_allowed: bool = True
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
        files = {param: file_io.getvalue() for param, file_io in input_data.files.items()}

        extras = {"data": data} if payload_type == RequestPayloadType.RAW else {"json": data}
        url = input_data.url or self.url or self.connection.url
        if not url:
            raise ValueError("No url provided.")
        headers = input_data.headers
        params = input_data.params
        method = self.method or self.connection.method

        try:
            response = self.client.request(
                method=method,
                url=url,
                headers=self.connection.headers | self.headers | headers,
                params=self.connection.params | self.params | params,
                timeout=self.timeout,
                files=files,
                **extras,
            )
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}")
            raise ToolExecutionException(
                f"Request failed with error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        if response.status_code not in self.success_codes:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results.")
            raise ToolExecutionException(
                f"Request failed with unexpected status code: {response.status_code} and response: {response.text}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
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
