import enum
import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from dynamiq.connections import Http as HttpConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ActionParsingException, ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig


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


def recursive_merge(dict1: dict, dict2: dict) -> dict:
    """
    Recursively merge two dictionaries.

    If a key exists in both dictionaries:
    - If value are not lists or dictionaries, the value in dict1 will be used.
    - If both values are lists or dictionaries, they are merged recursively.

    Args:
        dict1 (dict): The first dictionary to merge.
        dict2 (dict): The second dictionary to merge.

    Returns:
        dict: The merged dictionary
    """
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                dict1[key] = recursive_merge(dict1[key], value)
            elif isinstance(dict1[key], list) and isinstance(value, list):
                dict1[key].extend(x for x in value if x not in dict1[key])
        else:
            dict1[key] = value
    return dict1


def concatenate_parameters(
    input_params: HttpApiCallInputSchema, additional_params: HttpApiCallInputSchema
) -> HttpApiCallInputSchema:
    """
    Concatenate parameters for an API call.

    Args:
        input_params (HttpApiCallInputSchema): The first set of parameters for the API call.
        additional_params (HttpApiCallInputSchema): The second set of parameters to be merged into the first.

    Returns:
        HttpApiCallInputSchema: A new object with the combined parameters.
    """
    combined_params = recursive_merge(input_params.params.copy(), additional_params.params)
    combined_headers = recursive_merge(input_params.headers.copy(), additional_params.headers)
    combined_data = recursive_merge(input_params.data.copy(), additional_params.data)

    return HttpApiCallInputSchema(
        params=combined_params,
        data=combined_data,
        url=input_params.url or additional_params.url,
        payload_type=input_params.payload_type or additional_params.payload_type,
        headers=combined_headers,
    )


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
        additional_input_data(Optional[HttpApiCallInputSchema]): Additional predefined input data parameters.
    """

    name: str = "Api Call Tool"
    description: str = "The description of the API call tool"
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
    additional_input_data: HttpApiCallInputSchema | None = None

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

        if self.additional_input_data is not None:
            input_data = concatenate_parameters(input_data, self.additional_input_data)

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
        return {"content": content, "status_code": response.status_code}
