from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import HuggingFace, Lakera
from dynamiq.nodes.node import ConnectionNode, ErrorHandling, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig

HUGGINGFACE_URL = "https://api-inference.huggingface.co/models/"


class PIIDetectorInputSchema(BaseModel):
    message: str = Field(..., description="Parameter to provide message to check for PII.")


class PIIDetector(ConnectionNode):
    group: Literal[NodeGroup.DETECTORS] = NodeGroup.DETECTORS
    name: str = "pii-detector"
    description: str = "Node that detects PII"
    connection: HuggingFace | Lakera | None = None
    model: str = "iiiorg/piiranha-v1-detect-personal-information"
    timeout: float = 30
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(max_retries=1))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[PIIDetectorInputSchema]] = PIIDetectorInputSchema

    def __init__(self, **kwargs):
        """Initialize the Prompt Injection Detector.

        If neither client nor connection is provided in kwargs, a new HuggingFace connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = HuggingFace()
        super().__init__(**kwargs)

    def execute(self, input_data: PIIDetectorInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Detects Personally Identifiable Information (PII) in input data

        Args:
            input_data (PIIDetectorInputSchema): input data for the tool, which includes the message to check.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing detected PII information.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        message = input_data.message
        try:
            if isinstance(self.connection, HuggingFace):
                result = self.huggingface_detect(message=message)
            elif isinstance(self.connection, Lakera):
                result = self.lakera_detect(message=message)
            else:
                raise ValueError(f"Unsupported provider: {type(self.connection)}")
            return result
        except Exception as e:
            msg = f"Encountered an error while performing validation. \nError details: {e}"
            raise ValueError(msg)

    def huggingface_detect(self, message: str) -> dict[str, Any]:
        """
        Detects prompt injection attempts in a given message using a Hugging Face model.

        Args:
            message (str): The input message to analyze for prompt injection.
        """
        api_url = urljoin(HUGGINGFACE_URL, self.model)
        headers = {"Authorization": f"Bearer {self.connection.api_key}"}
        payload = {
            "inputs": message,
        }
        response = requests.post(api_url, headers=headers, json=payload, timeout=self.timeout)

        if response.status_code == 200:
            result = [item["entity_group"] for item in response.json()]
            return {"is_detected": bool(result), "detected_pii": result}
        elif response.status_code in [400, 401]:
            raise ValueError("Request failed. Possible causes: bad request or invalid API key.")
        else:
            raise ValueError(f"Request failed with status code {response.status_code}")

    def lakera_detect(self, message: str) -> dict[str, Any]:
        """
        Detects Personal Identifiable Information in a given message using a Lakera endpoint.

        Args:
            message (str): The input message to analyze for prompt injection.
        """
        connection_url = urljoin(self.connection.url, "guard")
        response = self.client.post(
            connection_url,
            headers=self.connection.conn_params,
            json={"messages": [{"content": message, "role": "user"}], "breakdown": True},
            timeout=self.timeout,
        )

        if response.status_code == 200:
            result = [
                item["detector_type"] for item in response.json().get("breakdown", []) if item["detected"] is True
            ]
            return {"is_detected": bool(result), "detected_pii": result}
        elif response.status_code == 401:
            raise ValueError("Invalid API key")
        else:
            raise ValueError(f"Request failed with status code {response.status_code}")
