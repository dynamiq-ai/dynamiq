from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import HuggingFace, Lakera
from dynamiq.nodes.node import ConnectionNode, ErrorHandling, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig

HUGGINGFACE_URL = "https://api-inference.huggingface.co/models/"


class PromptInjDetectorInputSchema(BaseModel):
    message: str = Field(..., description="Parameter to provide value to validate.")


class PromptInjectionDetector(ConnectionNode):
    group: Literal[NodeGroup.DETECTORS] = NodeGroup.DETECTORS
    name: str = "Prompt Injection Detector"
    description: str = "Node that detects prompt injection."
    connection: HuggingFace | Lakera | None = None
    model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    timeout: float = 30
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(max_retries=1))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[PromptInjDetectorInputSchema]] = PromptInjDetectorInputSchema

    def __init__(self, **kwargs):
        """Initialize the PII Detector.

        If neither client nor connection is provided in kwargs, a new HuggingFace connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = HuggingFace()
        super().__init__(**kwargs)

    def execute(
        self, input_data: PromptInjDetectorInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Detects prompt injection in input data

        Args:
            input_data (PromptInjDetectorInputSchema): input data for the detector, contains message to validate.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing detection results.
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
            try:
                injection_score = next(item["score"] for item in response.json()[0] if item["label"] == "INJECTION")
                safe_score = next(item["score"] for item in response.json()[0] if item["label"] == "SAFE")
                return {"prompt_detected": injection_score > safe_score}
            except Exception:
                raise ValueError("Failed not determine prompt injection.")
        elif response.status_code in [400, 401]:
            raise ValueError("Request failed. Possible causes: bad request or invalid API key.")
        else:
            raise ValueError(f"Request failed with status code {response.status_code}")

    def lakera_detect(self, message: str) -> dict[str, Any]:
        """
        Detects prompt injection attempts in a given message using a Lakera endpoint.

        Args:
            message (str): The input message to analyze for prompt injection.
        """
        connection_url = urljoin(self.connection.url, "guard")
        response = self.client.post(
            connection_url,
            headers=self.connection.conn_params,
            json={"messages": [{"content": message, "role": "user"}], "breakdown": False},
            timeout=self.timeout,
        )
        if response.status_code == 200:
            return {"prompt_detected": response.json().get("flagged")}
        elif response.status_code == 401:
            raise ValueError("Invalid API key")
        else:
            raise ValueError(f"Request failed with status code {response.status_code}")
