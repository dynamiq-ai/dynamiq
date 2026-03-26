import enum
from typing import Any, ClassVar, Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Replicate
from dynamiq.nodes.node import ConnectionNode, ErrorHandling, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig

REPLICATE_URL = "https://api.replicate.com/v1/predictions"


class LLamaGuardModels(str, enum.Enum):
    llama_guard_2_8b = "b063023ee937f28e922982abdbf97b041ffe34ad3b35a53d33e1d74bb19b36c4"


class LLamaGuardInputSchema(BaseModel):
    message: str = Field(..., description="Parameter to provide message to validate.")


class LlamaGuardDetector(ConnectionNode):
    group: Literal[NodeGroup.DETECTORS] = NodeGroup.DETECTORS
    name: str = "Llama Guard"
    description: str = "Guardrail node powered by special version of Llama"
    connection: Replicate
    model: LLamaGuardModels = LLamaGuardModels.llama_guard_2_8b
    timeout: float = 100
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(max_retries=1))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[LLamaGuardInputSchema]] = LLamaGuardInputSchema

    def __init__(self, **kwargs):
        """Initialize the LLama Guard Detector.

        If neither client nor connection is provided in kwargs, a new Replicate connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Replicate()
        super().__init__(**kwargs)

    def execute(self, input_data: LLamaGuardInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        This node is a special guardrail powered by LLama.
        It can work both as a safety evaluator and as policy enforcement, following customizable guidelines.

        Args:
            input_data (LLamaGuardInputSchema): input data for the tool, which includes the message to check.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing check results.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        message = input_data.message
        try:
            headers = {
                "Authorization": f"Bearer {self.connection.api_key}",
                "Content-Type": "application/json",
                "Prefer": "wait",
            }
            data = {
                "version": self.model,
                "input": {
                    "prompt": message,
                },
            }
            response = requests.post(REPLICATE_URL, json=data, headers=headers, timeout=self.timeout)

            if response.status_code in [200, 201]:
                result = response.json().get("output")
                if result is None:
                    raise ValueError("Llama Guard returned no output")
                is_safe = result == "safe"
                if not is_safe:
                    parts = result.split("\n", 1)
                    violated_policies = parts[1].split(",") if len(parts) > 1 else [parts[0]]
                else:
                    violated_policies = []
                return {"is_safe": is_safe, "violated_policies": violated_policies}
            elif response.status_code == 401:
                raise ValueError("Invalid API key.")
            else:
                raise ValueError(f"Request failed with status code {response.status_code}")

        except Exception as e:
            msg = f"Encountered an error while performing validation. \nError details: {e}"
            raise ValueError(msg)
