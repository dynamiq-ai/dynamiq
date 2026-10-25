from typing import Any

from dynamiq.connections import Runcrate as RuncrateConnection
from dynamiq.nodes.llms.base import BaseLLM


class Runcrate(BaseLLM):
    """Runcrate LLM node.

    This class provides an implementation for Large Language Model node that sends
    requests to Runcrate's OpenAI-compatible API for open models.

    Runcrate model ids are `vendor/model` strings (e.g. `deepseek-ai/DeepSeek-V4.1-Flash`,
    `openai/gpt-oss-120b`). LiteLLM has no Runcrate provider, so requests go through its
    OpenAI-compatible client pointed at the Runcrate endpoint.

    Attributes:
        connection (RuncrateConnection): The connection to use for the Runcrate LLM.
        MODEL_PREFIX (str): The LiteLLM prefix for OpenAI-compatible endpoints.
    """

    connection: RuncrateConnection
    MODEL_PREFIX = "openai/"

    def __init__(self, **kwargs):
        """Initialize the Runcrate LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = RuncrateConnection()
        super().__init__(**kwargs)

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Keep the `openai/` vendor of Runcrate ids such as `openai/gpt-oss-120b`.

        Every Runcrate id has a vendor, so a model that is only `openai/<name>` after the
        LiteLLM prefix is an `openai/` vendor id. LiteLLM strips one `openai/` before
        sending, so add the prefix once more to send the id unchanged.
        """
        params = super().update_completion_params(params)
        model = params.get("model", "")
        if model.startswith(self.MODEL_PREFIX) and "/" not in model[len(self.MODEL_PREFIX) :]:
            params["model"] = f"{self.MODEL_PREFIX}{model}"
        return params
