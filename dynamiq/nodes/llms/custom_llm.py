from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit

from dynamiq.connections import HttpApiKey
from dynamiq.nodes.llms.base import BaseLLM


class CustomLLM(BaseLLM):
    """
    Custom LLM implementation for third-party providers requiring specific formatting.

    This class extends BaseLLM to support providers like OpenRouter, Anthropic, or custom
    endpoints that need special request formatting. It allows adding provider prefixes to
    model names.

    Attributes:
        name (str | None): Name of the LLM node. Defaults to "CustomLLM".
        connection (HttpApiKey): Connection to use for the LLM API.
        provider_prefix (str | None): Provider prefix to add to model names (e.g., "openrouter").
                                     When specified, this will automatically prepend
                                     "{provider_prefix}/" to the model name when sending requests.

    """

    name: str | None = "CustomLLM"
    connection: HttpApiKey
    provider_prefix: str | None = None

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Customizes LLM request parameters for third-party providers.

        This method adds the provider prefix to model name if specified
        (e.g., "openrouter/anthropic/claude-2").

        Args:
            params (dict[str, Any]): The original parameters to be sent to the LLM provider.
                                    This includes model, messages, temperature, etc.

        Returns:
            dict[str, Any]: The modified parameters with proper model name formatting.
        """
        params = super().update_completion_params(params)

        if self.provider_prefix and not params["model"].startswith(self.provider_prefix + "/"):
            params["model"] = f"{self.provider_prefix}/{params['model']}"

        return params

    @staticmethod
    def _is_official_minimax_route(endpoint, protocol, model):
        return (
            endpoint.scheme == "https"
            and endpoint.hostname in {"api.minimax.io", "api.minimaxi.com"}
            and endpoint.port in {None, 443}
            and endpoint.path.rstrip("/") == ("/anthropic" if protocol == "anthropic" else "/v1")
            and protocol in {"anthropic", "openai", "minimax"}
            and model in {"MiniMax-M3.1-Flash", "MiniMax-M3"}
        )

    def _build_stream_response(self, chunks, messages):
        response = super()._build_stream_response(chunks, messages)
        model = self.model.removeprefix(f"{self.provider_prefix}/")
        if self.provider_prefix == "anthropic" and self._is_official_minimax_route(
            urlsplit(self.connection.url), self.provider_prefix, model
        ):
            # LiteLLM 1.103 repeats the completed thought on each signature
            # fragment. Join consecutive signature fragments, not their snapshots.
            # A new thinking delta starts a distinct block, even with identical text.
            blocks = []
            signed_block = None
            for chunk in chunks:
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta.content or delta.tool_calls:
                        signed_block = None
                    for block in getattr(delta, "thinking_blocks", None) or []:
                        if block.get("type") == "redacted_thinking":
                            blocks.append(deepcopy(block))
                            signed_block = None
                        elif block.get("signature"):
                            if signed_block is None:
                                signed_block = deepcopy(block)
                                blocks.append(signed_block)
                            else:
                                signed_block["signature"] += block["signature"]
                        else:
                            signed_block = None
            if blocks:
                response.choices[0].message.thinking_blocks = blocks
        return response

    def _build_completion_params(self, *args, include_sync_client: bool = True, **kwargs) -> dict[str, Any]:
        params = super()._build_completion_params(*args, include_sync_client=include_sync_client, **kwargs)
        endpoint = urlsplit(self.connection.url)
        model = params["model"].removeprefix(f"{self.provider_prefix}/")
        protocol = self.provider_prefix
        if not self._is_official_minimax_route(endpoint, protocol, model):
            return params

        extra_body = deepcopy(params.get("extra_body") or {})
        if model == "MiniMax-M3":
            # M3 ignores effort, including xhigh. Do not let SDK validation reject it
            # or translate it into an instruction to enable thinking.
            for source in (params, extra_body):
                source.pop("reasoning_effort", None)
                if isinstance(source.get("output_config"), dict):
                    source["output_config"] = {k: v for k, v in source["output_config"].items() if k != "effort"}
                    if not source["output_config"]:
                        source.pop("output_config")

        if protocol == "anthropic":
            from dynamiq.nodes.llms._minimax import minimax_anthropic_client

            # Forward native controls after LiteLLM's Claude-specific validation.
            # This is per request: no provider aliases or shared clients are mutated.
            controls = {key: deepcopy(params.pop(key)) for key in ("thinking", "output_config") if key in params}
            effort = params.pop("reasoning_effort", None)
            if effort is not None:
                controls.setdefault("output_config", {})["effort"] = effort
            for key in ("thinking", "output_config"):
                if key in extra_body:
                    controls[key] = extra_body.pop(key)
            if controls:
                params["client"] = minimax_anthropic_client(controls, asynchronous=not include_sync_client)
        if extra_body:
            params["extra_body"] = extra_body
        else:
            params.pop("extra_body", None)
        return params
