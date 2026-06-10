from typing import Any

from dynamiq.connections import Cohere as CohereConnection
from dynamiq.nodes.llms._strict import SubsetStrictToolsNoFlagMixin
from dynamiq.nodes.llms.base import BaseLLM


class Cohere(SubsetStrictToolsNoFlagMixin, BaseLLM):
    """Cohere LLM node.

    This class provides an implementation for the Cohere Language Model node.

    Attributes:
        connection (CohereConnection): The connection to use for the Cohere LLM.
        strict_tools: Inherited from :class:`BaseLLM`. Cohere controls strictness
            at the request level (Chat API V2's ``strict_tools`` boolean), not via
            a per-tool ``strict`` field (which it rejects as an unknown field). When
            enabled, each tool's schema is tightened by
            :class:`SubsetStrictToolsNoFlagMixin` (no ``strict`` key) and
            ``strict_tools: true`` is forwarded on the request. Cohere requires
            Command-r7b-or-newer, at least one ``required`` parameter per tool, and
            a maximum of 200 fields across all tools (the feature is experimental).
            A whitelist is treated as "enable strict": Cohere's flag is
            request-level and applies to every tool in the call.
    """
    connection: CohereConnection

    def __init__(self, **kwargs):
        """Initialize the Cohere LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = CohereConnection()
        super().__init__(**kwargs)

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Forward Cohere's request-level ``strict_tools`` flag when strict is on.

        LiteLLM doesn't expose ``strict_tools`` as a supported Cohere param, so we
        inject it via ``extra_body`` — LiteLLM's HTTP handler merges ``extra_body``
        into the request body, reaching Cohere's Chat API V2. Only sent when strict
        is requested and tools are present.
        """
        params = super().update_completion_params(params)
        if self.strict_tools and params.get("tools"):
            params.setdefault("extra_body", {})["strict_tools"] = True
        return params
