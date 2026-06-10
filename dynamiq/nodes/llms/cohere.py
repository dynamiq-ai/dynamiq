from typing import Any

from dynamiq.connections import Cohere as CohereConnection
from dynamiq.nodes.llms._strict import to_strict_subset_function
from dynamiq.nodes.llms.base import BaseLLM


class Cohere(BaseLLM):
    """Cohere LLM node.

    This class provides an implementation for the Cohere Language Model node.

    Attributes:
        connection (CohereConnection): The connection to use for the Cohere LLM.
        strict_tools: Inherited from :class:`BaseLLM`. Cohere controls strictness
            at the request level (Chat API V2's ``strict_tools`` boolean), not via
            a per-tool ``strict`` field (which it rejects as an unknown field). When
            enabled, each tool's schema is tightened by :meth:`_to_strict_function`
            (no ``strict`` key) and ``strict_tools: true`` is forwarded on the
            request. Cohere requires Command-r7b-or-newer, at least one ``required``
            parameter per tool, and a maximum of 200 fields across all tools (the
            feature is experimental). A whitelist is treated as "enable strict":
            Cohere's flag is request-level and applies to every tool in the call.
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

    def _to_strict_function(self, fn: dict) -> dict:
        """Tighten one tool's schema to the strict subset WITHOUT the ``strict`` flag.

        Cohere rejects an unknown per-tool ``strict`` field, so we only tighten the
        schema (via :func:`to_strict_subset_function` with ``attach_flag=False``);
        strictness itself is enabled request-level in :meth:`update_completion_params`.
        """
        return to_strict_subset_function(fn, attach_flag=False)

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
