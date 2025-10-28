from __future__ import annotations

from typing import Any

from dynamiq.callbacks.base import BaseCallbackHandler
from dynamiq.utils.logger import logger


class AuthRequestLoggingCallback(BaseCallbackHandler):
    """
    Generic callback that logs authentication requests emitted by agents/tools.
    Useful for debugging interactive auth flows during development.
    """

    def on_node_auth_request(self, serialized: dict[str, Any], auth_request: dict[str, Any], **kwargs: Any) -> None:
        node_name = serialized.get("name") or serialized.get("id")
        logger.info(
            "Auth request from node %s: %s",
            node_name,
            auth_request,
        )


__all__ = ["AuthRequestLoggingCallback"]
