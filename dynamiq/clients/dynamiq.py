import json
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import requests

from dynamiq.clients import BaseTracingClient
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.env import get_env_var
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.callbacks.tracing import Run

DYNAMIQ_BASE_URL = "https://api.us-east-1.aws.getdynamiq.ai"


class DynamiqTracingClient(BaseTracingClient):

    def __init__(self, base_url: str = DYNAMIQ_BASE_URL, api_key: str | None = None):
        self.base_url = base_url
        self.api_key = api_key or get_env_var("DYNAMIQ_API_KEY")
        if self.api_key is None:
            raise ValueError("No API key provided")

    def trace(self, runs: list["Run"]) -> None:
        """Trace the given runs.

        Args:
            runs (list["Run"]): List of runs to trace.
        """
        try:
            requests.post(
                urljoin(self.base_url, "/v1/tracing/traces"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                data=json.dumps({"runs": [run.to_dict() for run in runs]}, cls=JsonWorkflowEncoder),
                timeout=60,
            )
        except Exception as e:
            logger.error(f"Failed to send traces. Error {e}")
