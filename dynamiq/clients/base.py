import abc
import json
import logging
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import requests
from fastapi import APIRouter

from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.env import get_env_var

if TYPE_CHECKING:
    from dynamiq.callbacks.tracing import Run

router = APIRouter()
logger = logging.getLogger(__name__)

BASE_URL = "https://api.sandbox.getdynamiq.ai"


class BaseTracingClient(abc.ABC):
    """Base class for tracing clients."""

    def __init__(self, base_url: str = BASE_URL, api_key: str | None = None, project_id: str | None = None):
        self.base_url = base_url
        self.api_key = api_key or get_env_var("DYNAMIQ_API_KEY")
        self.project_id = project_id or get_env_var("DYNAMIQ_PROJECT_ID")

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
