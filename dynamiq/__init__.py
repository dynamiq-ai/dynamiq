import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from dynamiq.utils.lazy_imports import lazy_exports

if TYPE_CHECKING:
    from .workflow import Workflow

ROOT_PATH = os.path.dirname(__file__)

# litellm loads a .env file when it is imported, unless LITELLM_MODE says otherwise. It used to be imported with
# dynamiq, and connections read their credentials from the environment, so the file is still loaded here.
if os.getenv("LITELLM_MODE", "DEV") == "DEV":
    load_dotenv()

_LAZY_EXPORTS = {
    "Workflow": ".workflow",
}

lazy_exports(__name__, _LAZY_EXPORTS, globals())

__all__ = [
    "ROOT_PATH",
    "Workflow",
]
