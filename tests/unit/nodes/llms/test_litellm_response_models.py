"""litellm's response models are built once, up front, so parallel LLM nodes never build them concurrently."""

from litellm.types import utils as litellm_types
from litellm.types.utils import Delta, ModelResponse, ModelResponseStream, StreamingChoices, Usage
from pydantic import BaseModel, ConfigDict

from dynamiq.nodes.llms.base import _build_litellm_response_models
from dynamiq.utils.litellm import get_litellm


def test_response_models_are_complete_once_litellm_is_loaded():
    get_litellm()

    for model in (ModelResponse, ModelResponseStream, Delta, StreamingChoices, Usage):
        assert model.__pydantic_complete__, model.__name__


def test_incomplete_model_is_built(monkeypatch):
    class Deferred(BaseModel):
        model_config = ConfigDict(defer_build=True)
        value: int

    monkeypatch.setattr(Deferred, "__module__", litellm_types.__name__)
    monkeypatch.setattr(litellm_types, "Deferred", Deferred, raising=False)
    assert not Deferred.__pydantic_complete__

    _build_litellm_response_models()

    assert Deferred.__pydantic_complete__
