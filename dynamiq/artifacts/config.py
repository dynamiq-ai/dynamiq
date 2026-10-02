from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.artifacts.backends.base import ArtifactBackend


class ArtifactConfig(BaseModel):
    """Configuration for an agent's artifacts.

    Attributes:
        enabled: Whether the agent gets an artifact tool at all.
        backend: The backend holding the artifacts.
    """

    enabled: bool = False
    backend: ArtifactBackend = Field(..., description="Backend holding the agent's artifacts.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """The backend serializes itself, so exclude it from the plain dump."""
        return {"backend": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the config to a dictionary, delegating the backend to its own ``to_dict``."""
        for_tracing = kwargs.pop("for_tracing", False)
        if for_tracing and not self.enabled:
            return {"enabled": False}
        include_secure_params = kwargs.pop("include_secure_params", False)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["backend"] = self.backend.to_dict(for_tracing=for_tracing, include_secure_params=include_secure_params)
        return data
