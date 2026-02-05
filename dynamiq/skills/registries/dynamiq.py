"""Dynamiq API skill registry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import HTTPMethod
from dynamiq.connections.managers import ConnectionManager
from dynamiq.skills.registries.base import BaseSkillRegistry
from dynamiq.skills.types import SkillInstructions, SkillMetadata, SkillRegistryError
from dynamiq.utils.logger import logger


class DynamiqSkillEntry(BaseModel):
    """Allowed skill entry describing which skills are available to the agent."""

    id: str = Field(..., min_length=1, description="Skill identifier.")
    version_id: str = Field(..., min_length=1, description="Skill version identifier.")
    name: str = Field(..., min_length=1, description="Skill name.")
    description: str | None = Field(default=None, description="Optional cached skill description.")


class Dynamiq(BaseSkillRegistry):
    """Dynamiq skill registry implementation."""

    connection: DynamiqConnection = Field(default_factory=DynamiqConnection)
    timeout: float = Field(default=10, description="Timeout in seconds for API requests.")
    allowed_skills: list[DynamiqSkillEntry] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("connection", mode="before")
    @classmethod
    def resolve_connection(cls, v: Any) -> Any:
        """Resolve connection same way as memory backends: None -> default, inline dict -> build instance."""
        if v is None:
            return DynamiqConnection()
        if isinstance(v, DynamiqConnection):
            return v
        if isinstance(v, dict) and v.get("type"):
            conn_cls = ConnectionManager.get_connection_by_type(v["type"])
            return conn_cls(**{k: val for k, val in v.items() if k != "type"})
        return v

    @field_validator("allowed_skills", mode="before")
    @classmethod
    def normalize_allowed_skills(cls, v: Any) -> Any:
        if v is None:
            return []
        return v

    def get_skills_metadata(self) -> list[SkillMetadata]:
        metadata: list[SkillMetadata] = []
        for entry in self.allowed_skills:
            metadata.append(SkillMetadata(name=entry.name, description=entry.description))
        return metadata

    def get_skill_instructions(self, name: str) -> SkillInstructions:
        entry = self._get_entry_by_name(name)
        instructions = self._fetch_skill_instructions(entry.id, entry.version_id)
        return SkillInstructions(
            name=entry.name,
            description=entry.description,
            instructions=instructions,
        )

    def _fetch_skill_instructions(self, skill_id: str, version_id: str) -> str:
        response = self._request(
            HTTPMethod.GET,
            f"/v1/skills/{skill_id}/versions/{version_id}/instructions",
        )
        if not isinstance(response, str):
            raise SkillRegistryError(
                "Unexpected response from Dynamiq skills API. Expected SKILL.md text.",
                details={
                    "skill_id": skill_id,
                    "version_id": version_id,
                    "response_type": type(response).__name__,
                },
            )
        return response

    def _get_entry_by_name(self, name: str) -> DynamiqSkillEntry:
        for entry in self.allowed_skills:
            if name == entry.name:
                return entry
        raise SkillRegistryError("Skill not in allowed skills.", details={"name": name})

    def _request(
        self,
        method: HTTPMethod,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        conn_params = self.connection.conn_params
        base_url = (conn_params.get("api_base") or "").rstrip("/")
        if not base_url:
            raise SkillRegistryError("Dynamiq API base URL is not configured.")

        url = f"{base_url}/{path.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        conn_headers = conn_params.get("headers")
        if isinstance(conn_headers, dict):
            headers.update(conn_headers)

        client = self.connection.connect()
        verb = method.value if isinstance(method, HTTPMethod) else method
        try:
            response = client.request(
                verb,
                url,
                headers=headers,
                params=params,
                json=json,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise SkillRegistryError(f"Failed to call Dynamiq skills API: {exc}") from exc

        if response.status_code >= 400:
            raise SkillRegistryError(
                f"Request to Dynamiq skills API failed: {response.status_code} {response.text}",
                details={"path": path},
            )

        if response.status_code == 204 or not response.content:
            logger.debug("Dynamiq skills API returned empty response for %s", url)
            return ""

        return response.text
