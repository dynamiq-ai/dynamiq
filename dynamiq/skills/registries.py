from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import HTTPMethod
from dynamiq.skills.models import SkillInstructions, SkillMetadata, SkillRegistryError
from dynamiq.utils.logger import logger


class BaseSkillRegistry(ABC, BaseModel):
    """Abstract base class for skill registries."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__}.{self.__class__.__name__}"

    @abstractmethod
    def get_skills_metadata(self) -> list[SkillMetadata]:
        raise NotImplementedError

    @abstractmethod
    def get_skill_instructions(self, name: str) -> SkillInstructions:
        raise NotImplementedError


class DynamiqSkillWhitelistEntry(BaseModel):
    """Whitelist entry describing which skills are available to the agent."""

    id: str = Field(..., min_length=1, description="Skill identifier.")
    version_id: str | None = Field(default=None, description="Skill version identifier.")
    name: str | None = Field(default=None, description="Optional cached skill name.")
    description: str | None = Field(default=None, description="Optional cached skill description.")


class Dynamiq(BaseSkillRegistry):
    """Dynamiq skill registry implementation."""

    connection: DynamiqConnection = Field(default_factory=DynamiqConnection)
    timeout: float = Field(default=10, description="Timeout in seconds for API requests.")
    whitelist: list[DynamiqSkillWhitelistEntry] = Field(default_factory=list)

    @field_validator("whitelist", mode="before")
    @classmethod
    def normalize_whitelist(cls, v: Any):
        if v is None:
            return []
        return v

    def get_skills_metadata(self) -> list[SkillMetadata]:
        metadata: list[SkillMetadata] = []
        for entry in self.whitelist:
            metadata.append(SkillMetadata(name=entry.name, description=entry.description))
        return metadata

    def get_skill_instructions(self, name: str) -> SkillInstructions:
        entry = self._get_whitelist_entry_by_name(name)
        instructions = self._fetch_skill_instructions(entry.id, entry.version_id)
        return SkillInstructions(name=entry.name, description=entry.description, instructions=instructions)

    def _fetch_skill_instructions(self, skill_id: str, version_id: str) -> str:
        response = self._request(
            HTTPMethod.GET,
            f"/v1/skills/{skill_id}/versions/{version_id}/instructions",
        )
        if not isinstance(response, str):
            raise SkillRegistryError(
                "Unexpected response from Dynamiq skills API. Expected SKILL.md text.",
                details={"skill_id": skill_id, "version_id": version_id, "response_type": type(response).__name__},
            )
        return response

    def _get_whitelist_entry_by_name(self, name: str) -> DynamiqSkillWhitelistEntry:
        for entry in self.whitelist:
            if entry.name == name:
                if not entry.version_id:
                    raise SkillRegistryError(
                        "Skill whitelist entry is missing version_id.",
                        details={"skill_id": entry.id, "name": name},
                    )
                return entry
        raise SkillRegistryError("Skill not found in whitelist.", details={"name": name})

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
                details={"skill_id": path},
            )

        if response.status_code == 204 or not response.content:
            logger.debug("Dynamiq skills API returned empty response for %s", url)
            return {}

        try:
            return response.json()
        except ValueError:
            logger.debug("Received non-JSON response from Dynamiq skills API for %s: %s", url, response.text)
            return response.text


class LocalSkillWhitelistEntry(BaseModel):
    """Whitelist entry for local skill registry."""

    name: str = Field(..., min_length=1, description="Skill name.")
    description: str | None = Field(default=None, description="Optional cached skill description.")


class Local(BaseSkillRegistry):
    """Local filesystem-backed skill registry."""

    base_path: str = Field(default="~/.dynamiq/skills", description="Base path for local skills.")
    whitelist: list[LocalSkillWhitelistEntry] = Field(default_factory=list)

    def get_skills_metadata(self) -> list[SkillMetadata]:
        metadata: list[SkillMetadata] = []
        for entry in self.whitelist:
            metadata.append(SkillMetadata(name=entry.name, description=entry.description))
        return metadata

    def get_skill_instructions(self, name: str) -> SkillInstructions:
        skill_path = self._resolve_skill_path(name)
        if not skill_path.exists():
            raise SkillRegistryError(
                "Local skill instructions not found.",
                details={"name": name, "path": str(skill_path)},
            )
        instructions = skill_path.read_text(encoding="utf-8")
        return SkillInstructions(name=name, instructions=instructions)

    def _resolve_skill_path(self, skill_name: str) -> Path:
        base = Path(self.base_path).expanduser()
        return base / skill_name / "SKILL.md"
