"""Dynamiq API skill source: discover and load skills via GET /v1/skills/.../instructions."""

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field

from dynamiq.connections import HTTPMethod
from dynamiq.skills.models import Skill, SkillMetadata, SkillReference, SkillWhitelistEntry
from dynamiq.skills.sources.base import SkillSource
from dynamiq.utils.logger import logger


class DynamiqSkillSourceError(Exception):
    """Error raised by Dynamiq skill source."""

    pass


class DynamiqSkillSource(SkillSource):
    """Skill source that loads skills from the Dynamiq API.

    Uses whitelist (id, name, description, version_id) for discovery and
    GET /v1/skills/{skill_id}/versions/{version_id}/instructions for content.
    """

    name: str = "DynamiqSkillSource"
    connection: Any = Field(..., description="Dynamiq connection (dynamiq.connections.Dynamiq)")
    whitelist: list[SkillWhitelistEntry] | None = Field(
        default=None,
        description="Whitelist of skills (id, name, description, version_id). Used for discover_skills.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _base_path: str = "/v1/skills"

    def discover_skills(self) -> list[SkillReference]:
        """Return skill references from whitelist (id, name, description, version_id)."""
        if not self.whitelist:
            logger.warning("DynamiqSkillSource: no whitelist; discover_skills returns empty")
            return []
        refs = []
        for entry in self.whitelist:
            refs.append(
                SkillReference.model_construct(
                    name=entry.name,
                    description=entry.description,
                    file_path="",
                    tags=[],
                    id=entry.id,
                    version_id=entry.version_id,
                )
            )
        return refs

    def load_skill(self, name: str) -> Skill | None:
        """Load full skill by name: resolve id/version_id from whitelist and fetch instructions."""
        entry = self._entry_by_name(name)
        if not entry:
            return None
        instructions = self._fetch_instructions(entry.id, entry.version_id)
        if instructions is None:
            return None
        metadata = SkillMetadata.model_construct(
            name=entry.name,
            description=entry.description,
            version="1.0.0",
            tags=[],
            dependencies=[],
            supporting_files=[],
            created_at=datetime.now(),
        )
        return Skill.model_construct(
            metadata=metadata,
            instructions=instructions,
            file_path=Path(""),
            supporting_files_paths=[],
        )

    def _entry_by_name(self, name: str) -> SkillWhitelistEntry | None:
        if not self.whitelist:
            return None
        for e in self.whitelist:
            if e.name == name:
                return e
        return None

    def _fetch_instructions(self, skill_id: str, version_id: str) -> str | None:
        """GET /v1/skills/{skill_id}/versions/{version_id}/instructions."""
        path = f"{self._base_path}/{skill_id}/versions/{version_id}/instructions"
        try:
            response = self._request(HTTPMethod.GET, path)
        except Exception as e:
            logger.exception("DynamiqSkillSource: failed to fetch instructions for %s/%s: %s", skill_id, version_id, e)
            return None
        if response is None:
            return None
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            instructions = response.get("instructions") or response.get("content") or response.get("body")
            if isinstance(instructions, str):
                return instructions
            if instructions is not None:
                return str(instructions)
        return None

    def _request(
        self,
        method: HTTPMethod,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        """Execute HTTP request against Dynamiq API."""
        conn_params = self.connection.conn_params
        base_url = (conn_params.get("api_base") or "").rstrip("/")
        if not base_url:
            raise DynamiqSkillSourceError("Dynamiq API base URL is not configured.")
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
                timeout=30,
            )
        except Exception as exc:
            raise DynamiqSkillSourceError(f"Failed to call Dynamiq API: {exc}") from exc
        if response.status_code >= 400:
            raise DynamiqSkillSourceError(f"Request to Dynamiq API failed: {response.status_code} {response.text}")
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text
