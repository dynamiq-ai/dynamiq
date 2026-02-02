import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

SKILL_NAME_MAX_LENGTH = 64
SKILL_DESCRIPTION_MAX_LENGTH = 1024
SKILL_RESERVED_NAMES = frozenset({"anthropic", "claude"})
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")


def _validate_skill_name(name: str) -> str:
    """Validate skill name: lowercase, alphanumeric, hyphens/underscores; no reserved words."""
    if not name or len(name) > SKILL_NAME_MAX_LENGTH:
        raise ValueError(f"Skill name must be 1–{SKILL_NAME_MAX_LENGTH} characters")
    if not SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError("Skill name must contain only lowercase letters, numbers, hyphens, and underscores")
    if name in SKILL_RESERVED_NAMES:
        raise ValueError(f"Skill name cannot be reserved: {SKILL_RESERVED_NAMES}")
    if "<" in name or ">" in name:
        raise ValueError("Skill name cannot contain XML tags")
    return name


def _validate_skill_description(description: str) -> str:
    """Validate description: non-empty, max length, no XML."""
    if not description or not description.strip():
        raise ValueError("Skill description must be non-empty")
    if len(description) > SKILL_DESCRIPTION_MAX_LENGTH:
        raise ValueError(f"Skill description must be at most {SKILL_DESCRIPTION_MAX_LENGTH} characters")
    if "<" in description or ">" in description:
        raise ValueError("Skill description cannot contain XML tags")
    return description.strip()


class SkillMetadata(BaseModel):
    """Metadata from YAML frontmatter in SKILL.md file.


    Attributes:
        name: Unique identifier (lowercase, numbers, hyphens; max 64 chars)
        version: Semantic version of the skill
        description: What the skill does and when to use it (max 1024 chars)
        author: Optional author information
        tags: List of tags for categorization
        dependencies: List of required Python packages
        supporting_files: List of relative paths to supporting files
        created_at: Timestamp when skill was created
    """

    name: str = Field(..., description="Unique skill identifier (Claude-compliant)")
    version: str = Field(default="1.0.0", description="Semantic version")
    description: str = Field(..., description="Brief description; what and when to use")
    author: str | None = Field(default=None, description="Author information")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    dependencies: list[str] = Field(default_factory=list, description="Required Python packages")
    supporting_files: list[str] = Field(default_factory=list, description="Supporting file paths")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _validate_skill_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        return _validate_skill_description(v)


class Skill(BaseModel):
    """Represents a fully loaded skill with all content and metadata.

    Attributes:
        metadata: Parsed metadata from YAML frontmatter
        instructions: Markdown content after frontmatter
        file_path: Absolute path to the SKILL.md file
        supporting_files_paths: Resolved absolute paths to supporting files
    """

    metadata: SkillMetadata = Field(..., description="Skill metadata")
    instructions: str = Field(..., description="Markdown instructions")
    file_path: Path = Field(..., description="Path to SKILL.md file")
    supporting_files_paths: list[Path] = Field(
        default_factory=list,
        description="Absolute paths to supporting files"
    )

    @property
    def name(self) -> str:
        """Get skill name from metadata."""
        return self.metadata.name

    @property
    def summary(self) -> str:
        """Get short summary for agent memory."""
        return f"{self.metadata.name}: {self.metadata.description}"

    def get_full_content(self) -> str:
        """Returns complete skill content for loading into agent context.

        Returns:
            Formatted markdown with skill name header and instructions
        """
        return f"# Skill: {self.metadata.name}\n\n{self.instructions}"


class SkillWhitelistEntry(BaseModel):
    """Whitelist entry for skills (e.g. from config or API).

    Used to restrict which skills are available and to pass id/version for API-backed skills.
    """

    id: str = Field(..., description="Skill ID (e.g. UUID for API skills)")
    name: str = Field(..., description="Skill name")
    description: str = Field(..., description="Brief description")
    version_id: str = Field(..., description="Version ID (e.g. UUID for API, or 'latest')")


class SkillReference(BaseModel):
    """Lightweight reference to a skill (progressive disclosure).

    Used when discovering skills without loading full content.

    Attributes:
        name: Skill identifier
        description: Brief description
        file_path: Path to SKILL.md (as string for serialization); optional for API-backed skills
        tags: Categorization tags
        id: Optional skill ID (for API-backed skills)
        version_id: Optional version ID (for API-backed skills)
    """

    name: str = Field(..., description="Skill identifier")
    description: str = Field(..., description="Brief description")
    file_path: str = Field(default="", description="Path to SKILL.md file (empty for API-backed skills)")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    id: str | None = Field(default=None, description="Skill ID (e.g. UUID for API skills)")
    version_id: str | None = Field(default=None, description="Version ID (for API-backed skills)")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _validate_skill_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        return _validate_skill_description(v)
