"""Data models for skills."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class SkillMetadata(BaseModel):
    """Metadata from YAML frontmatter in SKILL.md file.

    Attributes:
        name: Unique identifier for the skill
        version: Semantic version of the skill
        description: Brief description of what the skill does
        author: Optional author information
        tags: List of tags for categorization
        dependencies: List of required Python packages
        supporting_files: List of relative paths to supporting files
        created_at: Timestamp when skill was created
    """

    name: str = Field(..., description="Unique skill identifier")
    version: str = Field(default="1.0.0", description="Semantic version")
    description: str = Field(..., description="Brief description of the skill")
    author: str | None = Field(default=None, description="Author information")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    dependencies: list[str] = Field(default_factory=list, description="Required Python packages")
    supporting_files: list[str] = Field(default_factory=list, description="Supporting file paths")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


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


class SkillReference(BaseModel):
    """Lightweight reference to a skill (for agent memory).

    This is used when discovering skills without loading full content.
    Keeps memory footprint minimal while providing enough info for
    the agent to decide when to load a skill.

    Attributes:
        name: Skill identifier
        description: Brief description
        file_path: Path to SKILL.md (as string for serialization)
        tags: Categorization tags
    """

    name: str = Field(..., description="Skill identifier")
    description: str = Field(..., description="Brief description")
    file_path: str = Field(..., description="Path to SKILL.md file")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
