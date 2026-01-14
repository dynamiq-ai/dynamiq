"""Skill discovery and loading functionality."""

import re
from pathlib import Path

import yaml

from dynamiq.skills.models import Skill, SkillMetadata, SkillReference
from dynamiq.storages.file.base import FileStore
from dynamiq.utils.logger import logger


class SkillLoader:
    """Discovers and loads skills from FileStore.

    Scans FileStore for SKILL.md files, parses YAML frontmatter,
    and provides both lightweight references and full skill content.

    Attributes:
        file_store: FileStore instance for skill storage
        skills_prefix: Prefix path for skills in FileStore (default: ".skills/")
    """

    def __init__(self, file_store: FileStore, skills_prefix: str = ".skills/"):
        """Initialize the skill loader.

        Args:
            file_store: FileStore instance containing skill files
            skills_prefix: Prefix path for skills in FileStore
        """
        self.file_store = file_store
        self.skills_prefix = skills_prefix.rstrip("/") + "/"

    def discover_skills(self) -> list[SkillReference]:
        """Scan FileStore and return lightweight skill references.

        Discovers all SKILL.md files in the FileStore under skills_prefix
        and extracts only metadata for efficient discovery.

        Returns:
            List of skill references with name, description, and path
        """
        skills = []

        try:
            # List all files in FileStore
            all_files = self.file_store.list_files(recursive=True)

            # Find SKILL.md files
            for file_info in all_files:
                file_path = getattr(file_info, "path", file_info)
                file_path = str(file_path)
                if file_path.startswith(self.skills_prefix) and file_path.endswith("/SKILL.md"):
                    try:
                        skill_ref = self._parse_skill_reference(file_path)
                        skills.append(skill_ref)
                    except Exception as e:
                        logger.error(f"Failed to parse skill {file_path}: {e}")

            logger.info(f"Discovered {len(skills)} skills in FileStore")
        except Exception as e:
            logger.warning(f"Failed to discover skills in FileStore: {e}")

        return skills

    def load_skill(self, name: str) -> Skill | None:
        """Load full skill content by name.

        Args:
            name: Skill identifier (directory name)

        Returns:
            Fully loaded Skill object or None if not found
        """
        skill_path = f"{self.skills_prefix}{name}/SKILL.md"

        if not self.file_store.exists(skill_path):
            logger.error(f"Skill not found: {name} at {skill_path}")
            return None

        try:
            return self._parse_skill_file(skill_path, name)
        except Exception as e:
            logger.error(f"Failed to load skill {name}: {e}")
            return None

    def _parse_skill_reference(self, skill_path: str) -> SkillReference:
        """Parse SKILL.md to extract metadata only (lightweight).

        Args:
            skill_path: Path to SKILL.md file in FileStore

        Returns:
            SkillReference with basic metadata

        Raises:
            ValueError: If YAML frontmatter is missing or invalid
        """
        # Retrieve file content from FileStore
        content_bytes = self.file_store.retrieve(skill_path)
        content = content_bytes.decode('utf-8')

        # Extract YAML frontmatter
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if not match:
            raise ValueError(f"No YAML frontmatter found in {skill_path}")

        metadata_dict = yaml.safe_load(match.group(1))

        # Validate required fields
        if 'name' not in metadata_dict:
            raise ValueError(f"Missing required field 'name' in {skill_path}")
        if 'description' not in metadata_dict:
            raise ValueError(f"Missing required field 'description' in {skill_path}")

        return SkillReference(
            name=metadata_dict.get('name'),
            description=metadata_dict.get('description', ''),
            file_path=skill_path,
            tags=metadata_dict.get('tags', [])
        )

    def _parse_skill_file(self, skill_path: str, skill_name: str) -> Skill:
        """Parse complete SKILL.md file with full content.

        Args:
            skill_path: Path to SKILL.md file in FileStore
            skill_name: Name of the skill

        Returns:
            Fully loaded Skill object with metadata and instructions

        Raises:
            ValueError: If SKILL.md format is invalid
        """
        # Retrieve file content from FileStore
        content_bytes = self.file_store.retrieve(skill_path)
        content = content_bytes.decode('utf-8')

        # Extract YAML frontmatter and markdown content
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid SKILL.md format in {skill_path}")

        metadata_dict = yaml.safe_load(match.group(1))
        instructions = match.group(2).strip()

        # Validate and create metadata
        metadata = SkillMetadata(**metadata_dict)

        # Resolve supporting files in FileStore
        skill_dir = f"{self.skills_prefix}{skill_name}/"
        supporting_paths = []
        for rel_path in metadata.supporting_files:
            file_path_in_store = skill_dir + rel_path
            if self.file_store.exists(file_path_in_store):
                # Store as Path for compatibility, but it's actually a FileStore path
                supporting_paths.append(Path(file_path_in_store))
            else:
                logger.warning(
                    f"Supporting file not found: {file_path_in_store} for skill {metadata.name}"
                )

        return Skill(
            metadata=metadata,
            instructions=instructions,
            file_path=Path(skill_path),  # Store as Path for compatibility
            supporting_files_paths=supporting_paths
        )
