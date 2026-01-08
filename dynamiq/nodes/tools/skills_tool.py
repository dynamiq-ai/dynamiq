from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.skills.loader import SkillLoader
from dynamiq.skills.models import Skill
from dynamiq.storages.file.base import FileStore
from dynamiq.utils.logger import logger


class SkillsToolInputSchema(BaseModel):
    """Input schema for Skills tool.

    Attributes:
        action: Action to perform (list, load, unload, refresh)
        skill_name: Name of skill to load/unload (required for load/unload)
    """

    action: Literal["list", "load", "unload", "refresh"] = Field(
        ...,
        description=(
            "Action to perform: 'list' available skills, 'load' a skill, "
            "'unload' a skill, or 'refresh' skill list"
        )
    )
    skill_name: str | None = Field(
        default=None,
        description="Name of skill to load/unload (required for load/unload actions)"
    )


class SkillsTool(Node):
    """Tool for managing agent skills - discovering, loading, and unloading.

    This tool provides the agent with the ability to:
    - List all available skills in the FileStore
    - Load a skill's full content into context
    - Unload a skill to free up context space
    - Refresh the list of available skills

    Skills are loaded on-demand to optimize context usage through
    progressive disclosure. Skills are stored in FileStore under the
    .skills/ prefix.

    Attributes:
        file_store: FileStore instance containing skill files
        skills_prefix: Prefix path for skills in FileStore
        _loader: Internal SkillLoader instance
        _loaded_skills: Cache of currently loaded skills
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SkillsTool"
    description: str = (
        "Manages available skills for the agent. Use this tool to:\n"
        "- List all available skills with their descriptions\n"
        "- Load a skill to access its full instructions and resources\n"
        "- Unload a skill when no longer needed to free context\n"
        "- Refresh the skills list after adding new skills to FileStore\n\n"
        "Example: {\"action\": \"list\"} or {\"action\": \"load\", \"skill_name\": \"document_creator\"}"
    )

    file_store: FileStore = Field(
        ...,
        description="FileStore instance containing skill files"
    )
    skills_prefix: str = Field(
        default=".skills/",
        description="Prefix path for skills in FileStore"
    )
    input_schema: ClassVar[type[SkillsToolInputSchema]] = SkillsToolInputSchema

    _loader: SkillLoader | None = None
    _loaded_skills: dict[str, Skill] = {}

    def __init__(self, **kwargs):
        """Initialize the SkillsTool."""
        super().__init__(**kwargs)
        self._loader = SkillLoader(self.file_store, self.skills_prefix)
        self._loaded_skills = {}

    def execute(
        self, input_data: SkillsToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute skills management action.

        Args:
            input_data: Validated input with action and optional skill_name
            config: Optional runtime configuration
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with 'content' key containing action results

        Raises:
            ToolExecutionException: If action fails or input is invalid
        """
        action = input_data.action

        logger.info(f"SkillsTool: executing action '{action}'")

        if action == "list":
            return self._list_skills()
        elif action == "load":
            if not input_data.skill_name:
                raise ToolExecutionException(
                    "skill_name required for load action",
                    recoverable=True
                )
            return self._load_skill(input_data.skill_name)
        elif action == "unload":
            if not input_data.skill_name:
                raise ToolExecutionException(
                    "skill_name required for unload action",
                    recoverable=True
                )
            return self._unload_skill(input_data.skill_name)
        elif action == "refresh":
            return self._refresh_skills()
        else:
            raise ToolExecutionException(
                f"Unknown action: {action}",
                recoverable=True
            )

    def _list_skills(self) -> dict[str, Any]:
        """List all available skills.

        Returns:
            Dictionary with list of available skills and their metadata
        """
        skills = self._loader.discover_skills()

        skills_info = []
        for skill in skills:
            skills_info.append({
                "name": skill.name,
                "description": skill.description,
                "tags": skill.tags,
                "loaded": skill.name in self._loaded_skills
            })

        logger.info(f"SkillsTool: listed {len(skills)} available skills")

        return {
            "content": {
                "available_skills": skills_info,
                "total": len(skills),
                "loaded_count": len(self._loaded_skills)
            }
        }

    def _load_skill(self, skill_name: str) -> dict[str, Any]:
        """Load a skill and return its full content.

        Args:
            skill_name: Name of the skill to load

        Returns:
            Dictionary with skill status and full instructions

        Raises:
            ToolExecutionException: If skill not found
        """
        if skill_name in self._loaded_skills:
            logger.info(f"SkillsTool: skill '{skill_name}' already loaded")
            return {
                "content": {
                    "status": "already_loaded",
                    "message": f"Skill '{skill_name}' is already loaded",
                    "skill": self._loaded_skills[skill_name].get_full_content()
                }
            }

        skill = self._loader.load_skill(skill_name)

        if not skill:
            raise ToolExecutionException(
                f"Skill '{skill_name}' not found in FileStore",
                recoverable=True
            )

        self._loaded_skills[skill_name] = skill

        logger.info(f"SkillsTool: successfully loaded skill '{skill_name}'")

        return {
            "content": {
                "status": "loaded",
                "message": f"Successfully loaded skill '{skill_name}'",
                "skill_name": skill.name,
                "description": skill.metadata.description,
                "instructions": skill.get_full_content(),
                "supporting_files": [str(p) for p in skill.supporting_files_paths],
                "dependencies": skill.metadata.dependencies
            }
        }

    def _unload_skill(self, skill_name: str) -> dict[str, Any]:
        """Unload a skill from memory.

        Args:
            skill_name: Name of the skill to unload

        Returns:
            Dictionary with unload status
        """
        if skill_name not in self._loaded_skills:
            logger.info(f"SkillsTool: skill '{skill_name}' is not currently loaded")
            return {
                "content": {
                    "status": "not_loaded",
                    "message": f"Skill '{skill_name}' is not currently loaded"
                }
            }

        del self._loaded_skills[skill_name]

        logger.info(f"SkillsTool: successfully unloaded skill '{skill_name}'")

        return {
            "content": {
                "status": "unloaded",
                "message": f"Successfully unloaded skill '{skill_name}'"
            }
        }

    def _refresh_skills(self) -> dict[str, Any]:
        """Refresh the skills list by re-scanning FileStore.

        Returns:
            Dictionary with refresh status and updated skill count
        """
        skills = self._loader.discover_skills()

        logger.info(f"SkillsTool: refreshed skills list, found {len(skills)} skills")

        return {
            "content": {
                "status": "refreshed",
                "message": f"Refreshed skills list. Found {len(skills)} skills",
                "available_skills": [skill.name for skill in skills]
            }
        }
