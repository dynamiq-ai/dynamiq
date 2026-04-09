from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.skills import BaseSkillRegistry
from dynamiq.skills.utils import normalize_sandbox_skills_base_path
from dynamiq.utils.logger import logger


class SkillsToolAction(str, Enum):
    """Action for the Skills tool."""

    LIST = "list"
    GET = "get"


class SkillsToolInputSchema(BaseModel):
    """Input schema for Skills tool. Actions: list (discover), get (full content)."""

    action: SkillsToolAction = Field(
        ...,
        description="Action: 'list' discover skills, 'get' full skill content.",
    )
    skill_name: str | None = Field(default=None, description="Skill name (required for get)")


class SkillsTool(Node):
    """Tool for skills: discover and get content from a skill registry (Dynamiq or FileSystem).

    After get, apply the skill's instructions yourself and provide the result in your final answer.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "skills-tool"
    description: str = (
        "Manages skills (instructions and optional scripts). "
        "Available skills are already listed in the system prompt — "
        "do not call 'list' to re-discover them unless that list is missing. Use this tool to:\n"
        "- Get skill content: action='get', skill_name='...' "
        "— use only when skills are NOT available in the sandbox. "
        "When a sandbox is available and skills have been "
        "ingested (e.g. under /home/user/skills/), prefer reading "
        "skill content from the sandbox via sandbox-shell "
        "(e.g. cat /home/user/skills/<name>/SKILL.md or grep for a section) "
        "instead of calling get.\n\n"
        "When get returns a 'scripts_path', or when using the "
        "sandbox, scripts are under <skill_dir>/scripts/: "
        "run them via the sandbox (cd <path> then run scripts).\n\n"
        "After reading skill content (from sandbox or get), apply"
        " the instructions yourself and provide the result "
        "in your final answer. Do not call the tool again with"
        " user content to transform; the tool only provides "
        "instructions; you produce the output."
    )

    skill_registry: BaseSkillRegistry = Field(
        ...,
        description="Registry providing skills (Dynamiq or FileSystem).",
    )
    input_schema: ClassVar[type[SkillsToolInputSchema]] = SkillsToolInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {
            "skill_registry": True,
        }

    def execute(
        self, input_data: SkillsToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        action = input_data.action
        logger.info("SkillsTool - action=%s", action.value)

        if action == SkillsToolAction.LIST:
            return self._list_skills()
        if action == SkillsToolAction.GET:
            if not input_data.skill_name:
                raise ToolExecutionException("skill_name required for get", recoverable=True)
            return self._get_skill(input_data.skill_name)
        raise ToolExecutionException(f"Unknown action: {action.value}", recoverable=True)

    def _list_skills(self) -> dict[str, Any]:
        metadata_list = self.skill_registry.get_skills_metadata()
        base = normalize_sandbox_skills_base_path(getattr(self.skill_registry, "sandbox_skills_base_path", None))
        skills_info = []
        for m in metadata_list:
            entry: dict[str, Any] = {"name": m.name, "description": m.description}
            if base:
                entry["sandbox_path"] = f"{base}/{m.name}/SKILL.md"
            scripts_path = self.skill_registry.get_skill_scripts_path(m.name)
            if scripts_path:
                entry["scripts_path"] = scripts_path
            skills_info.append(entry)
        names = [m.name for m in metadata_list]
        logger.info("SkillsTool - list: %d skill(s) %s", len(metadata_list), names)
        return {
            "content": {
                "available_skills": skills_info,
                "total": len(metadata_list),
            }
        }

    def _get_skill(self, skill_name: str) -> dict[str, Any]:
        try:
            instructions = self.skill_registry.get_skill_instructions(skill_name)
        except Exception as e:
            raise ToolExecutionException(f"Failed to get skill '{skill_name}': {e}", recoverable=True) from e

        out: dict[str, Any] = {
            "name": instructions.name,
            "description": instructions.description,
            "instructions": instructions.instructions,
        }
        if instructions.metadata:
            out["metadata"] = instructions.metadata
        path = self.skill_registry.get_skill_scripts_path(skill_name)
        if path:
            out["scripts_path"] = path

        one_line = instructions.instructions.replace("\n", " ").strip()
        preview = (one_line[:50] + "...") if len(one_line) > 50 else one_line
        logger.info(
            "SkillsTool - get: skill=%s -> content received (%d chars), preview: %s",
            skill_name,
            len(instructions.instructions),
            preview,
        )
        return {"content": out}
