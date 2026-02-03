from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.runnables import RunnableConfig
from dynamiq.skills import BaseSkillRegistry
from dynamiq.skills.utils import extract_skill_content_slice
from dynamiq.utils.logger import logger


class SkillsToolInputSchema(BaseModel):
    """Input schema for Skills tool. Actions: list (discover), get (full or partial content)."""

    action: Literal["list", "get"] = Field(
        ...,
        description="Action: 'list' discover skills, 'get' full or partial skill content.",
    )
    skill_name: str | None = Field(default=None, description="Skill name (required for get)")
    section: str | None = Field(
        default=None,
        description="For get: return only this markdown section (e.g. 'Welcome messages')",
    )
    line_start: int | None = Field(default=None, description="For get: 1-based start line (body only)")
    line_end: int | None = Field(default=None, description="For get: 1-based end line (inclusive)")


class SkillsTool(Node):
    """Tool for skills: discover and get content from a skill registry (Dynamiq or Local).

    After get, apply the skill's instructions yourself and provide the result in your final answer.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SkillsTool"
    description: str = (
        "Manages skills (instructions only). Use this tool to:\n"
        "- List available skills: action='list'\n"
        "- Get skill content: action='get', skill_name='...'. "
        "For large skills use section='Section title' or line_start/line_end to read only a part.\n\n"
        "After get, apply the skill's instructions yourself in your reasoning and provide the result "
        "in your final answer. Do not call the tool again with user content to transform; "
        "the tool only provides instructions; you produce the output."
    )

    skill_registry: BaseSkillRegistry = Field(
        ...,
        description="Registry providing skills (Dynamiq or Local).",
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
        logger.info("SkillsTool - action=%s", action)

        if action == "list":
            return self._list_skills()
        if action == "get":
            if not input_data.skill_name:
                raise ToolExecutionException("skill_name required for get", recoverable=True)
            return self._get_skill(
                input_data.skill_name,
                section=input_data.section,
                line_start=input_data.line_start,
                line_end=input_data.line_end,
            )
        raise ToolExecutionException(f"Unknown action: {action}", recoverable=True)

    def _list_skills(self) -> dict[str, Any]:
        metadata_list = self.skill_registry.get_skills_metadata()
        skills_info = [{"name": m.name, "description": m.description} for m in metadata_list]
        names = [m.name for m in metadata_list]
        logger.info("SkillsTool - list: %d skill(s) %s", len(metadata_list), names)
        return {
            "content": {
                "available_skills": skills_info,
                "total": len(metadata_list),
            }
        }

    def _get_skill(
        self,
        skill_name: str,
        section: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any]:
        try:
            instructions = self.skill_registry.get_skill_instructions(skill_name)
        except Exception as e:
            raise ToolExecutionException(f"Skill '{skill_name}' not found: {e}", recoverable=True) from e

        if section is not None or line_start is not None or line_end is not None:
            sliced, section_used = extract_skill_content_slice(
                instructions.instructions,
                section=section,
                line_start=line_start,
                line_end=line_end,
            )
            out = {
                "skill_name": instructions.name,
                "description": instructions.description,
                "instructions": sliced,
                "section_used": section_used,
            }
            logger.info(
                "SkillsTool - get: skill=%s (section=%s, lines=%s-%s) -> content received",
                skill_name,
                section,
                line_start,
                line_end,
            )
            return {"content": out}

        logger.info(
            "SkillsTool - get: skill=%s -> content received (%d chars)",
            skill_name,
            len(instructions.instructions),
        )
        return {
            "content": {
                "skill_name": instructions.name,
                "description": instructions.description,
                "instructions": instructions.instructions,
            }
        }
