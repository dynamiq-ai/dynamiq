import re

from dynamiq.nodes.agents.base import Agent
from dynamiq.prompts import Message, MessageRole, VisionMessage
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingMode
from dynamiq.utils.logger import logger

REFLECTION_REFLECT_PROMPT: str = (
    "You're an AI assistant that responds to the user with maximum accuracy. "
    "To do so, you will first think about what the user is asking for, thinking step by step. "
    "During the thinking phase, you will have reflections that will help you clarify ambiguities. "
    "In each reflection, you will list the possibilities and finally choose one. "
    "Between reflections, you can think again. At the end of thinking, you must draw a conclusion. "
    "You only need to generate the minimum text that will help you generate the better output. "
    "Don't be verbose while thinking. Finally, you will generate an output based on the previous thinking. "
    "This is the format to follow:"
    "<thinking>"
    "Here you will think about the user's request"
    "<reflection>"
    "Here you will reflect on the thinking"
    "</reflection>"
    "<reflection>"
    "This is another reflection"
    "</reflection>"
    "</thinking>"
    "<output>"
    "Here you will generate the output based on the thinking"
    "</output>"
    "Always use these tags in your responses. Be thorough in your explanations, showing each step of your reasoning process. "  # noqa: E501
    "Aim to be precise and logical in your approach, and don't hesitate to break down complex problems into simpler components. "  # noqa: E501
    "Your tone should be analytical and slightly formal, focusing on clear communication of your thought process. "
    "Remember: Both <thinking> and <reflection> and <output> MUST be tags and must be closed at their conclusion. "
    "Make sure all <tags> are on separate lines with no other text. "
    "Do not include other text on a line containing a tag."
)


class ReflectionAgent(Agent):
    name: str = "Agent Reflection"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_prompt_blocks()

    def _init_prompt_blocks(self):
        super()._init_prompt_blocks()
        prompt_blocks = {
            "instructions": REFLECTION_REFLECT_PROMPT,
        }
        self._prompt_blocks.update(prompt_blocks)

    @staticmethod
    def extract_tag_content(tag_name: str, text: str) -> list[str]:
        """
        Extracts content from <tag_name> tags in the given text.

        Args:
            text (str): The input text containing <tag_name> tags.

        Returns:
            List[str]: A list of content found within <tag_name> tags.
        """
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        output_content = re.findall(pattern, text, re.DOTALL)

        if not output_content:
            pattern = rf"<{tag_name}>(.*)"
            output_content = re.findall(pattern, text, re.DOTALL)

        return [content.strip() for content in output_content]

    def _run_agent(
        self,
        input_message: Message | VisionMessage,
        history_messages: list[Message] | None = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        try:
            system_message = Message(
                role=MessageRole.SYSTEM,
                content=self.generate_prompt(block_names=["introduction", "role", "date", "instructions", "context"]),
            )

            if history_messages:
                self._prompt.messages = [system_message, *history_messages, input_message]
            else:
                self._prompt.messages = [system_message, input_message]

            result = self._run_llm(self._prompt.messages, config=config, **kwargs).output["content"]

            self._prompt.messages.append(Message(role=MessageRole.ASSISTANT, content=result))

            output_content = self.extract_tag_content("output", result)
            reflection_content = self.extract_tag_content("reflection", result)

            if self.verbose:
                logger.info(f"Agent {self.name} - {self.id}: LLM output by REFLECTION prompt:\n{result[:200]}...")
            if self.streaming.enabled:
                if not output_content:
                    return ""

                if self.streaming.mode == StreamingMode.ALL:
                    self.stream_content(
                        content={"output_content": output_content[:-1], "reflection": reflection_content},
                        step="reasoning",
                        source=self.name,
                        config=config,
                        by_tokens=False,
                        **kwargs,
                    )

                return self.stream_content(
                    content=output_content[-1],
                    step="answer",
                    source=self.name,
                    config=config,
                    **kwargs,
                )

            if not output_content:
                return ""

            return output_content[-1]
        except Exception as e:
            raise e
