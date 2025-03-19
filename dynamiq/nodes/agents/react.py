import json
import re
import types
from enum import Enum
from typing import Any, Union, get_args, get_origin

from litellm import get_supported_openai_params, supports_function_calling
from pydantic import Field, model_validator

from dynamiq.nodes.agents.base import Agent, AgentIntermediateStep, AgentIntermediateStepModelObservation
from dynamiq.nodes.agents.exceptions import ActionParsingException, MaxLoopsExceededException, RecoverableAgentException
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.prompts import Message, MessageRole, VisionMessage
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingMode
from dynamiq.utils.logger import logger

REACT_BLOCK_TOOLS = """
You have access to a variety of tools,
and you are responsible for using them in any order you choose to complete the task:\n
{tool_description}

Input formats for tools:
{input_formats}
"""

REACT_BLOCK_TOOLS_NO_FORMATS = """
You have access to a variety of tools,
and you are responsible for using them in any order you choose to complete the task:\n
{tool_description}
"""

REACT_BLOCK_NO_TOOLS = """Always follow this exact format in your responses:

Thought: [Your detailed reasoning about the user's question]
Answer: [Your complete answer to the user's question]

IMPORTANT RULES:
- ALWAYS start with "Thought:" to explain your reasoning process
- Provide a clear, direct answer after your thought
- If you cannot fully answer, explain why in your thought
- Be thorough and helpful in your response
- Do not mention tools or actions since you don't have access to any
"""

REACT_BLOCK_XML_INSTRUCTIONS = """Always use this exact XML format in your responses:
<output>
    <thought>
        [Your detailed reasoning about what to do next]
    </thought>
    <action>
        [Tool name from ONLY [{tools_name}]]
    </action>
    <action_input>
        [JSON input for the tool]
    </action_input>
</output>

After each action, you'll receive:
Observation: [Result from the tool]

When you have enough information to provide a final answer:
<output>
    <thought>
        [Your reasoning for the final answer]
    </thought>
    <answer>
        [Your complete answer to the user's question]
    </answer>
</output>

For questions that don't require tools:
<output>
    <thought>
        [Your reasoning about the question]
    </thought>
    <answer>
        [Your direct response]
    </answer>
</output>

IMPORTANT RULES:
- ALWAYS include <thought> tags with detailed reasoning
- For tool use, include action and action_input tags
- For direct answers, only include thought and answer tags
- Ensure action_input contains valid JSON with double quotes
- Properly close all XML tags
- Do not use markdown formatting inside XML
"""  # noqa: E501


REACT_BLOCK_INSTRUCTIONS = """Always follow this exact format in your responses:

Thought: [Your detailed reasoning about what to do next]
Action: [Tool name from ONLY [{tools_name}]]
Action Input: [JSON input for the tool]

After each action, you'll receive:
Observation: [Result from the tool]

When you have enough information to provide a final answer:
Thought: [Your reasoning for the final answer]
Answer: [Your complete answer to the user's question]

For questions that don't require tools:
Thought: [Your reasoning about the question]
Answer: [Your direct response]

IMPORTANT RULES:
- ALWAYS start with "Thought:" even for simple responses
- Ensure Action Input is valid JSON without markdown formatting
- Use proper JSON syntax with double quotes for keys and string values
- Never use markdown code blocks (```) around your JSON
- JSON must be properly formatted with correct commas and brackets
- Only use tools from the provided list
- If you can answer directly, use only Thought followed by Answer
"""  # noqa: E501


REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT = """If you have sufficient information to provide final answer, provide your final answer in one of these two formats:
If you can answer on request:
{{thought: [Why you can provide final answer],
action: finish
action_input: [Response for request]}}

If you can't answer on request:
{{thought: [Why you can not answer on request],
action: finish
answer: [Response for request]}}

Structure you responses in JSON format.
{{thought: [Your reasoning about the next step],
action: [The tool you choose to use, if any from ONLY [{tools_name}]],
action_input: [JSON input in correct format you provide to the tool]}}

IMPORTANT RULES:
- You MUST ALWAYS include "thought" as the FIRST field in your JSON
- Each tool has a specific input format you must strictly follow
- In action_input field, provide properly formatted JSON with double quotes
- Avoid using extra backslashes
- Do not use markdown code blocks around your JSON
"""  # noqa: E501


REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING = """
You have to call appropriate functions.

Function descriptions:
plan_next_action - function that should be called to use tools [{tools_name}]].
provide_final_answer - function that should be called when answer on initial request can be provided.
Call this function if initial user input does not have any actionable request.
"""  # noqa: E501


REACT_BLOCK_INSTRUCTIONS_NO_TOOLS = """
Always structure your responses in this exact format:

Thought: [Your detailed reasoning about the user's question]
Answer: [Your complete response to the user's question]

IMPORTANT RULES:
- ALWAYS begin with "Thought:" to show your reasoning process
- Use the "Thought" section to analyze the question and plan your response
- Only after thinking through the problem, provide your answer
- If you cannot fully answer, explain why in your thinking
- Be thorough and helpful in your response
- Do not mention tools or actions as you don't have access to any

"""  # noqa: E501

REACT_BLOCK_XML_INSTRUCTIONS_NO_TOOLS = """Always use this exact XML format in your responses:
<output>
    <thought>
        [Your detailed reasoning about the question]
    </thought>
    <answer>
        [Your direct response to the user's question]
    </answer>
</output>

IMPORTANT RULES:
- ALWAYS include <thought> tags with detailed reasoning
- Only use thought and answer tags
- Properly close all XML tags
- Do not use markdown formatting inside XML
- Do not mention tools or actions since you don't have access to any
"""


REACT_BLOCK_OUTPUT_FORMAT = (
    "In your final answer, avoid phrases like 'based on the information gathered or provided.' "
)


REACT_MAX_LOOPS_PROMPT = """
You are tasked with providing a final answer based on information gathered during a process that has reached its maximum number of loops.
Your goal is to analyze the given context and formulate a clear, concise response.
First, carefully review the history, which contains thoughts and information gathered during the process.

Analyze the context to identify key information, patterns, or partial answers that can contribute to a final response. Pay attention to any progress made, obstacles encountered, or partial results obtained.
Based on your analysis, attempt to formulate a final answer to the original question or task. Your answer should be:
1. Fully supported by the information found in the context
2. Clear and concise
3. Directly addressing the original question or task, if possible
If you cannot provide a full answer based on the given context, explain that due to limitations in the number of steps or potential issues with the tools used, you are unable to fully answer the question. In this case, suggest one or more of the following:
1. Increasing the maximum number of loops for the agent setup
2. Reviewing the tools settings
3. Revising the input task description
Important: Do not mention specific errors in tools, exact steps, environments, code, or search results. Keep your response general and focused on the task at hand.
Provide your final answer or explanation within <answer> tags.
Your response should be clear, concise, and professional.
<answer>
[Your final answer or explanation goes here]
</answer>
"""  # noqa: E501


final_answer_function_schema = {
    "type": "function",
    "strict": True,
    "function": {
        "name": "provide_final_answer",
        "description": "Function should be called when if you can answer the initial request"
        " or if there is not request at all.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning about why you can answer original question.",
                },
                "answer": {"type": "string", "description": "Answer on initial request."},
            },
            "required": ["thought", "answer"],
        },
    },
}


TYPE_MAPPING = {
    int: "integer",
    float: "float",
    bool: "boolean",
    str: "string",
}


class ReActAgent(Agent):
    """Agent that uses the ReAct strategy for processing tasks by interacting with tools in a loop."""

    name: str = "React Agent"
    max_loops: int = Field(default=15, ge=2)
    inference_mode: InferenceMode = InferenceMode.DEFAULT
    behaviour_on_max_loops: Behavior = Field(
        default=Behavior.RAISE,
        description="Define behavior when max loops are exceeded. Options are 'raise' or 'return'.",
    )
    format_schema: list = []

    def log_reasoning(self, thought: str, action: str, action_input: str, loop_num: int) -> None:
        """
        Logs reasoning step of agent.

        Args:
            thought (str): Reasoning about next step.
            action (str): Chosen action.
            action_input (str): Input to the tool chosen by action.
            loop_num (int): Number of reasoning loop.
        """
        logger.info(
            "\n------------------------------------------\n"
            f"Agent {self.name}: Loop {loop_num}:\n"
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {action_input}"
            "\n------------------------------------------"
        )

    def log_final_output(self, final_output: str, loop_num: int) -> None:
        """
        Logs final output of the agent.

        Args:
            final_output (str): Final output of agent.
            loop_num (int): Number of reasoning loop
        """
        logger.info(
            "\n------------------------------------------\n"
            f"Agent {self.name}: Loop {loop_num + 1}\n"
            f"Final answer: {final_output}"
            "\n------------------------------------------\n"
        )

    @model_validator(mode="after")
    def validate_inference_mode(self):
        """Validate whether specified model can be inferenced in provided mode."""
        match self.inference_mode:
            case InferenceMode.FUNCTION_CALLING:
                if not supports_function_calling(model=self.llm.model):
                    raise ValueError(f"Model {self.llm.model} does not support function calling")

            case InferenceMode.STRUCTURED_OUTPUT:
                params = get_supported_openai_params(model=self.llm.model)
                if "response_format" not in params:
                    raise ValueError(f"Model {self.llm.model} does not support structured output")

        return self

    def parse_xml_content(self, text: str, tag: str) -> str:
        """Extract content from XML-like tags."""
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def parse_xml_and_extract_info(self, text: str) -> dict[str, Any]:
        """Parse XML-like structure and extract action and action_input."""
        output_content = self.parse_xml_content(text, "output")
        thought = self.parse_xml_content(output_content, "thought")
        action = self.parse_xml_content(output_content, "action")
        action_input_text = self.parse_xml_content(output_content, "action_input")

        if not thought or not action or not action_input_text:
            missing = []
            if not thought:
                missing.append("thought")
            if not action:
                missing.append("action")
            if not action_input_text:
                missing.append("action_input")

            error_message = (
                f"Error: Missing required XML tags: {', '.join(missing)}. "
                "Your response must include complete <output>, <thought>, <action>, "
                "and <action_input> tags."
            )
            raise ActionParsingException(error_message, recoverable=True)

        try:
            action_input = json.loads(action_input_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error: Unable to parse action and action input due to invalid JSON formatting. {e}")
            error_message = (
                "Error: Unable to parse action and action input due to invalid JSON formatting. "
                "Multiline strings are not allowed in JSON unless properly escaped. "
                "Ensure all newlines (\\n), quotes, and special characters are escaped. "
                "For example:\n\n"
                "Correct:\n"
                "{\n"
                '  "key": "Line 1\\nLine 2",\n'
                '  "code": "print(\\"Hello, World!\\")"\n'
                "}\n\n"
                "Incorrect:\n"
                "{\n"
                '  "key": "Line 1\nLine 2",\n'
                '  "code": "print("Hello, World!")"\n'
                "}\n\n"
                f"JSON Parsing Error Details: {e}"
            )
            raise ActionParsingException(error_message, recoverable=True)

        return thought, action, action_input

    def extract_output_and_answer_xml(self, text: str) -> tuple[str, Any]:
        """Extract output and answer from XML-like structure."""
        output = self.parse_xml_content(text, "output")
        answer = self.parse_xml_content(text, "answer")
        return {"output": output, "answer": answer}

    def _parse_thought(self, output: str) -> tuple[str | None, str | None]:
        """Extracts thought from the output string."""
        thought_match = re.search(
            r"Thought:\s*(.*?)Action",
            output,
            re.DOTALL,
        )

        if thought_match:
            return thought_match.group(1).strip()

        return ""

    def _parse_action(self, output: str) -> tuple[str | None, str | None]:
        """Parses the action, its input, and thought from the output string."""
        try:
            action_match = re.search(
                r"Action:\s*(.*?)\nAction Input:\s*(({\n)?.*?)(?:[^}]*$)",
                output,
                re.DOTALL,
            )
            if action_match:
                action = action_match.group(1).strip()
                action_input = action_match.group(2).strip()
                if "```json" in action_input:
                    action_input = action_input.replace("```json", "").replace("```", "").strip()

                action_input = json.loads(action_input)
                return self._parse_thought(output), action, action_input
            else:
                raise ActionParsingException()
        except Exception as e:
            raise ActionParsingException(
                (
                    f"Error {e}: Unable to parse thought, action and action input or thought and answer."
                    "Please rewrite using the correct Thought/Action/Action Input format"
                    "with action input as a valid dictionary."
                    "Or in case of direct answer or clarification, use Thought/Answer format."
                    "Ensure all quotes are included."
                ),
                recoverable=True,
            )

    def tracing_final(self, loop_num, final_answer, config, kwargs):
        self._intermediate_steps[loop_num]["final_answer"] = final_answer

    def tracing_intermediate(self, loop_num, formatted_prompt, llm_generated_output):
        self._intermediate_steps[loop_num] = AgentIntermediateStep(
            input_data={"prompt": formatted_prompt},
            model_observation=AgentIntermediateStepModelObservation(
                initial=llm_generated_output,
            ),
        ).model_dump()

    def _run_agent(
        self,
        input_message: Message | VisionMessage,
        history_messages: list[Message] | None = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        """
        Executes the ReAct strategy by iterating through thought, action, and observation cycles.
        Args:
            config (RunnableConfig | None): Configuration for the agent run.
            **kwargs: Additional parameters for running the agent.
        Returns:
            str: Final answer provided by the agent.
        Raises:
            RuntimeError: If the maximum number of loops is reached without finding a final answer.
            Exception: If an error occurs during execution.
        """
        if self.verbose:
            logger.info(f"Agent {self.name} - {self.id}: Running ReAct strategy")

        system_message = Message(
            role=MessageRole.SYSTEM,
            content=self.generate_prompt(
                tools_name=self.tool_names,
                input_formats=self.generate_input_formats(self.tools),
            ),
        )

        if history_messages:
            self._prompt.messages = [system_message, *history_messages, input_message]
        else:
            self._prompt.messages = [system_message, input_message]

        stop_sequences = []

        if self.inference_mode == InferenceMode.XML:
            stop_sequences.extend(["<observation>"])
        elif self.inference_mode == InferenceMode.DEFAULT:
            stop_sequences.extend(["Observation: "])
        self.llm.stop = stop_sequences

        for loop_num in range(self.max_loops):

            try:
                llm_result = self._run_llm(
                    self._prompt.messages,
                    config=config,
                    schema=self.format_schema,
                    inference_mode=self.inference_mode,
                    **kwargs,
                )
                action, action_input = None, None
                llm_generated_output = ""
                llm_reasoning = (
                    llm_result.output.get("content")[:200]
                    if llm_result.output.get("content")
                    else str(llm_result.output.get("tool_calls", ""))[:200]
                )
                logger.info(f"Agent {self.name} - {self.id}: Loop {loop_num + 1}, " f"reasoning:\n{llm_reasoning}...")

                match self.inference_mode:
                    case InferenceMode.DEFAULT:
                        llm_generated_output = llm_result.output.get("content", "")

                        self.tracing_intermediate(loop_num, self._prompt.messages, llm_generated_output)

                        if "Answer:" in llm_generated_output:
                            final_answer = self._extract_final_answer(llm_generated_output)
                            self.log_final_output(final_answer, loop_num + 1)
                            self.tracing_final(loop_num, final_answer, config, kwargs)
                            if self.streaming.enabled:
                                self.stream_content(
                                    content=final_answer,
                                    source=self.name,
                                    step="answer",
                                    config=config,
                                    **kwargs,
                                )

                            return final_answer

                        thought, action, action_input = self._parse_action(llm_generated_output)
                        self.log_reasoning(thought, action, action_input, loop_num + 1)

                        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                            self.stream_content(
                                content={"thought": thought, "action": action, "action_input": action_input},
                                source=self.name,
                                step=f"reasoning_{loop_num + 1}",
                                config=config,
                                by_tokens=False,
                                **kwargs,
                            )

                    case InferenceMode.FUNCTION_CALLING:
                        if self.verbose:
                            logger.info(f"Agent {self.name} - {self.id}: using function calling inference mode")

                        if "tool_calls" not in dict(llm_result.output):
                            logger.error("Error: No function called.")
                            raise ActionParsingException(
                                "Error: No function called, you need to call the correct function."
                            )

                        action = list(llm_result.output["tool_calls"].values())[0]["function"]["name"].strip()
                        llm_generated_output_json = list(llm_result.output["tool_calls"].values())[0]["function"][
                            "arguments"
                        ]

                        llm_generated_output = json.dumps(llm_generated_output_json)

                        self.tracing_intermediate(loop_num, self._prompt.messages, llm_generated_output)

                        if action == "provide_final_answer":
                            final_answer = llm_generated_output_json["answer"]
                            self.log_final_output(final_answer, loop_num + 1)
                            self.tracing_final(loop_num, final_answer, config, kwargs)
                            if self.streaming.enabled:
                                self.stream_content(
                                    content=final_answer,
                                    source=self.name,
                                    step="answer",
                                    config=config,
                                    **kwargs,
                                )
                            return final_answer

                        thought = llm_generated_output_json["thought"]
                        action_input = llm_generated_output_json["action_input"]

                        self.log_reasoning(thought, action, action_input, loop_num + 1)

                        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                            self.stream_content(
                                content={"thought": thought, "action": action, "action_input": action_input},
                                source=self.name,
                                step=f"reasoning_{loop_num + 1}",
                                config=config,
                                by_tokens=False,
                                **kwargs,
                            )

                    case InferenceMode.STRUCTURED_OUTPUT:
                        if self.verbose:
                            logger.info(f"Agent {self.name} - {self.id}: using structured output inference mode")

                        llm_generated_output = llm_result.output["content"]
                        self.tracing_intermediate(loop_num, self._prompt.messages, llm_generated_output)
                        llm_generated_output_json = json.loads(llm_generated_output)

                        thought = llm_generated_output_json["thought"]
                        action = llm_generated_output_json["action"]
                        action_input = llm_generated_output_json["action_input"]

                        if action == "finish":
                            self.log_final_output(action_input, loop_num + 1)
                            self.tracing_final(loop_num, action_input, config, kwargs)
                            if self.streaming.enabled:
                                self.stream_content(
                                    content=action_input,
                                    source=self.name,
                                    step="answer",
                                    config=config,
                                    **kwargs,
                                )
                            return action_input

                        action_input = json.loads(action_input)
                        self.log_reasoning(thought, action, action_input, loop_num + 1)

                        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                            self.stream_content(
                                content={"thought": thought, "action": action, "action_input": action_input},
                                source=self.name,
                                step=f"reasoning_{loop_num + 1}",
                                config=config,
                                by_tokens=False,
                                **kwargs,
                            )

                    case InferenceMode.XML:
                        if self.verbose:
                            logger.info(f"Agent {self.name} - {self.id}: using XML inference mode")

                        llm_generated_output = llm_result.output["content"]
                        self.tracing_intermediate(loop_num, self._prompt.messages, llm_generated_output)

                        if "<answer>" in llm_generated_output:
                            final_answer = self._extract_final_answer_xml(llm_generated_output)
                            self.log_final_output(final_answer, loop_num + 1)
                            self.tracing_final(loop_num, final_answer, config, kwargs)
                            if self.streaming.enabled:
                                self.stream_content(
                                    content=final_answer,
                                    source=self.name,
                                    step="answer",
                                    config=config,
                                    **kwargs,
                                )
                            return final_answer

                        thought, action, action_input = self.parse_xml_and_extract_info(llm_generated_output)
                        self.log_reasoning(thought, action, action_input, loop_num + 1)

                        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                            self.stream_content(
                                content={"thought": thought, "action": action, "action_input": action_input},
                                source=self.name,
                                step=f"reasoning_{loop_num + 1}",
                                config=config,
                                by_tokens=False,
                                **kwargs,
                            )

                self._prompt.messages.append(Message(role=MessageRole.ASSISTANT, content=llm_generated_output))

                if action:
                    if self.tools:

                        try:
                            tool = self._get_tool(action)
                            tool_result = self._run_tool(tool, action_input, config, **kwargs)

                        except RecoverableAgentException as e:
                            tool_result = f"{type(e).__name__}: {e}"

                        observation = f"\nObservation: {tool_result}\n"
                        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                            self.stream_content(
                                content={"name": tool.name, "input": action_input, "result": tool_result},
                                source=tool.name if tool else action,
                                step=f"tool_{loop_num}",
                                config=config,
                                by_tokens=False,
                                **kwargs,
                            )

                        self._intermediate_steps[loop_num]["model_observation"].update(
                            AgentIntermediateStepModelObservation(
                                tool_using=action,
                                tool_input=action_input,
                                tool_output=tool_result,
                                updated=llm_generated_output,
                            ).model_dump()
                        )
                        self._prompt.messages.append(Message(role=MessageRole.USER, content=observation))

            except ActionParsingException as e:
                self._prompt.messages.append(
                    Message(role=MessageRole.ASSISTANT, content="Response is:" + llm_generated_output)
                )
                self._prompt.messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=f"Fix the reasoning error: {type(e).__name__}: {e}",
                    )
                )
                continue

        if self.behaviour_on_max_loops == Behavior.RAISE:
            error_message = (
                f"Agent {self.name} (ID: {self.id}) has reached the maximum loop limit of {self.max_loops} without finding a final answer. "  # noqa: E501
                f"Last response: {self._prompt.messages[-1].content}\n"
                f"Consider increasing the maximum number of loops or reviewing the task complexity to ensure completion."  # noqa: E501
            )
            raise MaxLoopsExceededException(message=error_message)
        else:
            max_loop_final_answer = self._handle_max_loops_exceeded(config, **kwargs)
            if self.streaming.enabled:
                self.stream_content(
                    content=max_loop_final_answer,
                    source=self.name,
                    step="answer",
                    config=config,
                    **kwargs,
                )
            return max_loop_final_answer

    def _handle_max_loops_exceeded(self, config: RunnableConfig | None = None, **kwargs) -> str:
        """
        Handle the case where max loops are exceeded by crafting a thoughtful response.
        """
        self._prompt.messages.append(Message(role=MessageRole.USER, content=REACT_MAX_LOOPS_PROMPT))
        llm_final_attempt = self._run_llm(self._prompt.messages, config=config, **kwargs).output["content"]
        self._run_depends = [NodeDependency(node=self.llm).to_dict()]
        final_answer = self.parse_xml_content(llm_final_attempt, "answer")

        return f"{final_answer}"

    def _extract_final_answer_xml(self, llm_output: str) -> str:
        """Extract the final answer from the LLM output."""
        final_answer = self.extract_output_and_answer_xml(llm_output)
        return final_answer["answer"]

    def generate_input_formats(self, tools: list[Node]) -> str:
        """Generate formatted input descriptions for each tool."""
        input_formats = []
        for tool in tools:
            params = []
            for name, field in tool.input_schema.model_fields.items():
                if not field.json_schema_extra or field.json_schema_extra.get("is_accessible_to_agent", True):
                    if get_origin(field.annotation) in (Union, types.UnionType):
                        type_str = str(field.annotation)
                    else:
                        type_str = getattr(field.annotation, "__name__", str(field.annotation))

                    description = field.description or "No description"
                    params.append(f"{name} ({type_str}): {description}")

            input_formats.append(f" - {self.sanitize_tool_name(tool.name)}\n \t* " + "\n\t* ".join(params))
        return "\n".join(input_formats)

    def generate_structured_output_schemas(self):
        tool_names = [self.sanitize_tool_name(tool.name) for tool in self.tools]

        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "plan_next_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "required": ["thought", "action", "action_input"],
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Your reasoning about the next step.",
                        },
                        "action": {
                            "type": "string",
                            "description": f"Next action to make (choose from [{tool_names}, finish]).",
                        },
                        "action_input": {
                            "type": "string",
                            "description": "Input for chosen action.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }

        self.format_schema = schema

    @staticmethod
    def filter_format_type(param_type: str | type) -> str:
        """Filters proper type for a function calling schema."""

        if get_origin(param_type) in (Union, types.UnionType):
            param_type = next((arg for arg in get_args(param_type) if arg is not type(None)), None)

        return param_type

    def generate_property_schema(self, properties, name, field):
        if not field.json_schema_extra or field.json_schema_extra.get("is_accessible_to_agent", True):
            description = field.description or "No description"
            description += f" Defaults to: {field.default}." if field.default else ""
            param = self.filter_format_type(field.annotation)

            if param_type := TYPE_MAPPING.get(param):
                properties[name] = {"type": param_type, "description": description}

            elif issubclass(param, Enum):
                element_type = TYPE_MAPPING.get(
                    self.filter_format_type(type(list(field.annotation.__members__.values())[0].value))
                )
                properties[name] = {
                    "type": element_type,
                    "description": description,
                    "enum": [field.value for field in field.annotation.__members__.values()],
                }

            elif param.__origin__ is list:
                properties[name] = {"type": "array", "items": {"type": TYPE_MAPPING.get(param.__args__[0])}}

    def generate_function_calling_schemas(self):
        """Generate schemas for function calling."""
        self.format_schema.append(final_answer_function_schema)
        for tool in self.tools:
            properties = {}
            for name, field in tool.input_schema.model_fields.items():
                self.generate_property_schema(properties, name, field)

            schema = {
                "type": "function",
                "function": {
                    "name": self.sanitize_tool_name(tool.name),
                    "description": tool.description[:1024],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Your reasoning about why you can answer original question.",
                            },
                            "action_input": {
                                "type": "object",
                                "description": "Input for chosen action.",
                                "properties": properties,
                                "required": list(properties.keys()),
                                "additionalProperties": False,
                            },
                        },
                        "additionalProperties": False,
                        "required": ["thought", "action_input"],
                    },
                    "strict": True,
                },
            }

            self.format_schema.append(schema)

    def _init_prompt_blocks(self):
        """Initialize the prompt blocks required for the ReAct strategy."""
        super()._init_prompt_blocks()

        prompt_blocks = {
            "tools": "" if not self.tools else REACT_BLOCK_TOOLS,
            "instructions": REACT_BLOCK_INSTRUCTIONS_NO_TOOLS if not self.tools else REACT_BLOCK_INSTRUCTIONS,
            "output_format": REACT_BLOCK_OUTPUT_FORMAT,
        }

        match self.inference_mode:
            case InferenceMode.FUNCTION_CALLING:
                self.generate_function_calling_schemas()
                prompt_blocks["instructions"] = REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING
                if self.tools:
                    prompt_blocks["tools"] = REACT_BLOCK_TOOLS_NO_FORMATS

            case InferenceMode.STRUCTURED_OUTPUT:
                self.generate_structured_output_schemas()
                prompt_blocks["instructions"] = REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT

            case InferenceMode.XML:
                prompt_blocks["instructions"] = (
                    REACT_BLOCK_XML_INSTRUCTIONS_NO_TOOLS if not self.tools else REACT_BLOCK_XML_INSTRUCTIONS
                )

        self._prompt_blocks.update(prompt_blocks)
