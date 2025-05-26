import json
import re
import types
from enum import Enum
from typing import Any, Union, get_args, get_origin

from litellm import get_supported_openai_params, supports_function_calling
from pydantic import Field, model_validator

from dynamiq.nodes.agents.base import Agent, AgentIntermediateStep, AgentIntermediateStepModelObservation
from dynamiq.nodes.agents.exceptions import (
    ActionParsingException,
    AgentUnknownToolException,
    JSONParsingError,
    MaxLoopsExceededException,
    RecoverableAgentException,
    TagNotFoundError,
    XMLParsingError,
)
from dynamiq.nodes.agents.utils import XMLParser
from dynamiq.nodes.llms.gemini import Gemini
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.prompts import Message, MessageRole, VisionMessage, VisionMessageTextContent
from dynamiq.runnables import RunnableConfig
from dynamiq.types.llm_tool import Tool
from dynamiq.types.streaming import StreamingMode
from dynamiq.utils.logger import logger

REACT_BLOCK_TOOLS = """
You have access to a variety of tools,
and you are responsible for using them in any order you choose to complete the task:\n
{tool_description}

Input formats for tools:
{input_formats}

Note: For tools not listed in the input formats section,
refer to their descriptions in the AVAILABLE TOOLS section for usage instructions.
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
- For all tags other than <answer>, text content should ideally be XML-escaped.
- Special characters like & should be escaped as &amp; in <thought> and other tags, but can be used directly in <answer>
- Do not use markdown formatting (like ```) inside XML tags *unless* it's within the <answer> tag.
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
Always structure your responses in this JSON format:

{{thought: [Your reasoning about the next step],
action: [The tool you choose to use, if any from ONLY [{tools_name}]],
action_input: [JSON input in correct format you provide to the tool]}}

After each action, you'll receive:
Observation: [Result from the tool]

When you have enough information to provide a final answer:
{{thought: [Your reasoning for the final answer],
action: finish
action_input: [Response for initial request]}}

For questions that don't require tools:
{{thought: [Your reasoning for the final answer],
action: finish
action_input: [Your direct response]}}

IMPORTANT RULES:
- You MUST ALWAYS include "thought" as the FIRST field in your JSON
- Each tool has a specific input format you must strictly follow
- In action_input field, provide properly formatted JSON with double quotes
- Avoid using extra backslashes
- Do not use markdown code blocks around your JSON
- Never keep action_input empty.
"""  # noqa: E501


REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING = """
You need to use the right functions based on what the user asks.

Use the function `provide_final_answer` when you can give a clear answer to the user's first question,
 and no extra steps, tools, or work are needed.
Call this function if the user's input is simple and doesn’t require additional help or tools.

If the user's request requires the use of specific tools, such as [{tools_name}],
 you must first call the appropriate function to invoke those tools.
Only after utilizing the necessary tools and gathering the required information should
 you call `provide_final_answer` to deliver the final response.

Make sure to check each request carefully to see if you can answer it right away or if you need to use tools to help.
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
You are tasked with providing a final answer for initial user question based on information gathered during a process that has reached its maximum number of loops.
Your goal is to analyze the given context and formulate a clear, concise response.
First, carefully review the information gathered during the process, tool calls and their outputs.

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
    _tools: list[Tool] = []
    _response_format: dict[str, Any] | None = None

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

    def log_final_output(self, thought: str, final_output: str, loop_num: int) -> None:
        """
        Logs final output of the agent.

        Args:
            final_output (str): Final output of agent.
            loop_num (int): Number of reasoning loop
        """
        logger.info(
            "\n------------------------------------------\n"
            f"Agent {self.name}: Loop {loop_num}\n"
            f"Thought: {thought}\n"
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

    def _parse_action(self, output: str) -> tuple[str | None, str | None, dict | None]:
        """
        Parses the action, its input, and thought from the output string.

        Args:
            output (str): The input string containing Thought, Action, and Action Input.

        Returns:
            Tuple[Optional[str], Optional[str], Optional[dict]]: (thought, action, action_input)
            where thought is the extracted thought, action is the action name,
            and action_input is the parsed JSON input.

        Raises:
            ActionParsingException: If the output format is invalid or parsing fails.
        """
        try:
            thought_pattern = r"Thought:\s*(.*?)(?:Action:|$)"
            action_pattern = r"Action:\s*(.*?)\nAction Input:\s*((?:{\n)?.*?)(?:[^}]*$)"

            thought_match = re.search(thought_pattern, output, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else None

            action_match = re.search(action_pattern, output, re.DOTALL)
            if not action_match:
                raise ActionParsingException(
                    "No valid Action and Action Input found. "
                    "Ensure the format is 'Thought: ... Action: ... Action Input: ...' "
                    "with a valid dictionary as input.",
                    recoverable=True,
                )

            action = action_match.group(1).strip()
            raw_input = action_match.group(2).strip()

            json_markers = ["```json", "```JSON", "```"]
            for marker in json_markers:
                raw_input = raw_input.replace(marker, "").strip()
            try:
                action_input = json.loads(raw_input)
            except json.JSONDecodeError as e:
                raise ActionParsingException(
                    f"Invalid JSON in Action Input: {str(e)}. Ensure the Action Input is a valid JSON dictionary.",
                    recoverable=True,
                )

            return thought, action, action_input

        except Exception as e:
            if isinstance(e, ActionParsingException):
                raise
            raise ActionParsingException(
                f"Error parsing action: {str(e)}. "
                f"Please ensure the output follows the format 'Thought: <text> "
                f"Action: <action> Action Input: <valid JSON>'.",
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
        ).model_dump(by_alias=True)

    def _extract_final_answer(self, output: str) -> str:
        """Extracts the final thought and answer as a tuple from the output string."""
        match = re.search(r"Thought:\s*(.*?)\s*Answer:\s*(.*)", output, re.DOTALL)
        if match:
            thought = match.group(1).strip()
            answer = match.group(2).strip()
            return thought, answer
        else:
            return "", ""

    def stream_reasoning(self, content: dict[str, Any], config: RunnableConfig, **kwargs) -> None:
        """
        Streams intermediate reasoning of the Agent.

        Args:
            content (dict[str, Any]): Content that will be sent.
            config (RunnableConfig | None): Configuration for the agent run.
            **kwargs: Additional parameters for running the agent.
        """
        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
            self.stream_content(
                content=content,
                source=self.name,
                step="reasoning",
                config=config,
                by_tokens=False,
                **kwargs,
            )

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
        if self.inference_mode in [InferenceMode.XML, InferenceMode.DEFAULT]:
            stop_sequences.extend(["Observation: ", "\nObservation:"])
        self.llm.stop = stop_sequences

        for loop_num in range(1, self.max_loops + 1):
            try:
                llm_result = self._run_llm(
                    messages=self._prompt.messages,
                    tools=self._tools,
                    response_format=self._response_format,
                    config=config,
                    **kwargs,
                )
                action, action_input = None, None
                llm_generated_output = ""
                llm_reasoning = (
                    llm_result.output.get("content")[:200]
                    if llm_result.output.get("content")
                    else str(llm_result.output.get("tool_calls", ""))[:200]
                )
                logger.info(f"Agent {self.name} - {self.id}: Loop {loop_num}, " f"reasoning:\n{llm_reasoning}...")

                match self.inference_mode:
                    case InferenceMode.DEFAULT:
                        llm_generated_output = llm_result.output.get("content", "")

                        self.tracing_intermediate(loop_num, self._prompt.messages, llm_generated_output)

                        if "Answer:" in llm_generated_output:
                            thought, final_answer = self._extract_final_answer(llm_generated_output)
                            self.log_final_output(thought, final_answer, loop_num)
                            self.tracing_final(loop_num, final_answer, config, kwargs)

                            if self.streaming.enabled:
                                if self.streaming.mode == StreamingMode.ALL:

                                    self.stream_content(
                                        content={"thought": thought, "loop_num": loop_num},
                                        source=self.name,
                                        step="reasoning",
                                        config=config,
                                        **kwargs,
                                    )
                                self.stream_content(
                                    content=final_answer,
                                    source=self.name,
                                    step="answer",
                                    config=config,
                                    **kwargs,
                                )

                            return final_answer

                        thought, action, action_input = self._parse_action(llm_generated_output)
                        self.log_reasoning(thought, action, action_input, loop_num)

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
                        thought = llm_generated_output_json["thought"]
                        if action == "provide_final_answer":
                            final_answer = llm_generated_output_json["answer"]
                            self.log_final_output(thought, final_answer, loop_num)
                            self.tracing_final(loop_num, final_answer, config, kwargs)
                            if self.streaming.enabled:
                                if self.streaming.mode == StreamingMode.ALL:
                                    self.stream_content(
                                        content={"thought": thought, "loop_num": loop_num},
                                        source=self.name,
                                        step="reasoning",
                                        config=config,
                                        **kwargs,
                                    )
                                self.stream_content(
                                    content=final_answer,
                                    source=self.name,
                                    step="answer",
                                    config=config,
                                    **kwargs,
                                )
                            return final_answer

                        action_input = llm_generated_output_json["action_input"]

                        if isinstance(action_input, str):
                            try:
                                action_input = json.loads(action_input)
                            except json.JSONDecodeError as e:
                                raise ActionParsingException(
                                    f"Error parsing action_input string. {e}", recoverable=True
                                )

                        self.log_reasoning(thought, action, action_input, loop_num)

                    case InferenceMode.STRUCTURED_OUTPUT:
                        if self.verbose:
                            logger.info(f"Agent {self.name} - {self.id}: using structured output inference mode")

                        llm_generated_output = llm_result.output["content"]
                        self.tracing_intermediate(loop_num, self._prompt.messages, llm_generated_output)
                        try:
                            llm_generated_output_json = json.loads(llm_generated_output)
                        except json.JSONDecodeError as e:
                            raise ActionParsingException(f"Error parsing action. {e}", recoverable=True)

                        thought = llm_generated_output_json["thought"]
                        action = llm_generated_output_json["action"]
                        action_input = llm_generated_output_json["action_input"]

                        if action == "finish":
                            self.log_final_output(thought, action_input, loop_num)
                            self.tracing_final(loop_num, action_input, config, kwargs)
                            if self.streaming.enabled:
                                if self.streaming.mode == StreamingMode.ALL:
                                    self.stream_content(
                                        content={"thought": thought, "loop_num": loop_num},
                                        source=self.name,
                                        step="reasoning",
                                        config=config,
                                        **kwargs,
                                    )

                                self.stream_content(
                                    content=action_input,
                                    source=self.name,
                                    step="answer",
                                    config=config,
                                    **kwargs,
                                )
                            return action_input

                        try:
                            action_input = json.loads(action_input)
                        except json.JSONDecodeError as e:
                            raise ActionParsingException(f"Error parsing action_input string. {e}", recoverable=True)

                        self.log_reasoning(thought, action, action_input, loop_num)

                    case InferenceMode.XML:
                        if self.verbose:
                            logger.info(f"Agent {self.name} - {self.id}: using XML inference mode")

                        llm_generated_output = llm_result.output["content"]
                        self.tracing_intermediate(loop_num, self._prompt.messages, llm_generated_output)

                        try:
                            parsed_data = XMLParser.parse(
                                llm_generated_output, required_tags=["thought", "answer"], optional_tags=["output"]
                            )
                            thought = parsed_data.get("thought")
                            final_answer = parsed_data.get("answer")
                            self.log_final_output(thought, final_answer, loop_num)
                            self.tracing_final(loop_num, final_answer, config, kwargs)
                            if self.streaming.enabled:
                                if self.streaming.mode == StreamingMode.ALL:
                                    self.stream_content(
                                        content={"thought": thought, "loop_num": loop_num},
                                        source=self.name,
                                        step="reasoning",
                                        config=config,
                                        **kwargs,
                                    )
                                self.stream_content(
                                    content=final_answer,
                                    source=self.name,
                                    step="answer",
                                    config=config,
                                    **kwargs,
                                )
                            return final_answer

                        except TagNotFoundError:
                            logger.debug("XMLParser: Not a final answer structure, trying action structure.")
                            try:
                                parsed_data = XMLParser.parse(
                                    llm_generated_output,
                                    required_tags=["thought", "action", "action_input"],
                                    optional_tags=["output"],
                                    json_fields=["action_input"],
                                )
                                thought = parsed_data.get("thought")
                                action = parsed_data.get("action")
                                action_input = parsed_data.get("action_input")
                                self.log_reasoning(thought, action, action_input, loop_num)
                            except (XMLParsingError, TagNotFoundError, JSONParsingError) as e:
                                logger.error(f"XMLParser: Failed to parse XML for action or answer: {e}")
                                raise ActionParsingException(f"Error parsing LLM output: {e}", recoverable=True)

                        except (XMLParsingError, JSONParsingError) as e:
                            logger.error(f"XMLParser: Error parsing potential final answer XML: {e}")
                            raise ActionParsingException(f"Error parsing LLM output: {e}", recoverable=True)

                self._prompt.messages.append(Message(role=MessageRole.ASSISTANT, content=llm_generated_output))

                if action and self.tools:
                    try:
                        tool = self.tool_by_names.get(self.sanitize_tool_name(action))
                        if not tool:
                            self.stream_reasoning(
                                {
                                    "thought": thought,
                                    "action": action,
                                    "action_input": action_input,
                                    "loop_num": loop_num,
                                },
                                config,
                                **kwargs,
                            )

                            raise AgentUnknownToolException(
                                f"Unknown tool: {action}."
                                "Use only available tools and provide only the tool's name in the action field. "
                                "Do not include any additional reasoning. "
                                "Please correct the action field or state that you cannot answer the question."
                            )

                        self.stream_reasoning(
                            {
                                "thought": thought,
                                "action": action,
                                "tool": tool,
                                "action_input": action_input,
                                "loop_num": loop_num,
                            },
                            config,
                            **kwargs,
                        )

                        tool_result = self._run_tool(tool, action_input, config, **kwargs)

                    except RecoverableAgentException as e:
                        tool_result = f"{type(e).__name__}: {e}"

                    observation = f"\nObservation: {tool_result}\n"
                    if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                        self.stream_content(
                            content={"name": tool.name, "input": action_input, "result": tool_result},
                            source=tool.name if tool else action,
                            step="tool",
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
                    self._prompt.messages.append(Message(role=MessageRole.USER, content=observation, static=True))
                else:
                    self.stream_reasoning(
                        {
                            "thought": thought,
                            "action": action,
                            "action_input": action_input,
                            "loop_num": loop_num,
                        },
                        config,
                        **kwargs,
                    )

            except ActionParsingException as e:
                self._prompt.messages.append(
                    Message(role=MessageRole.ASSISTANT, content="Response is:" + llm_generated_output)
                )
                self._prompt.messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=f"Correction Instruction: The previous response could not be parsed due to "
                        f"the following error: '{type(e).__name__}: {e}'. "
                        f"Please regenerate the response strictly following the "
                        f"required XML format, ensuring all tags are present and "
                        f"correctly structured, and that any JSON content (like action_input) is valid.",
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
            max_loop_final_answer = self._handle_max_loops_exceeded(input_message, config, **kwargs)
            if self.streaming.enabled:
                self.stream_content(
                    content=max_loop_final_answer,
                    source=self.name,
                    step="answer",
                    config=config,
                    **kwargs,
                )
            return max_loop_final_answer

    def aggregate_history(self, messages: list[Message, VisionMessage]) -> str:
        """
        Concatenates multiple history messages into one unified string.

        Args:
            messages (list[Message, VisionMessage]): List of messages to aggregate.

        Returns:
            str: Aggregated content.
        """

        history = ""

        for message in messages:
            if isinstance(message, VisionMessage):
                for content in message.content:
                    if isinstance(content, VisionMessageTextContent):
                        history += content.text
            else:
                if message.role == MessageRole.ASSISTANT:
                    history += f"-TOOL DESCRIPTION START-\n{message.content}\n-TOOL DESCRIPTION END-\n"
                elif message.role == MessageRole.USER:
                    history += f"-TOOL OUTPUT START-\n{message.content}\n-TOOL OUTPUT END-\n"

        return history

    def _handle_max_loops_exceeded(
        self, input_message: Message | VisionMessage, config: RunnableConfig | None = None, **kwargs
    ) -> str:
        """
        Handle the case where max loops are exceeded by crafting a thoughtful response.
        Uses XMLParser to extract the final answer from the LLM's last attempt.

        Args:
            input_message (Message | VisionMessage): Initial user message.
            config (RunnableConfig | None): Configuration for the agent run.
            **kwargs: Additional parameters for running the agent.

        Returns:
            str: Final answer provided by the agent.
        """
        system_message = Message(content=REACT_MAX_LOOPS_PROMPT, role=MessageRole.SYSTEM)
        conversation_history = Message(
            content=self.aggregate_history(self._prompt.messages), role=MessageRole.USER, static=True
        )
        llm_final_attempt_result = self._run_llm(
            [system_message, input_message, conversation_history], config=config, **kwargs
        )
        llm_final_attempt = llm_final_attempt_result.output["content"]
        self._run_depends = [NodeDependency(node=self.llm).to_dict()]

        try:
            final_answer = XMLParser.extract_first_tag_lxml(llm_final_attempt, ["answer"])
            if final_answer is None:
                logger.warning("Max loops handler: lxml failed to extract <answer>, falling back to regex.")
                final_answer = XMLParser.extract_first_tag_regex(llm_final_attempt, ["answer"])

            if final_answer is None:
                logger.error(
                    "Max loops handler: Failed to extract <answer> tag even with fallbacks. Returning raw output."
                )
                final_answer = llm_final_attempt

        except Exception as e:
            logger.error(f"Max loops handler: Error during final answer extraction: {e}. Returning raw output.")
            final_answer = llm_final_attempt

        return f"{final_answer}"

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
            if params:
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

        self._response_format = schema

    @staticmethod
    def filter_format_type(param_annotation: Any) -> list[str]:
        """
        Filters proper type for a function calling schema.

        Args:
            param_annotation (Any): Parameter annotation.
        Returns:
            list[str]: List of parameter types that describe provided annotation.
        """

        if get_origin(param_annotation) in (Union, types.UnionType):
            return get_args(param_annotation)

        return [param_annotation]

    def generate_property_schema(self, properties, name, field):
        if not field.json_schema_extra or field.json_schema_extra.get("is_accessible_to_agent", True):
            description = field.description or "No description."

            description += f" Defaults to: {field.default}." if field.default and not field.is_required() else ""
            params = self.filter_format_type(field.annotation)

            properties[name] = {"type": [], "description": description}

            for param in params:
                if param is type(None):
                    properties[name]["type"].append("null")

                elif param_type := TYPE_MAPPING.get(param):
                    properties[name]["type"].append(param_type)

                elif issubclass(param, Enum):
                    element_type = TYPE_MAPPING.get(
                        self.filter_format_type(type(list(param.__members__.values())[0].value))[0]
                    )
                    properties[name]["type"].append(element_type)
                    properties[name]["enum"] = [field.value for field in param.__members__.values()]

                elif getattr(param, "__origin__", None) is list:
                    properties[name]["type"].append("array")
                    properties[name]["items"] = {"type": TYPE_MAPPING.get(param.__args__[0])}

    def generate_function_calling_schemas(self):
        """Generate schemas for function calling."""
        self._tools.append(final_answer_function_schema)
        for tool in self.tools:
            properties = {}
            input_params = tool.input_schema.model_fields.items()
            if list(input_params) and not isinstance(self.llm, Gemini):
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
                                    "description": "Your reasoning about using this tool.",
                                },
                                "action_input": {
                                    "type": "object",
                                    "description": "Input for the selected tool",
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

                self._tools.append(schema)

            else:
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
                                    "description": "Your reasoning about using this tool.",
                                },
                                "action_input": {
                                    "type": "string",
                                    "description": "Input for the selected tool in JSON string format.",
                                },
                            },
                            "additionalProperties": False,
                            "required": ["thought", "action_input"],
                        },
                        "strict": True,
                    },
                }

                self._tools.append(schema)

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
