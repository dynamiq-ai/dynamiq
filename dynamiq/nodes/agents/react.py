import json
import re
import types
from typing import Any, Union, get_args, get_origin

from litellm import get_supported_openai_params, supports_function_calling
from pydantic import Field, model_validator

from dynamiq.nodes.agents.base import Agent, AgentIntermediateStep, AgentIntermediateStepModelObservation
from dynamiq.nodes.agents.exceptions import ActionParsingException, MaxLoopsExceededException, RecoverableAgentException
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingMode
from dynamiq.utils.logger import logger

REACT_BLOCK_TOOLS = (
    "You have access to a variety of tools,"
    "and you are responsible for using them in any order you choose to complete the task:\n"
    "{tools_desc}"
)

REACT_BLOCK_NO_TOOLS = "You do not have access to any tools."

REACT_BLOCK_XML_INSTRUCTIONS = """
Here is how you will think about the user's request
<output>
    <thought>
        Here you reason about the next step
    </thought>
    <action>
        Here you choose the tool to use from [{tools_name}]
    </action>
    <action_input>
        Here you provide the input to the tool, correct JSON format
    </action_input>
</output>

REMEMBER:
* Inside 'action' provide just name of one tool from this list: [{tools_name}]. Don't wrap it with <>.
* Each 'action' has its own input format strictly adhere to it.
Input formats for tools:
{input_formats}

After each action, the user will provide an "Observation" with the result.
Continue this Thought/Action/Action Input/Observation sequence until you have enough information to answer the request.
When you have sufficient information, provide your final answer in one of these two formats:
If you can answer the request:
<output>
    <thought>
        I can answer without using any tools
    </thought>
    <answer>
        Your answer here
    </answer>
</output>

If you cannot answer the request:
<output>
    <thought>
        I cannot answer with the tools I have
    </thought>
    <answer>
        Explanation of why you cannot answer
    </answer>
</output>
"""  # noqa: E501


REACT_BLOCK_INSTRUCTIONS = """Always structure your responses in the following format:
Thought: [Your reasoning for the next step]
Action: [The tool you choose to use, if any, from ONLY [{tools_name}]]
Action Input: [The input you provide to the tool]
Remember:
- Each tool has its specific input format you have strickly adhere to it.
- Avoid using triple quotes (multi-line strings, docstrings) when providing multi-line code.
- Provide all necessary information in 'Action Input' for the next step to succeed.
- Action Input must be in JSON format.
- Always begin each response with a 'Thought' explaining your reasoning.
- If you use a tool, follow the 'Thought' with an 'Action' (chosen from the available tools) and an 'Action Input'.
- After each action, the user will provide an 'Observation' with the result.
- Continue this Thought/Action/Action Input/Observation sequence until you have enough information to answer the request.

Input formats for tools:
{input_formats}

When you have sufficient information, provide your final answer in one of these two formats:
If you can answer the request:
Thought: I can answer without using any tools
Answer: [Your answer here]

If you cannot answer the request:
Thought: I cannot answer with the tools I have
Answer: [Explanation of why you cannot answer]

Remember:
- Always start with a Thought.
- Never use markdown code markers in your response.
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

Remember:
- Each tool has is specific input format you have strickly adhere to it.
- In action_input you have to provide input in JSON format.
- Avoid using extra backslashes

Input formats for tools:
{input_formats}
"""  # noqa: E501


REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING = """
You have to call appropriate functions.

Function descriptions:
plan_next_action - function that should be called to use tools [{tools_name}]].
provide_final_answer - function that should be called when answer on initial request can be provided
"""  # noqa: E501


REACT_BLOCK_INSTRUCTIONS_NO_TOOLS = """
Always structure your responses in the following format:
Thought: [Your reasoning for why you cannot fully answer the initial question]
Observation: [Answer to the initial question or part of it]
- Only include information relevant to the main request.
- Always start each response with a 'Thought' explaining your reasoning.
- After each action, the user will provide an 'Observation' with the result.
- Continue this Thought/Action/Action Input/Observation sequence until you have enough information to fully answer the request.

When you have sufficient information, provide your final answer in one of these formats:
If you can answer the request:
Thought: I can answer without using any tools
Answer: [Your answer here]
If you cannot answer the request:
Thought: I cannot answer with the tools I have
Answer: [Explanation of why you cannot answer]

Remember:
- Always begin with a Thought.
- Do not use markdown code markers in your response."
"""  # noqa: E501


REACT_BLOCK_OUTPUT_FORMAT = (
    "In your final answer, avoid phrases like 'based on the information gathered or provided.'"
    "Simply give a clear and concise answer."
)

REACT_BLOCK_REQUEST = "User request: {input}"
REACT_BLOCK_CONTEXT = "Below is the conversation: {context}"


REACT_MAX_LOOPS_PROMPT = """
You are tasked with providing a final answer based on information gathered during a process that has reached its maximum number of loops.
Your goal is to analyze the given context and formulate a clear, concise response.
First, carefully review the following context, which contains thoughts and information gathered during the process:
<context>
{context}
</context>
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
        "description": "Function should be called when if you can answer the initial request",
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
        action = self.parse_xml_content(output_content, "action")
        action_input_text = self.parse_xml_content(output_content, "action_input")

        try:
            action_input = json.loads(action_input_text)
        except json.JSONDecodeError as e:
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

        return action, action_input

    def extract_output_and_answer_xml(self, text: str) -> dict[str, str]:
        """Extract output and answer from XML-like structure."""
        output = self.parse_xml_content(text, "output")
        answer = self.parse_xml_content(text, "answer")
        return {"output": output, "answer": answer}

    def tracing_final(self, loop_num, final_answer, config, kwargs):
        self._intermediate_steps[loop_num]["final_answer"] = final_answer

    def tracing_intermediate(self, loop_num, formatted_prompt, llm_generated_output):
        self._intermediate_steps[loop_num] = AgentIntermediateStep(
            input_data={"prompt": formatted_prompt},
            model_observation=AgentIntermediateStepModelObservation(
                initial=llm_generated_output,
            ),
        ).model_dump()

    def _run_agent(self, config: RunnableConfig | None = None, **kwargs) -> str:
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
        previous_responses = []
        for loop_num in range(self.max_loops):
            formatted_prompt = self.generate_prompt(
                user_request=kwargs.get("input", ""),
                tools_desc=self.tool_description,
                tools_name=self.tool_names,
                context="\n".join(previous_responses),
                input_formats=self.generate_input_formats(self.tools),
            )
            try:
                llm_result = self.llm.run(
                    input_data={},
                    config=config,
                    prompt=Prompt(messages=[Message(role="user", content=formatted_prompt)]),
                    run_depends=self._run_depends,
                    schema=self.format_schema,
                    inference_mode=self.inference_mode,
                    **kwargs,
                )
                self._run_depends = [NodeDependency(node=self.llm).to_dict()]

                if llm_result.status != RunnableStatus.SUCCESS:
                    previous_responses.append(llm_result.output["content"])
                    continue

                action, action_input = None, None
                llm_generated_output = ""
                logger.info(
                    f"Agent {self.name} - {self.id}: Loop {loop_num + 1}, reasoning:\n{llm_result.output['content']}"
                )
                match self.inference_mode:
                    case InferenceMode.DEFAULT:
                        llm_generated_output = llm_result.output["content"]
                        self.tracing_intermediate(loop_num, formatted_prompt, llm_generated_output)
                        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                            self.stream_content(
                                content=llm_generated_output,
                                source=self.name,
                                step=f"reasoning_{loop_num + 1}",
                                config=config,
                                **kwargs,
                            )
                        if "Answer:" in llm_generated_output:
                            final_answer = self._extract_final_answer(llm_generated_output)
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
                        action, action_input = self._parse_action(llm_generated_output)

                    case InferenceMode.FUNCTION_CALLING:
                        action = llm_result.output["tool_calls"][0]["function"]["name"].strip()
                        llm_generated_output_json = json.loads(
                            llm_result.output["tool_calls"][0]["function"]["arguments"]
                        )
                        llm_generated_output = json.dumps(llm_generated_output_json)
                        self.tracing_intermediate(loop_num, formatted_prompt, llm_generated_output)
                        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                            self.stream_content(
                                content=llm_generated_output,
                                source=self.name,
                                step=f"reasoning_{loop_num + 1}",
                                config=config,
                                **kwargs,
                            )
                        if action == "provide_final_answer":
                            final_answer = llm_generated_output_json["answer"]
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
                        action_input = llm_generated_output_json["action_input"]

                    case InferenceMode.STRUCTURED_OUTPUT:
                        if self.verbose:
                            logger.info(f"Agent {self.name} - {self.id}: using structured output inference mode")
                        llm_generated_output_json = json.loads(llm_result.output["content"])
                        action = llm_generated_output_json["action"]
                        self.tracing_intermediate(loop_num, formatted_prompt, llm_generated_output)
                        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                            self.stream_content(
                                content=llm_generated_output,
                                source=self.name,
                                step=f"reasoning_{loop_num + 1}",
                                config=config,
                                **kwargs,
                            )
                        if action == "finish":
                            final_answer = llm_generated_output_json["action_input"]
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
                        action_input = json.loads(llm_generated_output_json["action_input"])
                        llm_generated_output = json.dumps(llm_generated_output_json)

                    case InferenceMode.XML:
                        if self.verbose:
                            logger.info(f"Agent {self.name} - {self.id}: using XML inference mode")
                        llm_generated_output = llm_result.output["content"]
                        self.tracing_intermediate(loop_num, formatted_prompt, llm_generated_output)
                        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                            self.stream_content(
                                content=llm_generated_output,
                                source=self.name,
                                step=f"reasoning_{loop_num + 1}",
                                config=config,
                                **kwargs,
                            )
                        if "<answer>" in llm_generated_output:
                            final_answer = self._extract_final_answer_xml(llm_generated_output)
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
                        action, action_input = self.parse_xml_and_extract_info(llm_generated_output)
                if action:
                    if self.tools:
                        try:
                            tool = self._get_tool(action)
                            tool_result = self._run_tool(tool, action_input, config, **kwargs)

                        except RecoverableAgentException as e:
                            tool_result = f"{type(e).__name__}: {e}"

                        observation = f"\nObservation: {tool_result}\n"
                        llm_generated_output += observation
                        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                            self.stream_content(
                                content=observation,
                                source=tool.name,
                                step=f"tool_{loop_num}",
                                config=config,
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

                previous_responses.append(llm_generated_output)

            except ActionParsingException as e:
                previous_responses.append(f"{type(e).__name__}: {e}")
                continue
        if self.behaviour_on_max_loops == Behavior.RAISE:
            error_message = (
                f"Agent {self.name} (ID: {self.id}) has reached the maximum loop limit of {self.max_loops} without finding a final answer. "  # noqa: E501
                f"Last response: {previous_responses[-1]}\n"
                f"Consider increasing the maximum number of loops or reviewing the task complexity to ensure completion."  # noqa: E501
            )
            raise MaxLoopsExceededException(message=error_message)
        else:
            max_loop_final_answer = self._handle_max_loops_exceeded(previous_responses, config, **kwargs)
            if self.streaming.enabled:
                self.stream_content(
                    content=max_loop_final_answer,
                    source=self.name,
                    step="answer",
                    config=config,
                    **kwargs,
                )
            return max_loop_final_answer

    def _handle_max_loops_exceeded(
        self, previous_responses: list, config: RunnableConfig | None = None, **kwargs
    ) -> str:
        """
        Handle the case where max loops are exceeded by crafting a thoughtful response.
        """
        final_attempt_prompt = REACT_MAX_LOOPS_PROMPT.format(context="\n".join(previous_responses))
        llm_final_attempt = self._run_llm(final_attempt_prompt, config=config, **kwargs)
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
        type_mapping = {
            int: "integer",
            float: "float",
            bool: "boolean",
            str: "string",
        }

        if isinstance(param_type, str):
            match param_type:
                case "bool":
                    return "boolean"
                case "int":
                    return "integer"
                case "float":
                    return "float"
                case _:
                    return "string"
        elif get_origin(param_type) is Union:
            first_type = next((arg for arg in get_args(param_type) if arg is not type(None)), None)
            if first_type is None:
                return "string"
            return type_mapping.get(first_type, getattr(first_type, "__name__", "string"))
        else:
            return type_mapping.get(param_type, getattr(param_type, "__name__", "string"))

    def generate_function_calling_schemas(self):
        """Generate schemas for function calling."""
        self.format_schema.append(final_answer_function_schema)
        for tool in self.tools:
            properties = {}
            for name, field in tool.input_schema.model_fields.items():
                if not field.json_schema_extra or field.json_schema_extra.get("is_accessible_to_agent", True):
                    param_type = self.filter_format_type(field.annotation)
                    description = field.description or "No description"
                    properties[name] = {"type": param_type, "description": description}

            schema = {
                "type": "function",
                "strict": True,
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
                            },
                        },
                        "required": ["thought", "action_input"],
                    },
                },
            }

            self.format_schema.append(schema)

    def _init_prompt_blocks(self):
        """Initialize the prompt blocks required for the ReAct strategy."""
        super()._init_prompt_blocks()

        prompt_blocks = {
            "tools": REACT_BLOCK_TOOLS if self.tools else REACT_BLOCK_NO_TOOLS,
            "instructions": REACT_BLOCK_INSTRUCTIONS if self.tools else REACT_BLOCK_INSTRUCTIONS_NO_TOOLS,
            "output_format": REACT_BLOCK_OUTPUT_FORMAT,
            "context": REACT_BLOCK_CONTEXT,
            "request": REACT_BLOCK_REQUEST,
        }

        match self.inference_mode:
            case InferenceMode.FUNCTION_CALLING:
                self.generate_function_calling_schemas()
                prompt_blocks["instructions"] = REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING
            case InferenceMode.STRUCTURED_OUTPUT:
                self.generate_structured_output_schemas()
                prompt_blocks["instructions"] = REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT
            case InferenceMode.DEFAULT:
                if not self.tools:
                    prompt_blocks["tools"] = REACT_BLOCK_NO_TOOLS
                    prompt_blocks["instructions"] = REACT_BLOCK_INSTRUCTIONS_NO_TOOLS
            case InferenceMode.XML:
                prompt_blocks["instructions"] = REACT_BLOCK_XML_INSTRUCTIONS

        self._prompt_blocks.update(prompt_blocks)
