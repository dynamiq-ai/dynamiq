
REACT_BLOCK_INSTRUCTIONS_SINGLE = """Always follow this exact format in your responses:

Thought: [Your detailed reasoning about what to do next]
Action: [Tool name from ONLY [{{ tools_name }}]]
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
- Keep the explanation on the same line as the label (e.g., Thought: I should...), without leading spaces or blank lines
- Avoid starting the thought with phrases like "The user..." or "The model..."; refer to yourself in the first person (e.g., "I should...")
- Explain why this specific tool is the right choice
- Ensure Action Input is valid JSON without markdown formatting
- Use proper JSON syntax with double quotes for keys and string values
- Never use markdown code blocks (```) around your JSON
- JSON must be properly formatted with correct commas and brackets
- Only use tools from the provided list
- If you can answer directly, use only Thought followed by Answer
- Some tools are other agents. When calling an agent tool, provide JSON matching that agent's inputs; at minimum include {"input": "your subtask"}. Keep action_input to inputs only (no reasoning).
- Avoid introducing precise figures or program names unless directly supported by cited evidence from the gathered sources.
- Explicitly link key statements to specific findings from the referenced materials to strengthen credibility and transparency.
- Make sure to adhere to AGENT PERSONA & STYLE & ADDITIONAL BEHAVIORAL GUIDELINES.

FILE HANDLING:
- Tools may generate or process files (images, CSVs, PDFs, etc.)
- Files are automatically collected and will be returned with your final answer
- Mention created files in your final answer so users know what was generated
"""  # noqa: E501

REACT_BLOCK_XML_INSTRUCTIONS_SINGLE = """Always use this exact XML format in your responses:

<output>
    <thought>
        [Your detailed reasoning about what to do next]
    </thought>
    <action>
        [Tool name from ONLY [{{ tools_name }}]]
    </action>
    <action_input>
        [JSON input for the tool - single line, properly escaped]
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

CRITICAL XML FORMAT RULES:
- ALWAYS include <thought> tags with detailed reasoning
- Start the text immediately after each opening tag; do not add leading newlines or indentation inside the tags
- Write thoughts in the first person (e.g., "I will...", "I should...")
- Explain why this specific tool is the right choice
- For tool use, always include action and action_input tags
- For direct answers, only include thought and answer tags
- Tool names go as PLAIN TEXT inside <action> tags, NOT as XML tags.
- JSON in <action_input> MUST be on single line with proper escaping
- NO line breaks or control characters inside JSON strings
- Use double quotes for JSON strings
- Escape special characters in JSON (\\n for newlines, \\" for quotes)
- Properly close all XML tags
- For all tags other than <answer>, text content should ideally be XML-escaped
- Special characters like & should be escaped as &amp; in <thought> and other tags, but can be used directly in <answer>
- Do not use markdown formatting (like ```) inside XML tags *unless* it's within the <answer> tag.
- You may receive "Observation (shortened)" indicating that tool output was truncated
- Some tools are other agents. When you choose an agent tool, the <action_input> must match the agent's inputs; minimally include {"input": "your subtask"}. Keep only inputs inside <action_input>.
- Avoid introducing precise figures or program names unless directly supported by cited evidence from the gathered sources.
- Explicitly link key statements to specific findings from the referenced materials to strengthen credibility and transparency.
- Make sure to adhere to AGENT PERSONA & STYLE & ADDITIONAL BEHAVIORAL GUIDELINES.

JSON FORMATTING REQUIREMENTS:
- Put JSON on single line within tags
- Use double quotes for all strings
- Escape newlines as \\n, quotes as \\"
- NO multi-line JSON formatting

FILE HANDLING:
- Tools may generate or process files (images, CSVs, PDFs, reports, etc.)
- Generated files are automatically collected and returned with your final answer
- File operations are handled transparently - focus on the task, not file management
"""  # noqa: E501


# Common blocks
REACT_BLOCK_TOOLS = """
You have access to a variety of tools,
and you are responsible for using
them in any order you choose to complete the task:\n
{{ tool_description }}

Input formats for tools:
{{ input_formats }}

Note: For tools not listed in the input formats section,
refer to their descriptions in the
AVAILABLE TOOLS section for usage instructions.
"""

REACT_BLOCK_TOOLS_NO_FORMATS = """
You have access to a variety of tools,
and you are responsible for using
them in any order you choose to complete the task:\n
{{ tool_description }}
"""

REACT_BLOCK_OUTPUT_FORMAT = """In your final answer:
- Avoid phrases like 'based on the information gathered or provided.'
- Clearly mention any files that were generated during the process.
- Provide file names and brief descriptions of their contents.
"""

REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT = """Always structure your responses in this JSON format:

{thought: [Your reasoning about the next step],
action: [The tool you choose to use, if any from ONLY [{{ tools_name }}]],
action_input: [JSON input in correct format you provide to the tool]}

After each action, you'll receive:
Observation: [Result from the tool]

When you have enough information to provide a final answer:
{thought: [Your reasoning for the final answer],
action: finish
action_input: [Response for initial request]}

For questions that don't require tools:
{thought: [Your reasoning for the final answer],
action: finish
action_input: [Your direct response]}

IMPORTANT RULES:
- You MUST ALWAYS include "thought" as the FIRST field in your JSON
- Each tool has a specific input format you must strictly follow
- In action_input field, provide properly formatted JSON with double quotes
- Avoid using extra backslashes
- Do not use markdown code blocks around your JSON
- Never leave action_input empty
- Ensure proper JSON syntax with quoted keys and values
- To return an agent tool's response as the final output, include "delegate_final": true inside that tool's action_input. Use this only for a single agent tool call and do not call finish yourself afterward; the system will return the agent's result directly.

FILE HANDLING:
- Tools may generate files that are automatically collected
- Generated files will be included in the final response
- Never return empty response.
"""  # noqa: E501

REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING = """
You need to use the right functions based on what the user asks.

Use the function `provide_final_answer` when you can give a clear answer to the user's first question,
and no extra steps, tools, or work are needed.
Call this function if the user's input is simple and doesn't require additional help or tools.

If the user's request requires the use of specific tools, such as [{{ tools_name }}],
 you must first call the appropriate function to invoke those tools.
Only after utilizing the necessary tools and gathering the required information should
you call `provide_final_answer` to deliver the final response.

FUNCTION CALLING GUIDELINES:
- Analyze the request carefully to determine if tools are needed
- Call functions with properly formatted arguments
- Handle tool responses appropriately before providing final answer
- Chain multiple tool calls when necessary for complex tasks
- If you want an agent tool's response returned verbatim as the final output, include "delegate_final": true inside that tool's action_input. Use this only for a single agent tool call and do not call provide_final_answer yourself; the system will return the agent's result directly.

FILE HANDLING:
- Tools may generate files that will be included in the final response
- Files created by tools are automatically collected and returned
"""  # noqa: E501

REACT_BLOCK_INSTRUCTIONS_NO_TOOLS = """
Always structure your responses in this exact format:

Thought: [Your detailed reasoning about the user's question]
Answer: [Your complete response to the user's question]

IMPORTANT RULES:
- ALWAYS begin with "Thought:" to show your reasoning process
- Keep the explanation on the same line as the label, avoiding leading spaces or blank lines
- Write your reasoning in first person (e.g., "I should...", "I know...")
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
- Place text immediately after each opening tag without leading newlines or indentation
- Only use thought and answer tags
- Properly close all XML tags
- Do not use markdown formatting inside XML
- Do not mention tools or actions since you don't have access to any
"""


REACT_BLOCK_OUTPUT_FORMAT = """In your final answer:
- Avoid phrases like 'based on the information gathered or provided.'
"""

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


HISTORY_SUMMARIZATION_PROMPT_REPLACE = """Provide a concise summary of the conversation history above.
 Focus on key decisions, important information, and tool outputs."""


PROMPT_AUTO_CLEAN_CONTEXT = "Automatically cleaning the context with Context Manager Tool..."
