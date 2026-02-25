# Common blocks (function calling: tool schema is source of truth; no input_formats in prompt)
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

REACT_MAX_LOOPS_PROMPT = """
You are tasked with providing a final answer for the initial user question based on information gathered during a process that has reached its maximum number of loops.
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
Provide your final answer or explanation below. Your response should be clear, concise, and professional.
"""  # noqa: E501


HISTORY_SUMMARIZATION_PROMPT_REPLACE = """Provide a concise summary of the conversation history above.
 Focus on key decisions, important information, and tool outputs."""


PROMPT_AUTO_CLEAN_CONTEXT = "Automatically cleaning the context with Context Manager Tool..."
