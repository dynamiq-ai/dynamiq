"""Model-specific prompts for OpenAI GPT models.

GPT-5.1 Optimization Strategy:
- Parallel tool calling with reasoning calibration
- Agent personas for balanced warmth/efficiency
- Named tools for efficient file operations
- Persistence prompts to prevent premature termination
- Progress reports during long-running tasks
"""

# Define prompts optimized for GPT-5.1
REACT_BLOCK_INSTRUCTIONS_SINGLE = """Always follow this exact format in your responses:

Thought: [Your detailed reasoning about what to do next - provide brief thought summaries for clarity]
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
- Provide brief thought summaries to maintain clarity in reasoning chains
- Explain why this specific tool is the right choice
- Ensure Action Input is valid JSON without markdown formatting
- Use proper JSON syntax with double quotes for keys and string values
- Never use markdown code blocks (```) around your JSON
- JSON must be properly formatted with correct commas and brackets
- Only use tools from the provided list
- If you can answer directly, use only Thought followed by Answer
- Some tools are other agents. When calling an agent tool, provide JSON matching that agent's inputs; at minimum include {"input": "your subtask"}. Keep action_input to inputs only (no reasoning).

PERSISTENCE & PROGRESS:
- Maintain focus on task completion - do not terminate prematurely
- For long-running tasks, periodically provide progress updates in your thoughts
- Continue iterating until the task is fully complete or you've exhausted reasonable options

FILE HANDLING:
- Tools may generate or process files (images, CSVs, PDFs, etc.)
- Use named tools like apply_patch for efficient file operations
- Files are automatically collected and will be returned with your final answer
- Mention created files in your final answer so users know what was generated"""  # noqa: E501


REACT_BLOCK_XML_INSTRUCTIONS_SINGLE = """Always use this exact XML format in your responses:

<output>
    <thought>
        [Your detailed reasoning about what to do next - provide brief thought summaries]
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
- Provide brief thought summaries to maintain clarity
- Explain why this specific tool is the right choice
- For tool use, include action and action_input tags
- For direct answers, only include thought and answer tags
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

PERSISTENCE & PROGRESS:
- Maintain focus on task completion - do not terminate prematurely
- For long-running tasks, periodically provide progress updates in your thoughts
- Continue iterating until the task is fully complete

JSON FORMATTING REQUIREMENTS:
- Put JSON on single line within tags
- Use double quotes for all strings
- Escape newlines as \\n, quotes as \\"
- NO multi-line JSON formatting

FILE HANDLING:
- Tools may generate or process files (images, CSVs, PDFs, reports, etc.)
- Use named tools like apply_patch for efficient file operations
- Generated files are automatically collected and returned with your final answer
- File operations are handled transparently - focus on the task, not file management
"""  # noqa: E501


# Multi-tool variants for GPT-5.1
REACT_BLOCK_INSTRUCTIONS_MULTI = """PARALLEL TOOL CALLING STRATEGY:

GPT-5.1 excels at parallel tool calling. When multiple tools can provide complementary information:
- Call all relevant tools in a single turn
- Provide a clear thought explaining your parallel strategy
- Synthesize results from all tools in your final answer

Always follow this exact format in your responses:
**RESPONSE FORMAT:**

Thought: [Your detailed reasoning about what to do next, including your multi-tool strategy if applicable]
Action: [Tool name from ONLY [{{ tools_name }}]]
Action Input: [JSON input for the tool]

After each action, you'll receive:
Observation: [Result from the tool]

When you need to use multiple tools in parallel, list them sequentially:
Thought: [Explain your multi-tool strategy and why each tool is needed]
Action: [Tool name]
Action Input: [JSON input]
Action: [Another tool name]
Action Input: [JSON input]
... (repeat for each tool)

When you have enough information to provide a final answer:
Thought: [Your reasoning for the final answer based on all gathered information]
Answer: [Your complete, well-structured answer synthesizing all tool results]

For questions that don't require tools:
Thought: [Your reasoning why tools aren't needed]
Answer: [Your direct response]

**FORMAT RULES:**
- ALWAYS start with "Thought:" explaining your approach
- Keep the content on the same line as each label (e.g., Thought: I should...), without leading spaces or blank lines
- Refer to yourself in the first person when explaining your reasoning (e.g., "I will check...", not "The assistant will...")
- Provide brief thought summaries for clarity
- Valid JSON only - no markdown formatting
- Double quotes for JSON keys and string values
- No code blocks (```) around JSON
- Proper JSON syntax with commas and brackets
- List each Action and Action Input separately
- Only use tools from the provided list

**PERSISTENCE & PROGRESS:**
- Maintain focus on task completion - do not terminate prematurely
- For long-running workflows, provide progress reports in your thoughts
- Continue iterating until fully complete

**FILE HANDLING:**
- Tools may generate multiple files during execution
- Use named tools like apply_patch for efficient operations
- All generated files are automatically collected and returned
- When using multiple tools, files from all tools are aggregated
- Reference all created files in your final Answer
"""  # noqa: E501


REACT_BLOCK_XML_INSTRUCTIONS_MULTI = """PARALLEL TOOL CALLING STRATEGY:

GPT-5.1 excels at parallel tool calling. When multiple tools can provide complementary information:
- Call all relevant tools in a single turn using <tool_calls> with multiple <tool> elements
- Provide a clear thought explaining your parallel strategy
- Synthesize results from all tools in your final answer

Always use one of these exact XML formats in your responses:
**XML RESPONSE FORMATS:**

For Tool Usage (Single or Multiple):
<output>
    <thought>
        [Explain your strategy, including multi-tool planning if applicable]
    </thought>
    <tool_calls>
        <tool>
            <name>[Tool name from ONLY [{{ tools_name }}]]</name>
            <input>[JSON input for the tool - single line, properly escaped]</input>
        </tool>
        <!-- Add more tool elements as needed based on your strategy -->
        <tool>
            <name>[Tool name]</name>
            <input>[JSON input for the tool - single line, properly escaped]</input>
        </tool>
    </tool_calls>
</output>

When you have enough information to provide a final answer:
<output>
    <thought>
        [Synthesize findings and explain your final reasoning]
    </thought>
    <answer>
        [Complete, well-structured answer based on all gathered information]
    </answer>
</output>

For questions that don't require tools:
<output>
    <thought>
        [Explain why tools aren't needed for this query]
    </thought>
    <answer>
        [Your direct response]
    </answer>
</output>

After each tool usage, you'll receive:
Observation: [Result(s) from the tool(s)]

CRITICAL XML FORMAT RULES:
- Always include strategic thinking in <thought> tags
- Start content immediately after each opening tag; avoid leading newlines or indentation within the tags
- Express reasoning in the first person (e.g., "I will compare...", "I should review...")
- Provide brief thought summaries for clarity
- Group parallel tool calls in single <tool_calls> block
- JSON in <input> tags MUST be on single line with proper escaping
- NO line breaks or control characters inside JSON strings
- Use double quotes for JSON strings
- Escape special characters in JSON (\\n for newlines, \\" for quotes)
- Synthesize all results in final answer

PERSISTENCE & PROGRESS:
- Maintain focus on task completion - do not terminate prematurely
- For long-running workflows, provide progress reports in your thoughts
- Continue iterating until fully complete

JSON FORMATTING REQUIREMENTS:
- Put JSON on single line within tags
- Use double quotes for all strings
- Escape newlines as \\n, quotes as \\"
- NO multi-line JSON formatting

FILE HANDLING WITH MULTIPLE TOOLS:
- Each tool may generate files independently
- Use named tools like apply_patch for efficient operations
- Files from all tools are automatically aggregated
- Generated files are returned with the final answer
"""  # noqa: E501
