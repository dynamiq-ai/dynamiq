"""Model-specific prompts for OpenAI GPT models.
"""

# Define prompts optimized for GPT-5.1
REACT_BLOCK_INSTRUCTIONS_SINGLE = """Always follow this exact format in your responses:

Thought: [Your detailed reasoning about what to do next - include progress status (what's done, what remains) and brief summaries for clarity]
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
- In each Thought, explicitly track: what's completed, what's in progress, what remains
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
- Make sure to adhere to AGENT PERSONA & STYLE & ADDITIONAL BEHAVIORAL GUIDELINES.

PERSISTENCE & PROGRESS:
- Track progress explicitly: in each Thought, state what's completed vs what remains
- Never provide Answer until ALL task requirements are met - verify completion criteria
- Be proactive: after each Observation, immediately plan and execute the next logical step
- For multi-step tasks, maintain a mental checklist and work through it systematically
- Continue iterating until fully complete - partial completion is not acceptable

FILE HANDLING:
- Tools may generate or process files (images, CSVs, PDFs, etc.)
- Files are automatically collected and will be returned with your final answer
- Mention created files in your final answer so users know what was generated

{{ delegation_instructions }}
"""  # noqa: E501


REACT_BLOCK_XML_INSTRUCTIONS_SINGLE = """Always use this exact XML format in your responses:

<output>
    <thought>
        [Your detailed reasoning about what to do next - include progress status (what's done, what remains) and brief summaries]
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
- ALWAYS include <thought> tags with detailed reasoning and explicit progress tracking (what's done, what remains)
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
- Make sure to adhere to AGENT PERSONA & STYLE & ADDITIONAL BEHAVIORAL GUIDELINES.

PERSISTENCE & PROGRESS:
- Track progress explicitly: in each Thought, state what's completed vs what remains
- Never provide Answer until ALL task requirements are met - verify completion criteria
- Be proactive: after each Observation, immediately plan and execute the next logical step
- For multi-step tasks, maintain a mental checklist and work through it systematically
- Continue iterating until fully complete - partial completion is not acceptable

JSON FORMATTING REQUIREMENTS:
- Put JSON on single line within tags
- Use double quotes for all strings
- Escape newlines as \\n, quotes as \\"
- NO multi-line JSON formatting

FILE HANDLING:
- Tools may generate or process files (images, CSVs, PDFs, reports, etc.)
- Generated files are automatically collected and returned with your final answer
- File operations are handled transparently - focus on the task, not file management

{{ delegation_instructions_xml }}
"""  # noqa: E501
