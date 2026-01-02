DELEGATION_INSTRUCTIONS = (
    "- Optional: If you want an agent tool's response returned verbatim as the final output, "
    'set "delegate_final": true in that tool\'s input. Use this only for a single agent tool call '
    "and do not provide your own final answer; the system will return the agent's result directly."
    "Do not set delegate_final: true inside metadata of the input, it has to be a separate field."
)

DELEGATION_INSTRUCTIONS_XML = (
    '- To return an agent tool\'s response as the final output, include "delegate_final": true inside that '
    "tool's <input> or <action_input>. Use this only for a single agent tool call and do not provide an "
    "<answer> yourself; the system will return the agent's result directly."
)


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

{{ delegation_instructions }}
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

{{ delegation_instructions_xml }}
"""  # noqa: E501

REACT_BLOCK_MULTI_TOOL_PLANNING = """
MULTI-TOOL PLANNING AND STRATEGY:

Core Principle: Scale tool usage to match task complexity
Start with the minimum number of tools needed and scale up based on the task's requirements.

Decision Framework:
1. Zero Tools - Answer directly when:
   - Question is within your knowledge base
   - No real-time data needed
   - Simple explanations or general concepts

2. Single Tool - Use one tool when:
   - Single fact verification needed
   - One specific data point required
   - Simple lookup or calculation

3. Multiple Tools (2-4) - Use parallel tools when:
   - Comparing information from different sources
   - Gathering complementary data points
   - Cross-referencing or validation needed

4. Comprehensive Research (5+) - Use extensive tooling when:
   - Deep analysis requested ("comprehensive", "detailed", "thorough")
   - Multiple aspects of complex topic
   - Creating reports or extensive documentation

MANDATORY MULTI-TOOL PATTERNS:

1. Research Tasks (Adaptive Scaling):
   - START: Analyze query complexity
   - IF simple fact: 1-2 tool calls
   - IF moderate complexity: 3-4 tool calls with complementary queries
   - IF comprehensive research: 5+ tool calls covering:
     * Broad overview query
     * Specific benefits/advantages/features
     * Limitations/challenges/drawbacks
     * Comparisons/alternatives
     * Recent developments/updates

   Example progression:
   - Query 1: General topic overview
   - Query 2: Specific aspect (benefits, use cases)
   - Query 3: Challenges or limitations
   - Query 4+: Deep dives on critical aspects

2. Coding and Technical Tasks:
   - Documentation lookup: 1-2 tools (official docs + examples)
   - Debugging: 2-3 tools (error search + solution patterns)
   - Architecture decisions: 3-5 tools (best practices + comparisons + examples)
   - Full implementation: 5+ tools (docs + patterns + edge cases + optimization)

3. Data Collection/Analysis:
   - Single source: 1 tool
   - Multiple sources: Use parallel calls for efficiency
   - Comparative analysis: Minimum 3 sources
   - Market research: 5+ diverse sources

4. Verification and Fact-Checking:
   - Simple facts: 1-2 authoritative sources
   - Controversial topics: 3+ diverse sources
   - Critical information: Cross-reference with 3+ sources

EFFICIENCY GUIDELINES:

1. Parallel vs Sequential:
   - Use PARALLEL calls when queries are independent
   - Use SEQUENTIAL only when later queries depend on earlier results
   - Group related queries in single tool_calls block

2. Query Optimization:
   - Start broad, then narrow based on results
   - Use different search parameters for variety
   - Avoid redundant or overlapping queries
   - Each tool call should add unique value

3. Smart Scaling:
   - Begin with essential queries
   - Add detail queries based on initial results
   - Stop when sufficient information gathered
   - Don't over-research simple questions

TASK-SPECIFIC STRATEGIES:

1. Current Events/News:
   - Recent headlines: 2-3 news sources
   - In-depth coverage: 5+ sources including analysis
   - Fact verification: Cross-reference 3+ sources

2. Technical Documentation:
   - Quick reference: 1-2 official sources
   - Implementation guide: 3-4 sources (docs + examples + gotchas)
   - Comprehensive tutorial: 5+ sources covering all aspects

3. Product/Service Research:
   - Basic info: 1-2 sources
   - Comparison shopping: 3-5 sources
   - Detailed analysis: 5+ including reviews, specs, alternatives

4. Scientific/Academic Topics:
   - Basic concepts: 1-2 authoritative sources
   - Current research: 3-5 recent papers/articles
   - Comprehensive review: 5+ sources spanning fundamentals to cutting-edge

MULTIPLE ENTITIES PATTERN:
When researching multiple distinct entities (companies, products, people, locations):
- BAD: "Report about Company A, Company B, Company C"
- GOOD: Separate queries for each entity
  - Query 1: "Company A financial performance"
  - Query 2: "Company B market position"
  - Query 3: "Company C recent developments"
- REASON: Each entity gets full search attention, avoiding diluted results

QUERY FORMULATION BEST PRACTICES:
1. Entity Separation: Search distinct subjects individually
2. Perspective Variation: Use different angles for same topic
   - "benefits of X" → "advantages of X" → "X success stories"
3. Temporal Layering: Mix timeframes
   - "latest developments in X"
   - "X trends 2024-2025"
   - "future of X"
4. Source Diversification: Target different source types
   - Official documentation
   - Independent reviews/analysis
   - Case studies/examples

REFLECTION AND VALIDATION:

1. Pre-Action Reflection: Before each tool use, ask:
   - "Is this tool the most appropriate for this specific information need?"
   - "How does this complement or build upon previous results?"
   - "What specific aspect of the problem does this address?"

2. Result Quality Assessment: After each tool use, evaluate:
   - "Does this result meet my expectations in terms of quality and relevance?"
   - "Are there any gaps or inconsistencies I need to address?"
   - "Should I refine my approach based on what I've learned?"

3. Strategic Adaptation: Throughout the process:
   - "Is my current strategy still optimal given the results so far?"
   - "Do I need to adjust my tool selection or query formulation?"
   - "Have I gathered enough information or do I need additional perspectives?"

4. Final Synthesis Validation: Before providing the answer:
   - "Have I addressed all aspects of the original question?"
   - "Are my conclusions well-supported by the gathered evidence?"
   - "Is there any conflicting information I need to reconcile?"

IMPORTANT RULES:
- Quality over quantity - each tool call must serve a purpose
- Explain your multi-tool strategy in your thought process
- Adapt the number of tools based on result quality
- If initial results are comprehensive, don't add unnecessary calls
- For coding: balance between documentation, examples, and best practices
- Always consider user's implicit needs beyond explicit request
- Employ strategic thinking and reflection at each step
"""  # noqa: E501


REACT_BLOCK_INSTRUCTIONS_MULTI = (
    REACT_BLOCK_MULTI_TOOL_PLANNING
    + """\nAlways follow this exact format in your responses:
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
- Valid JSON only - no markdown formatting
- Double quotes for JSON keys and string values
- No code blocks (```) around JSON
- Proper JSON syntax with commas and brackets
- List each Action and Action Input separately
- Only use tools from the provided list

{{ delegation_instructions }}

**FILE HANDLING:**
- Tools may generate multiple files during execution
- All generated files are automatically collected and returned
- When using multiple tools, files from all tools are aggregated
- Reference all created files in your final Answer

ADDITIONAL RULES:
- Avoid introducing precise figures or program names unless directly supported by cited evidence from the gathered sources.
- Explicitly link key statements to specific findings from the referenced materials to strengthen credibility and transparency.
"""  # noqa: E501
)

REACT_BLOCK_XML_INSTRUCTIONS_MULTI = (
    REACT_BLOCK_MULTI_TOOL_PLANNING
    + """\nAlways use one of these exact XML formats in your responses:
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
- Group parallel tool calls in single <tool_calls> block
- JSON in <input> tags MUST be on single line with proper escaping
- NO line breaks or control characters inside JSON strings
- Use double quotes for JSON strings
- Escape special characters in JSON (\\n for newlines, \\" for quotes)
- Use sequential outputs only when dependencies exist
- Synthesize all results in final answer

JSON FORMATTING REQUIREMENTS:
- Put JSON on single line within tags
- Use double quotes for all strings
- Escape newlines as \\n, quotes as \\"
- NO multi-line JSON formatting

FILE HANDLING WITH MULTIPLE TOOLS:
- Each tool may generate files independently
- Files from all tools are automatically aggregated
- Generated files are returned with the final answer

ADDITIONAL RULES:
- Avoid introducing precise figures or program names unless directly supported by cited evidence from the gathered sources.
- Explicitly link key statements to specific findings from the referenced materials to strengthen credibility and transparency.

{% if delegation_instructions_xml %}
OPTIONAL AGENT PASSTHROUGH:
{{ delegation_instructions_xml }}
{% endif %}
"""  # noqa: E501
)


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

REACT_BLOCK_NO_TOOLS = """Always follow this exact format in your responses:

Thought: [Your detailed reasoning about the user's question]
Answer: [Your complete answer to the user's question]

IMPORTANT RULES:
- ALWAYS start with "Thought:" to explain your reasoning process
- Keep the explanation on the same line as "Thought:" without inserting blank lines or leading spaces
- Use first-person language when thinking or answering (e.g., "I think...", "I recommend...")
- Provide a clear, direct answer after your thought
- If you cannot fully answer, explain why in your thought
- Be thorough and helpful in your response
- Do not mention tools or actions since you don't have access to any
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

{{ delegation_instructions }}
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

{{ delegation_instructions }}
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

HISTORY_SUMMARIZATION_PROMPT = """
Task: Extract valuable information from tool outputs and wrap each in numbered tags.

Format:
Each tool output is marked as: === TOOL_OUTPUT [tool_number] ===

Instructions:
1. Extract relevant information from each marked section
2. Wrap in tags: <tool_outputX>...content...</tool_outputX> (where X = tool number)
3. Example: <tool_output4>extracted content</tool_output4>

Guidelines:
* Always include all required tags for every tool output.
* If the tool output is irrelevant, provide only a general summary of it.
* In output provide only tags and extracted information inside.
* Try to keep information which responds for initial user request and is consistent with previous extracted information.
* Preserve as much important details as possible.
* Do not merge or combine content from different sections.
* Maintain the numbering to match the original section order.
* Do not leave tag empty.

Input request:
"""
