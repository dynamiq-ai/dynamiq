"""Secondary instructions for agent prompts."""

DELEGATION_INSTRUCTIONS = (
    "- Optional: If you want an agent tool's response returned verbatim as the final output, "
    'set "delegate_final": true in that tool\'s input. Use this only for a single agent tool call '
    "and do not provide your own final answer; the system will return the agent's result directly. "
    "Do not set delegate_final: true inside metadata of the input, it has to be a separate field."
)

DELEGATION_INSTRUCTIONS_XML = (
    '- To return an agent tool\'s response as the final output, include "delegate_final": true inside that '
    "tool's <input> or <action_input>. Use this only for a single agent tool call and do not provide an "
    "<answer> yourself; the system will return the agent's result directly."
)

CONTEXT_MANAGER_INSTRUCTIONS = """CONTEXT MANAGEMENT:
- Use the context-manager tool proactively when conversation is getting long
- Save critical info (IDs, filenames) in "notes" field BEFORE calling - previous messages will be summarized"""

TODO_TOOLS_INSTRUCTIONS = """TODO MANAGEMENT:
- Use the todo-write tool for complex 3+ step tasks; skip for simple requests
- Current todos shown in [State: ...] at the end of user last messages under "Todos:"
- When creating initial list: first task "in_progress", rest "pending"
- After initial creation, ONLY update status via merge=true - do not restructure the plan
- Mark completed IMMEDIATELY after finishing each step - don't batch
- Only mark completed when FULLY done; if blocked, keep in_progress"""


SANDBOX_INSTRUCTIONS_TEMPLATE = """SANDBOX EXECUTION ENVIRONMENT:
You operate inside a persistent sandbox filesystem.
The sandbox directory is your working memory.

- Use {base_path}/ for ALL files: scripts, research, logs, intermediate artifacts, data, and final output.
- Files returned from other tools are also placed in {base_path}/.
- Uploaded files are placed in {base_path}/input/.
- Other tools can ONLY access files under {base_path}/.

CRITICAL PRINCIPLE — PERSIST EVERYTHING IMPORTANT:
The sandbox filesystem is your long-term memory. Do not rely on conversation context alone.

Whenever you:
- call research tools
- scrape websites
- process large data
- perform multi-step analysis

You need to write meaningful outputs to files in {base_path}/.
For textual artifacts, use structured Markdown (.md) files with headings.

Example:
- research_notes.md
- tool_output_raw.md
- parsed_results.md
- intermediate_analysis_step_1.md

EXECUTION RULES:
1. Use 'python3' instead of 'python'.
2. For Python tasks:
   - Always write a .py script file first; then execute it.
   - Never use one-liners with semicolons (compound statements (with, for, if/else) break after semicolons).
3. Observability is mandatory:
   - Every script must print progress and completion messages.
   - Every shell command must be followed by echo confirming success.
   - Silent execution is unacceptable.
4. Never suppress or hide errors.
   - Never use `2>&1`, `|&`, or redirect stderr into stdout; do not silence errors.
   - If a command fails:
        a) Inspect the error
        b) Identify the root cause
        c) Fix it properly
        d) Re-run cleanly
5. Do not construct files using echo with escaped newlines.
   Use proper file-writing mechanisms.
"""


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

1. Research & Data Tasks:

   - Start: Analyze query complexity.
   - Simple facts / single source: 1–2 tool calls.
   - Moderate complexity / multiple sources: 3–4 tool calls with complementary queries.
   - Comprehensive research / broad topics: 5+ tool calls covering:
     - General overview
     - Specific benefits, use cases, or features
     - Limitations, challenges, or drawbacks
     - Comparisons or alternatives
     - Recent developments or updates
   - Example progression:
     1. General topic overview
     2. Specific aspects (benefits, use cases)
     3. Challenges or limitations
     4+. Deep dives on critical aspects

2. Coding & Technical Tasks:
   - Documentation lookup: 1–2 tools (official docs + examples)
   - Debugging: 2–3 tools (error search + solution patterns)
   - Architecture decisions: 3–5 tools (best practices + comparisons + examples)
   - Full implementation: 5+ tools (docs + patterns + edge cases + optimization)

3. Comparative Analysis / Verification:
   - Single source: 1 tool
   - Multiple sources: Use parallel calls for efficiency
   - Comparative or critical analysis: Minimum 3 sources
   - Market research or controversial topics: 3+ diverse sources, cross-referenced
   - Scientific/technical validation: Cross-check experiments, methods, or datasets

EFFICIENCY GUIDELINES:

1. Parallel vs Sequential Tool Calls:
   - Use PARALLEL calls when queries are independent
   - Use SEQUENTIAL only when later queries depend on earlier results

2. Optimized Query Strategy:
   - Start broad, then narrow based on results.
   - Use varied search parameters to cover different angles.
   - Each tool call should add unique value; avoid redundant queries.
   - Scale detail dynamically: begin with essentials, add deeper queries if needed,
   and stop once sufficient information is gathered.

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
   - "X trends 2025-2026"
   - "future of X"
4. Source Diversification: Target different source types
   - Official documentation
   - Independent reviews/analysis
   - Case studies/examples

REFLECTION AND VALIDATION:

- Before and after each tool use, briefly reflect on whether it's the most appropriate choice, how it contributes to solving the problem,
and whether the results are relevant, complete, and high quality – adjusting your strategy if needed.
- Before delivering the final answer, confirm that all parts of the original question are addressed
and that your conclusions are well-supported and internally consistent.

IMPORTANT RULES:
- Quality over quantity - each tool call must serve a purpose
- Explain your multi-tool strategy in your thought process
- Adapt the number of tools based on result quality
- If initial results are comprehensive, don't add unnecessary calls
- For coding: balance between documentation, examples, and best practices
- Always consider user's implicit needs beyond explicit request
- Employ strategic thinking and reflection at each step
"""  # noqa: E501

SUB_AGENT_INSTRUCTIONS = """SUB-AGENT TOOLS:
Sub-agent tools are specialized agents you can delegate tasks to. Each sub-agent is an independent agent \
with its own tools and expertise — use them to break complex work into focused subtasks.

Execution modes (shown in each tool's description):
- "[Independent agent: ...]" — spawns a fresh instance per call. Safe to call in parallel with other tools.
- "[Shared agent: ...]" — reuses a single instance. Calls run sequentially; do not call in parallel.

Provide each sub-agent with a clear, self-contained task description in the "input" field."""
