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
