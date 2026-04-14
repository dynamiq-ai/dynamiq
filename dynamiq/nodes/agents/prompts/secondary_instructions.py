"""Secondary instructions for agent prompts."""

DELEGATION_INSTRUCTIONS = (
    "## Delegation\n"
    "- If you want an agent tool's response returned verbatim as the final output, "
    'set "delegate_final": true in that tool\'s input.\n'
    "- Use this only for a single agent tool call "
    "and do not provide your own final answer; the system will return the agent's result directly.\n"
    "- Do not set delegate_final: true inside metadata of the input, it has to be a separate field."
)

DELEGATION_INSTRUCTIONS_XML = (
    "## Delegation\n"
    '- To return an agent tool\'s response as the final output, include "delegate_final": true inside that '
    "tool's <input> or <action_input>. Use this only for a single agent tool call and do not provide an "
    "<answer> yourself; the system will return the agent's result directly."
)

CONTEXT_MANAGER_INSTRUCTIONS = """## Context Management
- Use the context-manager tool proactively when conversation is getting long
- Save critical info (IDs, filenames) in "notes" field BEFORE calling - previous messages will be summarized"""

TODO_TOOLS_INSTRUCTIONS = """## Todo Management
- Use the todo-write tool for complex 3+ step tasks; skip for simple requests
- Current todos shown in [State: ...] at the end of user last messages under "Todos:"
- When creating initial list: first task "in_progress", rest "pending"
- After initial creation, ONLY update status via merge=true - do not restructure the plan
- Mark completed IMMEDIATELY after finishing each step - don't batch
- Only mark completed when FULLY done; if blocked, keep in_progress"""


SANDBOX_INSTRUCTIONS_TEMPLATE = """## Sandbox Environment
You operate inside a persistent sandbox filesystem.
The sandbox directory is your working memory.

- Use {base_path}/ for ALL files: scripts, research, logs, intermediate artifacts, data, and final output.
- Files returned from other tools are also placed in {base_path}/.
- Uploaded files are placed in {base_path}/input/.
- Other tools can ONLY access files under {base_path}/.

## Persistence
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

## Execution Rules
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


REACT_BLOCK_MULTI_TOOL_PLANNING = """## Multi-Tool Planning
- Scale tool usage to task complexity: answer directly when possible, use 1-2 tools for simple lookups,
  3-5 for multi-source research, more for comprehensive analysis.
- Use PARALLEL calls for independent queries; SEQUENTIAL only when results feed into the next call.
- When researching multiple entities, search each one separately - don't combine into one query.
- Each tool call must add unique value. Avoid redundant or semantically similar queries.
- Start broad, then narrow. Stop once sufficient information is gathered.
- Verify that all parts of the original question are addressed before delivering the final answer.
"""

SUB_AGENT_INSTRUCTIONS = """## Sub-Agent Tools
Sub-agent tools are specialized agents you can delegate tasks to. Each sub-agent is an independent agent \
with its own tools and expertise - use them to break complex work into focused subtasks.

Execution modes (shown in each tool's description):
- "[Independent agent: ...]" - spawns a fresh instance per call. Safe to call in parallel with other tools.
- "[Shared agent: ...]" - reuses a single instance. Calls run sequentially; do not call in parallel.

Provide each sub-agent with a clear, self-contained task description in the "input" field."""
