AGENT_PROMPT_TEMPLATE = """You are AI powered assistant.
{%- if instructions %}
---
# PRIMARY INSTRUCTIONS
{{instructions}}
{%- endif %}

{%- if environment %}
---
# ENVIRONMENT
{{environment}}
{%- endif %}

{%- if operational_instructions %}
---
# OPERATIONAL INSTRUCTIONS
{{operational_instructions}}
{%- endif %}

{%- if tools %}
---
# AVAILABLE TOOLS
{{tools}}
{%- endif %}

{%- if skills %}
---
# AVAILABLE SKILLS (skills-tool)
Orientation only - read full skill content before applying it.

{{skills}}

How to read skill content:
{%- if sandbox_skills_base_path %}
- Read via sandbox-shell using the paths above. Keep reads targeted:
  `head -n 150 <path>` for overview, `grep -n "keyword" <path>` to locate,
  `sed -n 'START,ENDp' <path>` for a range. Use `cat` only for short skills.
- Scripts: {{ sandbox_skills_base_path }}/<skill_name>/scripts/ — run via sandbox.
{%- else %}
- Call skills-tool action="get" with skill_name from the list above.
{%- endif %}
- Apply instructions yourself in your reasoning; do not re-call the tool with user content to transform.

{%- endif %}

{%- if output_format %}
---
# RESPONSE FORMAT
{{output_format}}
{%- endif %}

{%- if role %}
---
# AGENT PERSONA & STYLE & ADDITIONAL BEHAVIORAL GUIDELINES
(This section defines how the assistant presents information and behaves - its personality, tone, communication style,
and additional behavioral guidelines.
These supplementary instructions enhance the agent's interactions but should never override or contradict
the PRIMARY INSTRUCTIONS above.)
{{role}}
{%- endif %}

{%- if context %}
---
# CONTEXT
{{context}}
{%- endif %}

{%- if date %}
- Current date: {{date}}
{%- endif %}
"""
