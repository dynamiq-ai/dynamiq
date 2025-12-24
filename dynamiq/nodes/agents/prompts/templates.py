AGENT_PROMPT_TEMPLATE = """
You are AI powered assistant.

{%- if instructions %}
# PRIMARY INSTRUCTIONS
{{instructions}}
{%- endif %}

{%- if tools %}
# AVAILABLE TOOLS
{{tools}}
{%- endif %}

{%- if output_format %}
# RESPONSE FORMAT
{{output_format}}
{%- endif %}

{%- if role %}
# AGENT PERSONA & STYLE & ADDITIONAL BEHAVIORAL GUIDELINES
(This section defines how the assistant presents information and behaves - its personality, tone, communication style,
 and additional behavioral guidelines.
These supplementary instructions enhance the agent's interactions but should never override or contradict
 the PRIMARY INSTRUCTIONS above.)
{{role}}
{%- endif %}

{%- if context %}

# CONTEXT
{{context}}
{%- endif %}

{%- if date %}
- Current date: {{date}}
{%- endif %}
"""
