"""Local checks on a flow JSON, before anything is saved.

The platform accepts a flow that cannot run: an unknown key is dropped rather
than rejected, so a misplaced selector yields empty output instead of an error,
and a tool whose schema is wrong is simply never callable. Everything here is a
rule learned from a flow that saved cleanly and then did not work.

`validate(flow, known_types=None) -> (errors, warnings)`; errors block a save, warnings do not.

`known_types` is the platform's own list of node types, from GET /v1/agent-builder/nodes.
Nothing here reads the SDK's package layout to guess it: the folder a class happens to live
in is not the same question as what the API accepts, and reading one to answer the other made
this reject the SDK's own emitted types for knowledge-base nodes. Omit it and the type check
is skipped rather than guessed at.

A Rules node's checks, `applies_when`, derived values and messages, and an Expression node's
expressions, are Jinja2 text and are parsed here too (`check_expressions`), wherever the node
sits: at the top of the flow or inside another, a Map's `node` say. This module stays free of an
import of the SDK engine even so - that alone costs about 4.7s, which a `validate` call cannot
spend - so `RULE_HELPERS`, `RULE_TESTS` and `RULE_RESERVED_NAMES` below are a hand-kept copy of
`dynamiq.nodes.operators.rules.HELPERS`, `.TESTS` and `.RESERVED_NAMES`, and `_path_of` of the
engine's own; a drift test in the test file, which may import the engine, is the guard against
the names falling out of step, the filter and global names included.
"""
from __future__ import annotations

import difflib
import re

from jinja2 import TemplateSyntaxError
from jinja2 import nodes as jinja_nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment

NODE_ID_RE = re.compile(r"^[a-z0-9]([a-z0-9]|-[a-z0-9])*$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

INPUT_TYPE = "dynamiq.nodes.utils.Input"
OUTPUT_TYPE = "dynamiq.nodes.utils.Output"

# The decision operators the platform runs a rules workflow with. Their shape is checked here
# the way the platform checks it on save, so a flow written by hand fails before it is sent.
CHOICE_TYPE = "dynamiq.nodes.operators.Choice"
DECISION_TABLE_TYPE = "dynamiq.nodes.operators.DecisionTable"
RULES_TYPE = "dynamiq.nodes.operators.Rules"
EXPRESSION_TYPE = "dynamiq.nodes.operators.Expression"
SUB_WORKFLOW_TYPE = "dynamiq.nodes.operators.SubWorkflow"
JUDGEMENT_TYPE = "dynamiq.nodes.tools.Judgement"

CHOICE_HIT_POLICIES = ("first", "all")
TABLE_HIT_POLICIES = ("first", "unique", "collect")
TABLE_AGGREGATIONS = ("list", "sum", "min", "max", "count")
RULE_SEVERITIES = ("fail", "warn", "info")
RULE_MISSING_POLICIES = ("not_evaluated", "fail", "not_applicable")
QUESTION_TYPES = ("noul", "choice", "score")
CONFIDENCE_MODES = ("verbalized", "sampling")
SYSTEM_ONE_TYPE = "dynamiq.nodes.detectors.SystemOne"
JUDGE_TYPE_PREFIXES = ("dynamiq.nodes.llms.", "dynamiq.nodes.agents.")
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MAX_SAMPLES = 10
DEFAULT_SAMPLES = 1
# The output a decision table adds beside its columns, so no column may take it.
MATCHED_RULES_KEY = "matched_rules"
# A name an expression can read: an input, a derived value or an output key.
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Mirrors dynamiq.nodes.operators.rules.HELPERS - the callable names a Rules check, `applies_when`,
# derived value or Expression item may call as `name(...)` - and .TESTS - the tests it adds to
# Jinja's own (`is defined`, `is none`, ...). See the module docstring for why this is a hand-kept
# copy rather than an import.
RULE_HELPERS = frozenset(
    {
        "has",
        "days_between",
        "date",
        "today",
        "len",
        "abs",
        "min",
        "max",
        "sum",
        "round",
        "text",
        "number",
        "first_present",
    }
)
RULE_TESTS = frozenset({"present", "blank"})
# Mirrors dynamiq.nodes.operators.rules.RESERVED_NAMES - the helpers there were before the
# vocabulary grew. The engine refuses to build a derived value named after one, and an expression
# that calls one and reads the same name as a value; a name added since (`text`, `number`,
# `first_present`) it lets a flow that already uses it keep building.
RULE_RESERVED_NAMES = frozenset({"has", "days_between", "date", "today", "len", "abs", "min", "max", "sum", "round"})
# Jinja binds this name inside every expression, so the engine refuses to build one that reads it
# (rules.py's RESERVED_ROOT).
RESERVED_ROOT = "self"
# The tests that only ask about the value they test (rules.py's _EXEMPT_TESTS): the engine reads
# nothing else of one, not what `sameas` compares the value with, as it reads nothing of a keyword
# argument of `default`, so a name there refuses no build.
_EXEMPT_TESTS = frozenset({"defined", "undefined", "none", "sameas", "present", "blank"})
# The Rules node's own input, an ISO date fixing the effective-window comparison. Never declared
# in `input_fields`, but always readable, so it counts as a known root the way a declared input does.
AS_OF_KEY = "as_of"

# A jinja2-only environment (no SDK import) used only to `parse()` expression text into an AST and
# look up names in it - never to compile or evaluate one, so a record never reaches it. Its
# filters and tests are Jinja's own defaults, untouched by the engine's sandbox (RecordSandbox in
# rules.py wraps a few filters without changing their names, so a plain environment's filter names
# already match); RULE_HELPERS (as globals) and RULE_TESTS are the engine's own addition on top.
_EXPRESSION_ENVIRONMENT = ImmutableSandboxedEnvironment()
_FILTER_NAMES = frozenset(_EXPRESSION_ENVIRONMENT.filters)
_TEST_NAMES = frozenset(_EXPRESSION_ENVIRONMENT.tests) | RULE_TESTS
_GLOBAL_NAMES = frozenset(_EXPRESSION_ENVIRONMENT.globals) | RULE_HELPERS
# What `_parse_and_walk` puts around the author's own text before parsing it - never seen by the
# author, so a syntax error naming a piece of it (a raw character position, or the wrapper's own
# closing brace swallowed as a mismatched bracket) needs translating back before it is shown; see
# `_without_wrapper_leak`.
_EXPRESSION_PREFIX = "{{ "

# Shapes people write when they think a flow is something else.
WRONG_SHAPE_KEYS = {
    "actions": "a Pipedream component list",
    "steps": "a step/pipeline config",
    "workflow": "a wrapper around the flow - pass the flow itself",
    "data": "a full API response - pass `data.flow`, not the whole body",
}

PLACEHOLDER_HINTS = ("your-", "example-", "desired-", "target-page", "-id-here")
# A slot name: `<PAGE_ID>`, `<your page id>`. Not a Slack mention (`<@U123>`, `<#C1>`, which
# do not start with a letter) and not markup (`<b>`, `<div>`, which are lowercase and unspaced).
PLACEHOLDER_TEMPLATE = re.compile(r"<[A-Za-z_][\w .-]*>")
# Free text: a "<" or a word ending in "-id" here is prose, not an unfilled template.
PROSE_KEYS = frozenset({"role", "description", "label", "instructions", "prompt", "content", "system_prompt"})


def unwrap_response_format(value: dict) -> dict:
    """The raw schema inside a response_format, however it was written.

    Agent.response_format normalizes on the way in - a `mode="before"` validator routes every
    value through `unwrap_response_format`, which strips litellm's
    `{"type": "json_schema", "json_schema": {"schema": ...}}` wrapper. Reading `properties` off
    the value as written therefore rejects the wrapped form, whose top-level keys are `type`
    and `json_schema`, even though it constrains the answer perfectly well - and the repo's own
    examples are written that way.
    """
    if value.get("type") == "json_schema" and "json_schema" in value:
        inner = value["json_schema"]
        if isinstance(inner, dict):
            schema = inner.get("schema")
            return schema if isinstance(schema, dict) else inner
    return value


def looks_like_placeholder(value) -> bool:
    """Whether a string is an unfilled template rather than real content."""
    if not isinstance(value, str) or len(value) >= 200:
        return False
    text = value.strip().lower()
    if any(hint in text for hint in PLACEHOLDER_HINTS):
        return True
    for match in PLACEHOLDER_TEMPLATE.findall(value):
        inner = match[1:-1]
        if " " in inner or "_" in inner or inner.isupper():
            return True
    return False


def walk_strings(value, path="$"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def slug_for(name) -> str:
    """The id-shaped form of a name, for the "use this instead" hint."""
    slug = re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-"))
    return slug or "my-node"


def named_parts(node):
    """Everything in a node that carries its own API-validated `name`."""
    label = node.get("id", "?")
    yield f"node {label!r}", node
    llm = node.get("llm")
    if isinstance(llm, dict):
        yield f"llm on node {label!r}", llm
    for tool in node.get("tools") or []:
        if isinstance(tool, dict):
            yield f"tool {tool.get('name') or tool.get('type', '?')} on node {label!r}", tool


def coerce_depends(value) -> list:
    """`depends` shorthand, matching what `normalize_flow` accepts on save.

    A bare string is one dependency, not a sequence of characters: iterating it produced one
    invented error per letter and buried the real problems under them.
    """
    if isinstance(value, str):
        return [{"node": value}]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [{"node": d} if isinstance(d, str) else d for d in value]
    return []


def _as_dict(value) -> dict:
    """`value` when it is already an object, else `{}` - the shape a JSON key should have had.

    Shared by every reader of `input_transformer`/`input_transformer.selector` (`validate` below
    and `_declared_roots` further down) so "missing or the wrong type" is one rule, not one written
    out at each site. A caller that also needs to REPORT the wrong type still checks it directly
    first (this throws that fact away on purpose) - only the safe, coerced value is common.
    """
    return value if isinstance(value, dict) else {}


def validate(flow, known_types: set | None = None):
    """Return (errors, warnings)."""
    errors, warnings = [], []

    if not isinstance(flow, dict):
        return [f"flow must be a JSON object, got {type(flow).__name__}."], warnings

    nodes = flow.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        hint = next((why for key, why in WRONG_SHAPE_KEYS.items() if key in flow), None)
        detail = f" This looks like {hint}." if hint else ""
        return [
            "flow has no `nodes` list." + detail + ' A flow is {"id": "<uuid>", "nodes": [...]} '
            "where each node has id/name/type - copy the template in SKILL.md, or run "
            "`dynamiq workflow get <id>` on a workflow that works."
        ], warnings

    flow_id = flow.get("id")
    if flow_id is not None and not UUID_RE.match(str(flow_id)):
        warnings.append(f"flow.id {flow_id!r} is not a UUID; the CLI will generate one.")

    ids = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{index}] is not an object.")
            continue
        node_id = node.get("id")
        if not node_id:
            errors.append(f"nodes[{index}] has no `id`.")
        else:
            if not NODE_ID_RE.match(str(node_id)):
                errors.append(
                    f"node id {node_id!r} is not a valid slug - lowercase letters, digits and "
                    "single hyphens only (e.g. 'notion-agent')."
                )
            ids.append(node_id)
        if not node.get("type"):
            errors.append(f"node {node_id or index!r} has no `type` (e.g. {INPUT_TYPE}).")

    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        errors.append(f"duplicate node ids: {', '.join(duplicates)}. Node ids must be unique.")
    known = set(ids)
    by_id = {node.get("id"): node for node in nodes if isinstance(node, dict)}

    types = [n.get("type") for n in nodes if isinstance(n, dict)]
    inputs = types.count(INPUT_TYPE)
    if inputs == 0:
        errors.append(f"no Input node. Every flow starts with one node of type {INPUT_TYPE}.")
    elif inputs > 1:
        errors.append(f"{inputs} Input nodes; a flow has exactly one.")

    if not [t for t in types if t not in (INPUT_TYPE, OUTPUT_TYPE)]:
        errors.append(
            "this flow only has Input/Output nodes, so it does nothing. Add the agent or tool "
            "node that does the work."
        )

    if types.count(OUTPUT_TYPE) == 0:
        errors.append(
            f"no Output node, so this flow returns nothing to the caller. Add a node of type "
            f"{OUTPUT_TYPE} that depends on the last working node and selects its output."
        )

    for node_type in types:
        if not node_type:
            continue
        text = str(node_type)
        parts = text.split(".")
        # A short label like "llm" or "exa-search" gets a bare `"type": "must be a valid
        # value"` from the API, naming neither the node nor the fix.
        if not text.startswith("dynamiq.nodes.") or len(parts) < 4:
            hint = " Did you mean a `dynamiq.nodes.<group>.<Class>` path?" if "." not in text else ""
            errors.append(
                f"type {node_type!r} is not a node type. Types are fully qualified dotted paths "
                f"like `dynamiq.nodes.agents.Agent`, `dynamiq.nodes.llms.OpenAI` or "
                f"`dynamiq.nodes.tools.Pipedream` - never a short label.{hint} "
                "Copy the exact string from `dynamiq workflow get <id>` on a workflow that works."
            )
            continue
        if known_types and text not in known_types:
            errors.append(
                f"type {node_type!r} is not a type the platform accepts. Third-party apps "
                "(Notion, Slack, GitHub) are NOT their own node types: they are "
                "`dynamiq.nodes.tools.Pipedream` tools placed inside an agent's `tools` array. "
                "Run `dynamiq workflow node-types` for the real list."
            )

    agent_ids = {
        n.get("id") for n in nodes
        if isinstance(n, dict) and str(n.get("type", "")).startswith("dynamiq.nodes.agents.")
    }
    if not agent_ids:
        warnings.append(
            "this flow has no Agent node - it is a fixed pipeline, not an agent. If the user asked "
            "for an agent, add a dynamiq.nodes.agents.Agent node and put the tools inside it."
        )

    for node in nodes:
        if not isinstance(node, dict):
            continue
        label = node.get("id", "?")
        node_type = str(node.get("type") or "")

        for dependency in coerce_depends(node.get("depends")):
            target = dependency.get("node") if isinstance(dependency, dict) else dependency
            if isinstance(target, dict):
                target = target.get("id")
            if target not in known:
                errors.append(f"node {label!r} depends on {target!r}, which is not a node in this flow.")
                continue
            option = dependency.get("option") if isinstance(dependency, dict) else None
            if option is not None:
                errors.extend(check_branch(by_id.get(target), target, option, label))
        if node_type == INPUT_TYPE and node.get("depends"):
            errors.append(f"Input node {label!r} must not depend on anything.")
        if node_type != INPUT_TYPE and not node.get("depends"):
            errors.append(f"node {label!r} has no `depends`, so it never runs. Wire it to an upstream node.")

        transformer = node.get("input_transformer") or {}
        if not isinstance(transformer, dict):
            errors.append(
                f"node {label!r}: `input_transformer` must be an object like "
                f'{{"selector": {{"field": "$.node.output.x"}}}}, got {type(transformer).__name__}. '
                "The JSONPath goes inside `selector`, not on `input_transformer` itself."
            )
        transformer = _as_dict(transformer)
        selector = transformer.get("selector") or {}
        if not isinstance(selector, dict):
            errors.append(f"node {label!r}: input_transformer.selector must be an object of field -> JSONPath.")
        selector = _as_dict(selector)
        # The runtime resolves more than `$.<node-id>...`: `input_transformer.path` selects a
        # sub-tree first, so a selector under it is relative and names no node at all; bracket
        # notation addresses the same thing as dotted; and a literal is a legal constant.
        # Checking the node name is only meaningful for an absolute, dotted, path-less selector.
        relative = bool(transformer.get("path"))
        for field, expression in selector.items():
            if not isinstance(expression, str):
                continue                      # a literal constant is legal
            if not expression.startswith("$"):
                continue                      # so is a plain string value
            if relative:
                continue                      # resolved against `path`, not against a node
            match = re.match(r"^\$\.([A-Za-z0-9_-]+)\b", expression) or \
                re.match(r"^\$\[[\"']([^\"']+)[\"']\]", expression)
            if not match:
                continue                      # a shape this checker does not model
            source = match.group(1)
            if source not in known:
                errors.append(
                    f"node {label!r}: selector {field!r} reads from {source!r}, which is not a node in "
                    "this flow. An agent's tools are NOT nodes - read the agent's own output instead."
                )

        if "selector" in node:
            errors.append(
                f"node {label!r} has a top-level `selector`. The API ignores it silently - move it to "
                "`input_transformer.selector`."
            )

        for part_label, part in named_parts(node):
            sub_type = part.get("type")
            if part is not node and sub_type is not None:
                text = str(sub_type)
                if not text.startswith("dynamiq.nodes.") or len(text.split(".")) < 4:
                    errors.append(
                        f"{part_label}: type {sub_type!r} is not a node type. Use the fully "
                        "qualified path, e.g. `dynamiq.nodes.llms.OpenAI` or "
                        "`dynamiq.nodes.tools.Pipedream`."
                    )
            name = part.get("name")
            if name is not None and not NODE_ID_RE.match(str(name)):
                errors.append(
                    f"{part_label}: name {name!r} must be in a valid format - the API validates `name` "
                    "like an id (lowercase letters, digits, single hyphens). Use e.g. "
                    f"{slug_for(name)!r}. "
                    "Human-readable text belongs in `role`/`description`."
                )

        if node_type.startswith("dynamiq.nodes.agents."):
            llm = node.get("llm")
            if not isinstance(llm, dict):
                errors.append(f"agent {label!r} has no `llm` object.")
            elif requirement_problems(llm.get("connection"), f"agent {label!r}: llm.connection") is not None:
                errors.extend(requirement_problems(llm.get("connection"), f"agent {label!r}: llm.connection"))
            elif not UUID_RE.match(str(llm.get("connection") or "")):
                errors.append(
                    f"agent {label!r}: llm.connection {llm.get('connection')!r} is not a connection UUID. "
                    "Run `dynamiq connection list --type dynamiq.connections.OpenAI`."
                )
            if not str(node.get("role") or "").strip():
                warnings.append(f"agent {label!r} has no `role`, so it has no instructions.")
            if not node.get("tools"):
                warnings.append(
                    f"agent {label!r} has an empty `tools` array - it can only talk, not act. "
                    "Intentional for a plain Q&A agent."
                )

            # `memory` saves and deploys even when it can never switch on.
            memory = node.get("memory")
            if memory is not None:
                if not isinstance(memory, dict):
                    errors.append(f"agent {label!r}: `memory` must be an object, got {type(memory).__name__}.")
                else:
                    backend = memory.get("backend")
                    backend_ref = memory.get("backend_ref")
                    if backend and backend_ref:
                        errors.append(
                            f"agent {label!r}: `memory` has both `backend` and `backend_ref`. They are "
                            "alternatives - keep exactly one."
                        )
                    elif not backend and not backend_ref:
                        errors.append(
                            f"agent {label!r}: `memory` needs a `backend` (or a `backend_ref` to a saved "
                            'one), e.g. {"type": "dynamiq.memory.backends.Dynamiq", "memory_id": "<uuid>"}.'
                        )
                    elif isinstance(backend, dict):
                        backend_type = str(backend.get("type") or "")
                        if backend_type not in MEMORY_BACKENDS:
                            errors.append(
                                f"agent {label!r}: memory.backend.type {backend.get('type')!r} is not a "
                                "backend the PLATFORM accepts, so `workflow save` will reject this flow "
                                "even if the SDK runs it (InMemory and SQLite are SDK-only). Use one of: "
                                + ", ".join(sorted(MEMORY_BACKENDS))
                                + "."
                            )
                        elif backend_type.endswith(".Dynamiq") and not UUID_RE.match(
                            str(backend.get("memory_id") or "")
                        ):
                            errors.append(
                                f"agent {label!r}: memory.backend.memory_id "
                                f"{backend.get('memory_id')!r} is not a UUID. Create one with "
                                "`POST /v1/memories` and use the id it returns."
                            )
                    if memory.get("save_mode") not in (None, "full", "input_output"):
                        errors.append(
                            f"agent {label!r}: memory.save_mode {memory.get('save_mode')!r} is not valid. "
                            'Use "full" or "input_output".'
                        )
                    # Memory switches on when the agent RECEIVES user_id or session_id, not
                    # when a selector maps them: Agent.run reads them off its own input schema
                    # (`self.memory and (input_data.user_id or input_data.session_id)`), and
                    # transform_input has two shapes that deliver them without any selector -
                    # no transformer at all passes the flow payload straight through, and a
                    # `path` alone hands over that whole sub-tree. Only an explicit selector
                    # can drop them, and even then the flow may supply them another way, so
                    # this is a warning.
                    explicit = bool(selector) and not transformer.get("path")
                    if explicit and not ({"user_id", "session_id"} & set(selector)):
                        warnings.append(
                            f"agent {label!r} has `memory`, and its selector lists neither `user_id` nor "
                            "`session_id`. Memory only engages when the agent receives one of them, and a "
                            "selector replaces the input rather than adding to it - so unless the caller "
                            'supplies them another way, add "user_id": "$.input.output.user_id" here.'
                        )

            # STRUCTURED_OUTPUT without a schema is the mode change without the guarantee.
            response_format = node.get("response_format")
            if response_format is not None and not isinstance(response_format, dict):
                errors.append(
                    f"agent {label!r}: `response_format` must be a JSON Schema object, got "
                    f"{type(response_format).__name__}."
                )
            elif isinstance(response_format, dict) and not unwrap_response_format(
                response_format
            ).get("properties"):
                errors.append(
                    f"agent {label!r}: `response_format` has no `properties`, so it constrains nothing. "
                    'Use e.g. {"type": "object", "properties": {"answer": {"type": "string"}}, '
                    '"required": ["answer"]}, or the litellm-wrapped form with the same schema '
                    'under json_schema.schema.'
                )
            if isinstance(llm, dict) and llm.get("response_format") is not None and response_format is None:
                warnings.append(
                    f"agent {label!r}: `response_format` is on the `llm` sub-object, which shapes a raw "
                    "model call, not the agent's answer. Move it onto the agent node to fix the shape "
                    "of `output.content`."
                )

        for tool in node.get("tools") or []:
            if not isinstance(tool, dict):
                errors.append(f"node {label!r}: every entry in `tools` must be an object.")
                continue
            where = f"tool {tool.get('name') or tool.get('type', '?')} on node {label!r}"
            if tool.get("type") == "dynamiq.nodes.tools.Pipedream":
                tool_errors, tool_advisory = check_pipedream(tool, where)
                errors.extend(tool_errors)
                warnings.extend(tool_advisory)
            # A Judgement is built to be an agent's tool, so it reaches the loader from here too.
            elif tool.get("type") == JUDGEMENT_TYPE:
                errors.extend(check_judgement(tool, f"{tool.get('name') or 'judgement'} on node {label}"))
            elif (tool.get("connection") is not None
                    and requirement_problems(tool.get("connection"), f"{where}: connection") is not None):
                errors.extend(requirement_problems(tool.get("connection"), f"{where}: connection"))
            elif tool.get("connection") is not None and not UUID_RE.match(str(tool.get("connection"))):
                errors.append(
                    f"{where}: connection {tool.get('connection')!r} is not a UUID. "
                    "Run `dynamiq connection list`."
                )

        # The API reports these as `cannot be blank` with no node name.
        if node_type.startswith("dynamiq.nodes.llms."):
            errors.extend(llm_requirements(node, f"node {label!r}"))

        # A Pipedream node placed in the DAG is validated exactly like one inside an agent.
        if node_type == "dynamiq.nodes.tools.Pipedream":
            node_errors, node_advisory = check_pipedream(node, f"node {label!r}")
            errors.extend(node_errors)
            warnings.extend(node_advisory)

        # A Judgement that depends on an agent judges that agent's answer; it belongs in the DAG, not in `tools`.
        if node_type.startswith("dynamiq.nodes.tools.") and node_type != JUDGEMENT_TYPE:
            after = [
                d.get("node") for d in coerce_depends(node.get("depends"))
                if isinstance(d, dict) and d.get("node") in agent_ids
            ]
            if after:
                warnings.append(
                    f"node {label!r} is a standalone step that runs AFTER agent {after[0]!r} and receives "
                    'its finished prose. To give the agent a tool it can call, move this into that '
                    "agent's \"tools\" array instead."
                )

        if node_type.startswith("dynamiq.nodes.operators.") or node_type == JUDGEMENT_TYPE:
            errors.extend(check_operator(node, label))
        if node_type in (RULES_TYPE, EXPRESSION_TYPE):
            expression_errors, expression_warnings = check_expressions(node, label)
            errors.extend(expression_errors)
            warnings.extend(expression_warnings)
        # The loader builds a node defined inside this one with the flow, so a Rules or an
        # Expression node there compiles its text as a top-level one does, and is checked the
        # same way, labelled with the path to it.
        for nested_label, nested in nested_nodes(node, label):
            if nested.get("type") in (RULES_TYPE, EXPRESSION_TYPE):
                errors.extend(check_operator(nested, nested_label))
                expression_errors, expression_warnings = check_expressions(nested, nested_label)
                errors.extend(expression_errors)
                warnings.extend(expression_warnings)

    for path, text in walk_strings(flow):
        if path.rsplit(".", 1)[-1] in PROSE_KEYS or len(text) > 200:
            continue
        if looks_like_placeholder(text):
            errors.append(f"{path} is still the placeholder {text!r} - replace it with a real value.")

    return errors, warnings


def nested_nodes(node: dict, label):
    """Each node defined inside `node`, at any depth, with its label: the path to it,
    `map-1 > rules-1`.

    A node is an object with a dotted `type` held by a node's field, alone or in a list: a Map's
    `node`, an agent's `llm` and `tools`, a Judgement's `judge`, as the loader follows them for the
    flows they reference. The loader builds a node held deeper as well, in a mapping with no `type`
    or in a list inside a list, which this does not reach. A flow a node names by its id, in `flow`
    or `flows`, is a flow of its own and is not followed.
    """
    for key, value in node.items():
        for index, item in enumerate(value if isinstance(value, list) else [value]):
            if isinstance(item, dict) and isinstance(item.get("type"), str) and "." in item["type"]:
                place = f"{key}[{index}]" if isinstance(value, list) else key
                item_label = f"{label} > {item.get('id') or item.get('name') or place}"
                yield item_label, item
                yield from nested_nodes(item, item_label)


def check_branch(source, source_id, option, label) -> list:
    """A dependency's `option` names a branch of a Choice: one of its options, by id."""
    if not isinstance(source, dict) or source.get("type") != CHOICE_TYPE:
        return [
            f"node {label!r} depends on option {option!r} of {source_id!r}, which is not a Choice node. "
            "Only a Choice has branches; drop `option` or point it at the Choice."
        ]
    options = [o for o in (source.get("options") or []) if isinstance(o, dict)]
    if any(option == o.get("id") for o in options):
        return []
    # The runtime matches the option's id alone: a name would pass here and gate nothing there, so the
    # branch's node would run whatever the Choice decided.
    if by_name := next((o for o in options if option == o.get("name") and o.get("id")), None):
        return [
            f"node {label!r} depends on option {option!r} of choice {source_id!r} by its name; the runtime "
            f"matches the option's id, so the branch would never gate it. Use {by_name['id']!r}."
        ]
    ids = ", ".join(str(o.get("id") or o.get("name")) for o in options) or "none"
    return [
        f"node {label!r} depends on option {option!r} of choice {source_id!r}, which has no such "
        f"option (it has: {ids}). Use the option's id."
    ]


def check_operator(node, label) -> list:
    """What the platform refuses on save for a decision operator, phrased for the author."""
    node_type = node.get("type")
    if node_type == CHOICE_TYPE:
        return check_choice(node, label)
    if node_type == DECISION_TABLE_TYPE:
        return check_decision_table(node, label)
    if node_type == RULES_TYPE:
        return check_rules(node, label)
    if node_type == EXPRESSION_TYPE:
        return check_expression(node, label)
    if node_type == SUB_WORKFLOW_TYPE:
        return check_sub_workflow(node, label)
    if node_type == JUDGEMENT_TYPE:
        return check_judgement(node, label)
    return []


def check_choice(node, label) -> list:
    errors = []
    options = node.get("options")
    if not isinstance(options, list) or not options:
        errors.append(
            f'choice {label!r} has no `options`. Each option is {{"id", "name", "condition"}}; '
            "an option without a condition is the fallback that runs when no other holds."
        )
    else:
        for index, option in enumerate(options):
            if not isinstance(option, dict) or not option.get("id") or not option.get("name"):
                errors.append(f"choice {label!r}: options[{index}] needs `id` and `name`.")
    policy = node.get("hit_policy")
    if policy is not None and policy not in CHOICE_HIT_POLICIES:
        errors.append(
            f"choice {label!r}: hit_policy {policy!r} is not one of {', '.join(CHOICE_HIT_POLICIES)}: "
            "`first` runs the first branch whose condition holds, `all` runs every one."
        )
    return errors


def check_decision_table(node, label) -> list:
    errors = []
    inputs = [c for c in (node.get("input_columns") or []) if isinstance(c, dict)]
    outputs = [c for c in (node.get("output_columns") or []) if isinstance(c, dict)]
    if not inputs:
        errors.append(f'decision table {label!r} has no `input_columns` ({{"id", "name", "type"}} each).')
    if not outputs:
        errors.append(f"decision table {label!r} has no `output_columns`.")
    for column in outputs:
        if column.get("name") == MATCHED_RULES_KEY:
            errors.append(
                f"decision table {label!r}: {MATCHED_RULES_KEY!r} is reserved for the rules that fired; "
                "name the output column differently."
            )
    for side, columns in (("input", inputs), ("output", outputs)):
        names = [str(column.get("name") or "") for column in columns]
        duplicates = sorted({name for name in names if name and names.count(name) > 1})
        if duplicates:
            errors.append(
                f"decision table {label!r}: {side} column names used more than once: {', '.join(duplicates)}."
            )
    for index, rule in enumerate(node.get("rules") or []):
        if not isinstance(rule, dict):
            errors.append(f"decision table {label!r}: rules[{index}] is not an object.")
            continue
        rule_label = rule.get("id") or index
        if not rule.get("id"):
            errors.append(f"decision table {label!r}: rules[{index}] has no `id`.")
        # A disabled row is never compiled, so a draft left short of cells loads and runs; only a row that
        # is on has to fit the columns, the line the node itself draws.
        if not rule.get("enabled", True):
            continue
        when, then = rule.get("when") or [], rule.get("then") or []
        if inputs and len(when) != len(inputs):
            errors.append(
                f"decision table {label!r}: rule {rule_label!r} has {len(when)} `when` cells for "
                f"{len(inputs)} input columns. Cells are positional, one per column; an empty cell is null."
            )
        if outputs and len(then) != len(outputs):
            errors.append(
                f"decision table {label!r}: rule {rule_label!r} has {len(then)} `then` cells for "
                f"{len(outputs)} output columns."
            )
    policy = node.get("hit_policy")
    if policy is not None and policy not in TABLE_HIT_POLICIES:
        errors.append(f"decision table {label!r}: hit_policy {policy!r} is not one of {', '.join(TABLE_HIT_POLICIES)}.")
    aggregation = node.get("aggregation")
    if aggregation is not None and aggregation not in TABLE_AGGREGATIONS:
        errors.append(
            f"decision table {label!r}: aggregation {aggregation!r} is not one of {', '.join(TABLE_AGGREGATIONS)}."
        )
    return errors


def check_rules(node, label) -> list:
    errors = []
    inputs = [f for f in (node.get("input_fields") or []) if isinstance(f, dict)]
    if not inputs:
        errors.append(f"rules {label!r} has no `input_fields`; a rule reads the record by the names declared here.")
    taken = set()
    for field in inputs:
        name = str(field.get("name") or "")
        if not IDENTIFIER_RE.match(name):
            errors.append(
                f"rules {label!r}: input name {name!r} is not an identifier (letters, digits and "
                "underscores, not starting with a digit), so no check could read it."
            )
        taken.add(name)
    for derived in node.get("derived_values") or []:
        if not isinstance(derived, dict):
            continue
        name = str(derived.get("name") or "")
        if not IDENTIFIER_RE.match(name):
            errors.append(f"rules {label!r}: derived value name {name!r} is not an identifier.")
        elif name in taken:
            errors.append(
                f"rules {label!r}: derived value {name!r} takes a name an input or another derived value has."
            )
        elif name in RULE_RESERVED_NAMES:
            errors.append(f"rules {label!r}: derived value {name!r} is already the name of a helper.")
        elif name == RESERVED_ROOT:
            errors.append(
                f"rules {label!r}: derived value {name!r} could not be read by a rule: Jinja reserves the name "
                "inside an expression."
            )
        taken.add(name)
        if not str(derived.get("expression") or "").strip():
            errors.append(f"rules {label!r}: derived value {name!r} has no `expression`.")
    ids = []
    for index, rule in enumerate(node.get("rules") or []):
        if not isinstance(rule, dict):
            errors.append(f"rules {label!r}: rules[{index}] is not an object.")
            continue
        rule_label = rule.get("id") or index
        if not rule.get("id"):
            errors.append(f"rules {label!r}: rules[{index}] has no `id`.")
        ids.append(rule.get("id"))
        if rule.get("enabled", True) and not str(rule.get("check") or "").strip():
            errors.append(f"rules {label!r}: rule {rule_label!r} is on but has no `check`.")
        severity = rule.get("severity")
        if severity is not None and severity not in RULE_SEVERITIES:
            errors.append(f"rules {label!r}: rule {rule_label!r} severity {severity!r} is not fail, warn or info.")
        # The engine treats an unrecognized value the same as unset - a logged warning, and the
        # node's own policy applies - so a typo here would keep building silently; validate is
        # the place to be strict about it instead.
        policy = rule.get("on_missing")
        if isinstance(policy, str) and not policy.strip():
            policy = None
        if policy is not None and policy not in RULE_MISSING_POLICIES:
            errors.append(
                f"rules {label!r}: rule {rule_label!r} on_missing {policy!r} is not one of "
                f"{', '.join(RULE_MISSING_POLICIES)}."
            )
    duplicates = sorted({str(i) for i in ids if i and ids.count(i) > 1})
    if duplicates:
        errors.append(f"rules {label!r}: rule ids used more than once: {', '.join(duplicates)}.")
    policy = node.get("on_missing")
    if policy is not None and policy not in RULE_MISSING_POLICIES:
        errors.append(f"rules {label!r}: on_missing {policy!r} is not one of {', '.join(RULE_MISSING_POLICIES)}.")
    return errors


def check_expression(node, label) -> list:
    errors = []
    items = node.get("expressions")
    if not isinstance(items, list) or not items:
        errors.append(f'expression {label!r} has no `expressions` ({{"key", "expression"}} each).')
        return errors
    keys = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"expression {label!r}: expressions[{index}] is not an object.")
            continue
        key = str(item.get("key") or "")
        if not IDENTIFIER_RE.match(key):
            errors.append(f"expression {label!r}: key {key!r} is not an identifier.")
        keys.append(key)
        if not str(item.get("expression") or "").strip():
            errors.append(f"expression {label!r}: key {key!r} has no `expression`.")
    duplicates = sorted({k for k in keys if k and keys.count(k) > 1})
    if duplicates:
        errors.append(f"expression {label!r}: keys used more than once: {', '.join(duplicates)}.")
    return errors


def check_expressions(node, label) -> tuple[list, list]:
    """Parses every Jinja2 expression on a Rules or Expression node: a Rules node's rule `check`,
    `applies_when` and each derived value's `expression`; an Expression node's each
    `expressions[].expression`. A rule's `message` is a template, parsed as one (`_parse_message`).
    Returns (errors, warnings).

    Compiling the text is not enough: Jinja only checks a filter or test used inside a
    conditional expression (`x | lowr if a else b`) at RUN time, letting it stay undefined at
    compile time so the branch not taken never has to resolve it. So this walks the parsed AST
    instead, looking at every `Filter`, `Test` and `Call` node regardless of which branch it sits
    in, plus every `Name` read as a record root.

    A syntax error, an unknown filter, an unknown test and a call of a name no helper has are all
    errors: none of them can ever be right, whatever the record turns out to hold - a record read
    from JSON is never itself callable, so calling a bare name is always meant as a helper. So is
    what the engine refuses when it builds the node, with the engine's own reason: text that is
    more than one expression (`loan.a }} loan.b`), text nested too deeply to read, a read of
    `self`, a private `__` attribute on a Rules node (the Expression node leaves that to its
    sandbox), and a name read as a value and called as a helper, where the helper is one of the
    original ten or one of Jinja's own globals. A root name this node's own declared shape does not
    vouch for is only a WARNING, and is skipped where that shape cannot be known at all - a node
    with neither `input_fields` nor a selector, or one whose `input_transformer` sets a `path`, so
    the record is a sub-tree whose keys nothing here can see.

    A disabled rule (`enabled: false`) is not parsed at all: it never compiles on the node either,
    so a draft left broken while switched off must keep validating clean.
    """
    errors: list = []
    warnings: list = []
    node_type = node.get("type")
    if node_type == RULES_TYPE:
        _check_rules_expressions(node, label, errors, warnings)
    elif node_type == EXPRESSION_TYPE:
        _check_expression_items(node, label, errors, warnings)
    return errors, warnings


def _check_rules_expressions(node, label, errors: list, warnings: list) -> None:
    declared, confident = _declared_roots(node)
    known = (declared | _GLOBAL_NAMES | {AS_OF_KEY}) if confident else None

    derived_items = [d for d in (node.get("derived_values") or []) if isinstance(d, dict)]
    # Every derived value the node declares, valid name or not: a rule's check or applies_when can
    # be checked against all of them regardless (see below), and whether one derived value's
    # expression reaches for another defined after it is knowable from this list alone, whatever
    # the node's own declared inputs look like.
    derived_names = frozenset(name for d in derived_items if IDENTIFIER_RE.match(name := str(d.get("name") or "")))

    computed: set[str] = set()
    for derived in derived_items:
        name = str(derived.get("name") or "?")
        text = str(derived.get("expression") or "")
        if text.strip():
            where = f"rules {label!r}: derived value {name!r}"
            reads_known = (known | computed) if known is not None else None
            not_yet_computed = derived_names - computed
            _parse_and_walk(text, reads_known, not_yet_computed, where, errors, warnings, private=True)
        computed.add(name)

    # Every derived value is computed before any rule runs, so all of them are fair game to a
    # rule's check or applies_when - none is ever "not yet computed" from there.
    rule_known = (known | derived_names) if known is not None else None
    for index, rule in enumerate(node.get("rules") or []):
        if not isinstance(rule, dict) or not rule.get("enabled", True):
            continue
        rule_label = rule.get("id") or index
        for attr in ("check", "applies_when"):
            text = str(rule.get(attr) or "")
            if text.strip():
                where = f"rules {label!r}: rule {rule_label!r} {attr}"
                _parse_and_walk(text, rule_known, frozenset(), where, errors, warnings, private=True)
        message = str(rule.get("message") or "")
        if message.strip():
            _parse_message(message, f"rules {label!r}: rule {rule_label!r} message", errors)


def _check_expression_items(node, label, errors: list, warnings: list) -> None:
    declared, confident = _declared_roots(node)
    known = (declared | _GLOBAL_NAMES) if confident else None
    for index, item in enumerate(node.get("expressions") or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("expression") or "")
        if not text.strip():
            continue
        key = item.get("key") or index
        where = f"expression {label!r}: key {key!r}"
        _parse_and_walk(text, known, frozenset(), where, errors, warnings, private=False)


def _declared_roots(node) -> tuple[frozenset, bool]:
    """The root names this node's own declared shape can vouch for - its `input_fields` and the
    keys of `input_transformer.selector` - and whether that shape is trustworthy enough to warn
    about a name outside it. The second value is False when the node declares neither, or
    `input_transformer.path` is set: the record is then a sub-tree shaped by something this
    checker cannot see, so any name might be legitimate.
    """
    inputs = {
        str(field.get("name"))
        for field in (node.get("input_fields") or [])
        if isinstance(field, dict) and field.get("name")
    }
    transformer = _as_dict(node.get("input_transformer"))
    selector_keys = {str(key) for key in _as_dict(transformer.get("selector"))}
    declared = inputs | selector_keys
    return frozenset(declared), bool(declared) and not transformer.get("path")


# Jinja names a lone `}` this way only from one place (jinja2.lexer's brace-balancing check): an
# open bracket still pending from the author's own text makes the lexer read the wrapper's own
# closing `}}` as a mismatched attempt to close THAT bracket, one `}` at a time - and that branch
# always names what it expected instead. A *bare* "unexpected '}'", with nothing further, is a
# different thing: the author's own text closing something that was never open, which the lexer
# catches with an empty balancing stack and no "expected" to report - left alone here, since it
# names a character the author actually wrote.
_UNCLOSED_BRACKET_RE = re.compile(r"unexpected '\}', expected ")
# Jinja's other way of naming a spot: an absolute character offset into whatever text it parsed -
# here, the WRAPPED text, so "at N" is off by the length of the prefix the author never wrote.
_POSITION_RE = re.compile(r"\bat (\d+)\b")


def _without_wrapper_leak(message: str) -> str:
    """Jinja's own syntax-error message, with anything that only makes sense against
    `_EXPRESSION_PREFIX + text + " }}"` - never the text the author wrote - translated back to
    their own expression. Built from the message itself, two small substitutions, not a hard-coded
    string per error shape: any other message, an ordinary mistake in the author's own text, comes
    back unchanged.
    """
    message = _UNCLOSED_BRACKET_RE.sub("unexpected end of expression, expected ", message)

    def _shift(match: re.Match) -> str:
        position = int(match.group(1)) - len(_EXPRESSION_PREFIX)
        return f"at {position}" if position >= 0 else "at the start of the expression"

    return _POSITION_RE.sub(_shift, message)


def _parse_and_walk(
    text: str,
    known: frozenset | None,
    not_yet_computed: frozenset,
    where: str,
    errors: list,
    warnings: list,
    *,
    private: bool,
) -> None:
    """Parses one expression and walks it (`_walk`). `private` says whether the engine refuses a
    read of a private `__` attribute when it builds the node: the Rules node does, the Expression
    node leaves that to its sandbox."""
    try:
        parsed = _EXPRESSION_ENVIRONMENT.parse(_EXPRESSION_PREFIX + text + " }}")
    except TemplateSyntaxError as e:
        errors.append(f"{where} is not a valid expression: {_without_wrapper_leak(str(e))}")
        return
    except SyntaxError as e:
        # Python's own, from the parser Jinja's lexer reads a number with (`1١.5`).
        errors.append(f"{where} is not a valid expression: {e.msg}")
        return
    except RecursionError:
        errors.append(_too_deep(where))
        return
    if not _is_one_expression(parsed):
        errors.append(
            f"{where} is not a valid expression: chunk after expression; write the expression alone, "
            "without '{{' or '}}'"
        )
        return
    scan = _Scan(known, not_yet_computed, where, private)
    try:
        _walk(parsed, scan)
    except RecursionError:
        errors.append(_too_deep(where))
        return
    if (clash := scan.clash()) is not None:
        scan.errors.append(f"{where} reads {clash!r} as a value and calls it as a helper")
    # Collected locally and de-duplicated before joining the caller's lists: the same typo read
    # twice in one expression should be named once, not once per occurrence.
    errors.extend(dict.fromkeys(scan.errors))
    warnings.extend(dict.fromkeys(scan.warnings))


def _parse_message(text: str, where: str, errors: list) -> None:
    """Parses a rule's message as the template the engine compiles it into, and walks it (`_walk`).

    What the engine refuses to build is an error, as for a check: a syntax error, a filter or a
    test the sandbox does not have, and a read of `self`. Jinja looks a filter or a test up inside
    `{% if %}` only when that branch renders, and the message then comes back as it was written,
    so the walk finds one there too. A name the node does not declare is no error, nor a warning:
    a message prints a value the record lacks as empty text. Nor is a call of a name no helper
    has, since a message may call a macro or a `joiner()` it defines itself, nor a private `__`
    segment, which the engine builds a message over: it prints a key of that name a record holds.
    """
    try:
        parsed = _EXPRESSION_ENVIRONMENT.parse(text)
    except TemplateSyntaxError as e:
        errors.append(f"{where} is not a valid template: {e}")
        return
    except SyntaxError as e:
        errors.append(f"{where} is not a valid template: {e.msg}")
        return
    except RecursionError:
        errors.append(_too_deep(where))
        return
    scan = _Scan(None, frozenset(), where, private=False, template=True)
    try:
        _walk(parsed, scan)
    except RecursionError:
        errors.append(_too_deep(where))
        return
    errors.extend(dict.fromkeys(scan.errors))


def _is_one_expression(parsed) -> bool:
    """Whether the wrapped text parsed as one expression, all the engine's `compile_expression`
    reads: `loan.a }} loan.b` parses as a template holding an expression and then text, where the
    engine stops at the `}}` with "chunk after expression"."""
    body = parsed.body
    return (
        len(body) == 1
        and isinstance(body[0], jinja_nodes.Output)
        and len(body[0].nodes) == 1
        and not isinstance(body[0].nodes[0], jinja_nodes.TemplateData)
    )


def _too_deep(where: str) -> str:
    # Jinja's parser recurses once per level of nesting, and so does the walk here, so text nested a
    # few hundred levels deep exhausts Python's recursion limit rather than parsing.
    return f"{where} is nested too deeply to read; split it into smaller expressions"


class _Scan:
    """One walk of a parsed expression: what it may read (`known`, None where the node's shape
    cannot be known, and `not_yet_computed`), where it sits, whether the engine refuses a private
    attribute there, whether it is a message's template, which may define names of its own to
    call, and what the walk finds - errors, warnings, and for the engine's refusal of a name both
    read and called, the global names the expression calls and the roots of the paths the engine
    reads, in the order it reads them."""

    def __init__(
        self, known: frozenset | None, not_yet_computed: frozenset, where: str, private: bool, template: bool = False
    ) -> None:
        self.known = known
        self.not_yet_computed = not_yet_computed
        self.where = where
        self.private = private
        self.template = template
        self.errors: list = []
        self.warnings: list = []
        self.called: list = []
        self.roots: list = []

    def clash(self) -> str | None:
        """The name the engine refuses to build an expression over for reading it as a value and
        calling it, or None: the first name both read and called, where that is one of the original
        helpers or one of Jinja's own globals. A name the vocabulary added since, `text` say, the
        engine lets through, holding the expression when it runs instead."""
        clash = next((name for name in self.roots if name in self.called), None)
        return clash if clash in RULE_RESERVED_NAMES or clash not in RULE_HELPERS else None

    def read(self, path: str, segments: list) -> None:
        """Judges a path the engine reads as it does when it builds the node: a read of `self`, a
        private segment where the engine refuses one, and a root named like a global, which `clash`
        weighs."""
        if segments[0] == RESERVED_ROOT:
            self.errors.append(
                f"{self.where} reads {path!r}: Jinja reserves the name 'self' inside an expression, so a "
                "top-level key of that name cannot be read; nest it inside a record or rename the input"
            )
        if self.private and any(isinstance(segment, str) and segment.startswith("__") for segment in segments):
            self.errors.append(f"{self.where} reads a private attribute ({path})")
        if segments[0] in _GLOBAL_NAMES:
            self.roots.append(segments[0])


def _path_of(node) -> tuple[str, list] | None:
    """The path a name with attributes and constant keys after it reads, written as the engine
    writes it (`loan.a`, `docs['Flood.Cert']`, `items[0]`), with its segments; None for any other
    node. A copy of dynamiq.nodes.operators.rules._path_of, whose paths the engine's refusals judge."""
    if isinstance(node, jinja_nodes.Name):
        return node.name, [node.name]
    if isinstance(node, jinja_nodes.Getattr):
        base = _path_of(node.node)
        return (f"{base[0]}.{node.attr}", [*base[1], node.attr]) if base else None
    if isinstance(node, jinja_nodes.Getitem) and isinstance(node.arg, jinja_nodes.Const):
        base = _path_of(node.node)
        key = node.arg.value
        if base is None:
            return None
        if isinstance(key, str):
            if IDENTIFIER_RE.match(key):
                return f"{base[0]}.{key}", [*base[1], key]
            escaped = key.replace("\\", "\\\\").replace("'", "\\'")
            return f"{base[0]}['{escaped}']", [*base[1], key]
        if isinstance(key, int) and not isinstance(key, bool):
            return f"{base[0]}[{key}]", [*base[1], key]
    return None


def _walk(node, scan: _Scan, counted: bool = True) -> None:
    """Visits every node of a parsed expression, whichever branch of a conditional it sits in -
    see `check_expressions` for why compiling alone would miss a filter or test used inside one.

    A path, a name with the attributes and constant keys after it, is judged whole, as the engine
    reads it. `counted` is False inside what the engine reads nothing of: what a test that only asks
    about its value is given besides (`x is sameas y`), and a keyword argument of `default`. A name
    there is still checked against the node's shape, but refuses no build."""
    exclude: tuple[str, ...] = ()
    # The fields of this node the engine does not read.
    uncounted: tuple[str, ...] = ()
    if isinstance(node, jinja_nodes.Filter):
        if node.name not in _FILTER_NAMES:
            scan.errors.append(_unknown(scan.where, "filter", node.name, _FILTER_NAMES))
        if node.name == "default":
            uncounted = ("kwargs", "dyn_args", "dyn_kwargs")
    elif isinstance(node, jinja_nodes.Test):
        if node.name not in _TEST_NAMES:
            scan.errors.append(_unknown(scan.where, "test", node.name, _TEST_NAMES))
        if node.name in _EXEMPT_TESTS:
            uncounted = ("args", "kwargs", "dyn_args", "dyn_kwargs")
    elif isinstance(node, jinja_nodes.Call) and isinstance(node.node, jinja_nodes.Name):
        name = node.node.name
        if name in _GLOBAL_NAMES:
            if counted:
                scan.called.append(name)
        else:
            # A bare name called like a function is always meant as a helper: a record read from JSON
            # never holds anything callable, so this is an error rather than merely an unknown root.
            # A message's template may call a name it defines itself, a macro say.
            if not scan.template:
                scan.errors.append(_unknown(scan.where, "helper", name, _GLOBAL_NAMES))
            # The engine reads the name it calls as a member of the record all the same.
            if counted:
                scan.read(name, [name])
        exclude = ("node",)  # the callee is judged above, not walked again as a root read below
    elif isinstance(node, jinja_nodes.Call) and isinstance(node.node, jinja_nodes.Getattr):
        # A method call reads the object it is called on, never a member named like the method:
        # `loan.get('rate')` reads `loan`.
        _walk(node.node.node, scan, counted)
        exclude = ("node",)
    elif (read := _path_of(node)) is not None:
        path, segments = read
        if counted:
            scan.read(path, segments)
        root = segments[0]
        if counted and root == RESERVED_ROOT:
            pass  # refused above; a warning about an undeclared name would only repeat it
        elif root in scan.not_yet_computed:
            scan.warnings.append(f"{scan.where} reads {root!r}, which is computed after it.")
        elif scan.known is not None and root not in scan.known:
            scan.warnings.append(_unknown_root(scan.where, root, scan.known))
        return
    for field, value in node.iter_fields(exclude=exclude):
        for child in value if isinstance(value, list) else [value]:
            if isinstance(child, jinja_nodes.Node):
                _walk(child, scan, counted and field not in uncounted)


def _hint(name: str, candidates) -> str:
    """A " Did you mean 'lower'?" suffix, or '' when nothing in `candidates` is close enough to `name`."""
    matches = difflib.get_close_matches(name, candidates, n=1)
    return f" Did you mean {matches[0]!r}?" if matches else ""


def _unknown(where: str, kind: str, name: str, candidates: frozenset) -> str:
    return f"{where}: unknown {kind} {name!r}.{_hint(name, candidates)}"


def _unknown_root(where: str, name: str, known: frozenset) -> str:
    return f"{where} reads {name!r}, which this node does not declare.{_hint(name, known)}"


def check_judgement(node, label) -> list:
    errors = []
    judge = node.get("judge")
    judge_type = str(judge.get("type") or "") if isinstance(judge, dict) else ""
    if judge is None:
        errors.append(
            f"judgement {label!r} has no `judge`. It needs the node that answers: a "
            f"{SYSTEM_ONE_TYPE!r}, an LLM or an agent."
        )
    elif not (judge_type == SYSTEM_ONE_TYPE or judge_type.startswith(JUDGE_TYPE_PREFIXES)):
        errors.append(
            f"judgement {label!r}: `judge` must be a System One, an LLM or an agent node object, "
            f"got {judge_type or type(judge).__name__!r}."
        )
    # The judge is loaded as a node of its own, so it needs what that class requires - the
    # reason an agent's `llm` is checked here too.
    elif judge_type == SYSTEM_ONE_TYPE:
        errors.extend(
            connection_requirement(
                judge, f"judgement {label!r}: judge", "A System One judge carries a TypeSafe `connection`."
            )
        )
    elif judge_type.startswith("dynamiq.nodes.agents."):
        if isinstance(judge.get("llm"), dict):
            errors.extend(llm_requirements(judge["llm"], f"judgement {label!r}: judge.llm"))
        else:
            errors.append(f"judgement {label!r}: the agent judge has no `llm` object.")
    else:
        errors.extend(llm_requirements(judge, f"judgement {label!r}: judge"))
    names = []
    for index, question in enumerate(node.get("questions") or []):
        if not isinstance(question, dict):
            errors.append(f"judgement {label!r}: questions[{index}] is not an object.")
            continue
        name = str(question.get("name") or "")
        where = f"judgement {label!r}: question {name or index!r}"
        if not IDENTIFIER_RE.match(name):
            errors.append(f"{where} has a name that is not an identifier, so no node could read its answer.")
        names.append(name)
        kind = question.get("type") or "noul"
        if kind not in QUESTION_TYPES:
            errors.append(f"{where}: type {kind!r} is not one of {', '.join(QUESTION_TYPES)}.")
        if not str(question.get("instructions") or "").strip():
            errors.append(f"{where} has no `instructions`.")
        if kind in ("choice", "score"):
            limit, what = (MAX_CHOICE_OPTIONS, "option") if kind == "choice" else (MAX_SCORE_LEVELS, "level")
            a_what = f"{'an' if what == 'option' else 'a'} {what}"
            written = question.get("options") if isinstance(question.get("options"), list) else []
            options = [o for o in written if isinstance(o, dict)]
            option_names = [str(o.get("name") or "").strip() for o in options]
            # Counting what survived the filter would let a mixed list through and fail at load.
            if not 2 <= len(written) <= limit:
                errors.append(f"{where} needs between 2 and {limit} {what}s, got {len(written)}.")
            if malformed := len(written) - len(options):
                errors.append(
                    f"{where} has {a_what} that is not an object ({malformed} of {len(written)}); "
                    f"each one needs a `name`."
                )
            if any(not option_name for option_name in option_names):
                errors.append(f"{where} has {a_what} without a name.")
            if len(set(option_names)) != len(option_names):
                errors.append(f"{where} names {a_what} twice.")
    duplicates = sorted({n for n in names if n and names.count(n) > 1})
    if duplicates:
        errors.append(f"judgement {label!r}: question names used more than once: {', '.join(duplicates)}.")
    for key in ("noul_threshold", "min_confidence"):
        value = node.get(key)
        if value is not None and not (_is_number(value) and 0 <= value <= 1):
            errors.append(f"judgement {label!r}: {key} {value!r} is not a number between 0 and 1.")
    mode = node.get("confidence_mode")
    if mode is not None and mode not in CONFIDENCE_MODES:
        errors.append(f"judgement {label!r}: confidence_mode {mode!r} is not one of {', '.join(CONFIDENCE_MODES)}.")
    samples = node.get("samples")
    if samples is not None and not (_is_number(samples) and samples == int(samples) and 1 <= samples <= MAX_SAMPLES):
        errors.append(f"judgement {label!r}: samples {samples!r} is not a whole number between 1 and {MAX_SAMPLES}.")
    if mode == "sampling":
        if judge_type == SYSTEM_ONE_TYPE:
            errors.append(
                f"judgement {label!r}: sampling needs an LLM or agent judge; "
                "a System One connection returns calibrated probabilities in one call."
            )
        # An omitted `samples` is the node's default of 1, which sampling refuses on load.
        elif _is_number(effective := DEFAULT_SAMPLES if samples is None else samples) and effective < 2:
            errors.append(
                f"judgement {label!r}: sampling needs at least 2 samples, "
                f"and `samples` is {'unset, so it defaults to 1' if samples is None else effective!r}."
            )
    return errors


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_sub_workflow(node, label) -> list:
    errors = []
    if not UUID_RE.match(str(node.get("workflow_id") or "")):
        errors.append(
            f"sub-workflow {label!r}: workflow_id {node.get('workflow_id')!r} is not a workflow UUID. "
            "Run `dynamiq workflow list`; the workflow must be in the same project and have a released version."
        )
    version = node.get("workflow_version_id")
    if version is not None and not UUID_RE.match(str(version)):
        errors.append(
            f"sub-workflow {label!r}: workflow_version_id {version!r} is not a version UUID. "
            "Omit it to run the newest released version, or pick one from `dynamiq workflow versions <id>`."
        )
    if node.get("flow") is not None:
        errors.append(
            f"sub-workflow {label!r} carries `flow`. The platform inlines the chosen version itself; leave it out."
        )
    return errors


# What `workflow save` accepts. Shorter than the SDK's list: InMemory and SQLite run locally
# and are rejected on save.
MEMORY_BACKENDS = {
    "dynamiq.memory.backends.Dynamiq",
    "dynamiq.memory.backends.PostgreSQL",
    "dynamiq.memory.backends.Pinecone",
    "dynamiq.memory.backends.Qdrant",
    "dynamiq.memory.backends.Weaviate",
    "dynamiq.memory.backends.DynamoDB",
}

# Declared by dynamiq/nodes/tools/pipedream.py. `props` is a component schema, not a field.
PIPEDREAM_FIELDS = {
    "id", "name", "type", "action_id", "external_user_id", "input_props", "configurable_props",
    "dynamic_props_id", "stash_id", "connection", "timeout",
    "is_optimized_for_agents", "input_transformer", "output_transformer", "streaming",
    "error_handling", "approval", "description",
    # also legal when the same object sits in the DAG as its own node rather than in tools[]
    "depends", "schema", "caching", "flows",
}


def find_nested(obj, key, path="tool"):
    """Where a key actually lives, when it is not where it should be.

    Only BELOW the top level. Matching at the top too reported a key that was already in the
    right place as misplaced - "`configurable_props` exists but at tool.configurable_props" -
    an instruction nobody can follow, in place of the message that says what is actually wrong.
    """
    if not isinstance(obj, dict):
        return None
    for k, v in obj.items():
        found = _find_below(v, key, f"{path}.{k}")
        if found:
            return found
    return None


def _find_below(obj, key, path):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return f"{path}.{k}"
            found = _find_below(v, key, f"{path}.{k}")
            if found:
                return found
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found = _find_below(item, key, f"{path}[{i}]")
            if found:
                return found
    return None


def requirement_problems(value, where):
    """A `{"$type": "requirement", "$id": ...}` placeholder is a legal value ANYWHERE.

    It is how a multi-user workflow lets each caller bring their own account, so a checker
    that insists on a literal `apn_...` would reject exactly the flows that are done right.
    Returns None when `value` is not a placeholder, otherwise a (possibly empty) problem list.
    """
    if not (isinstance(value, dict) and value.get("$type") == "requirement"):
        return None
    problems = []
    if not str(value.get("$id") or "").strip():
        problems.append(f"{where}: a requirement placeholder needs `$id` - the id returned by "
                        "POST /v1/workflows/<id>/requirements.")
    path = value.get("value_path")
    if path is not None and (not isinstance(path, str) or not path.startswith("$.")):
        problems.append(f"{where}: requirement value_path {path!r} must be a JSONPath like "
                        '"$.account_id". It has to match exactly one value.')
    return problems


def connection_requirement(node, where, hint) -> list:
    """A provider node carries its own `connection`, which the API resolves by id."""
    from_requirement = requirement_problems(node.get("connection"), f"{where}: connection")
    if from_requirement is not None:
        return from_requirement
    if not UUID_RE.match(str(node.get("connection") or "")):
        return [
            f"{where}: connection {node.get('connection')!r} is not a connection UUID. "
            f"{hint} Run `dynamiq connection list`."
        ]
    return []


def llm_requirements(llm, where) -> list:
    """`connection` and `model` have no defaults on an LLM node, so one missing either is refused
    on load. Applies to a node in the DAG and to a judge nested inside a Judgement alike."""
    problems = connection_requirement(llm, where, "An LLM node carries its own `connection`.")
    if not str(llm.get("model") or "").strip():
        problems.append(f'{where}: an LLM node needs `model` (e.g. "gpt-4o").')
    return problems


def check_pipedream(tool, where):
    """-> (problems, advisory). An unknown key is dropped by the API rather than rejected, so
    it is worth reporting and never worth blocking a save over."""
    problems: list = []
    advisory: list = []

    unknown = sorted(set(tool) - PIPEDREAM_FIELDS)
    if unknown:
        detail = ""
        if "props" in unknown:
            detail = (
                " `props` is a component schema, not a field on the tool. Rebuild the tool with "
                "`pipedream_node <app> <key> --out tool.json`: EVERY prop is declared in "
                '`input_props.configurableProps` as [{"name": "title", "type_": "string"}, ...], '
                "and `configurable_props` holds only the values you pin (the account, and any "
                "fixed target the user named). They overlap on purpose - a pinned prop stays in "
                "input_props and becomes an overridable default."
            )
        advisory.append(f"{where}: unknown field(s) {', '.join(unknown)} - they are ignored." + detail)

    if not tool.get("action_id"):
        problems.append(
            f'{where}: a Pipedream tool needs `action_id` (e.g. "notion-create-page"). '
            "List the real keys with `dynamiq integration components <app_slug>`."
        )
    external_user = tool.get("external_user_id")
    from_requirement = requirement_problems(external_user, f"{where}: external_user_id")
    if from_requirement is not None:
        problems.extend(from_requirement)
    elif not external_user:
        problems.append(
            f"{where}: needs `external_user_id` - the project id, because Pipedream accounts are "
            "bound to the project rather than to a person. `workflow save` and `create` fill it in "
            "from the current project; set it yourself with the id from `dynamiq config show` if you "
            "are building the flow for somewhere else."
        )

    # A hand-written schema is either missing or carries `type` where the SDK reads `type_`,
    # in which case every prop is ignored and the agent gets a tool it cannot call.
    declared = tool.get("input_props")
    declared_props = declared.get("configurableProps") if isinstance(declared, dict) else None
    if not isinstance(declared_props, list) or not declared_props:
        problems.append(
            f"{where}: needs `input_props.configurableProps` - the FULL prop list from the "
            f"component, not a subset. Build it with `pipedream_node <app> {tool.get('action_id') or '<key>'} "
            "--out tool.json`, which fetches the real record instead of reconstructing it."
        )
    else:
        missing = [
            prop.get("name")
            for prop in declared_props
            if isinstance(prop, dict) and "type_" not in prop and "type" not in prop
        ]
        if missing:
            problems.append(
                f"{where}: input_props prop(s) {', '.join(map(str, missing))} declare no type. "
                'Each needs `type_` (or `type`, which the SDK renames): {"name": "title", '
                '"type_": "string"}.'
            )

    props = tool.get("configurable_props")
    if not isinstance(props, dict) or not props:
        nested = find_nested(tool, "configurable_props")
        if nested:
            problems.append(
                f"{where}: `configurable_props` exists but at {nested} - it belongs at the TOP level "
                "of the tool object, a sibling of `action_id`, not nested inside another key. "
                "Move it up one level."
            )
        else:
            problems.append(
                f"{where}: needs `configurable_props` binding the account, e.g. "
                '{"notion": {"authProvisionId": "apn_..."}}. Run `dynamiq integration accounts`. '
                f"Present fields: {', '.join(sorted(tool)) or '(none)'}."
            )
        return problems, advisory

    bound = False
    for key, value in props.items():
        if isinstance(value, dict) and "authProvisionId" in value:
            bound = True
            apn = value["authProvisionId"]
            from_requirement = requirement_problems(apn, f"{where}: authProvisionId")
            if from_requirement is not None:
                problems.extend(from_requirement)
            elif not isinstance(apn, str) or not apn.startswith("apn_"):
                problems.append(
                    f"{where}: authProvisionId {apn!r} is not a real connected account. Use the "
                    "`account_id` (apn_...) from `dynamiq integration accounts`, or a requirement "
                    'placeholder {"$type": "requirement", "$id": "...", "value_path": "$.account_id"} '
                    "so each caller brings their own."
                )
    if not bound:
        problems.append(
            f'{where}: no account binding in configurable_props. Add {{"<app>": '
            '{"authProvisionId": "apn_..."}} from `dynamiq integration accounts`.'
        )
    return problems, advisory
