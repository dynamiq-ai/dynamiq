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
"""
from __future__ import annotations

import re

NODE_ID_RE = re.compile(r"^[a-z0-9]([a-z0-9]|-[a-z0-9])*$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

INPUT_TYPE = "dynamiq.nodes.utils.Input"
OUTPUT_TYPE = "dynamiq.nodes.utils.Output"

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
            transformer = {}
        selector = transformer.get("selector") or {}
        if not isinstance(selector, dict):
            errors.append(f"node {label!r}: input_transformer.selector must be an object of field -> JSONPath.")
            selector = {}
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
            elif isinstance(response_format, dict) and not response_format.get("properties"):
                errors.append(
                    f"agent {label!r}: `response_format` has no `properties`, so it constrains nothing. "
                    'Use e.g. {"type": "object", "properties": {"answer": {"type": "string"}}, '
                    '"required": ["answer"]}.'
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
            from_requirement = requirement_problems(node.get("connection"), f"node {label!r}: connection")
            if from_requirement is not None:
                errors.extend(from_requirement)
            elif not UUID_RE.match(str(node.get("connection") or "")):
                errors.append(
                    f"node {label!r}: connection {node.get('connection')!r} is not a connection UUID. "
                    "An LLM node carries its own `connection`. Run `dynamiq connection list`."
                )
            if not str(node.get("model") or "").strip():
                errors.append(
                    f"node {label!r}: an LLM node needs `model` (e.g. \"gpt-4o\")."
                )

        # A Pipedream node placed in the DAG is validated exactly like one inside an agent.
        if node_type == "dynamiq.nodes.tools.Pipedream":
            node_errors, node_advisory = check_pipedream(node, f"node {label!r}")
            errors.extend(node_errors)
            warnings.extend(node_advisory)

        if node_type.startswith("dynamiq.nodes.tools."):
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

    for path, text in walk_strings(flow):
        if path.rsplit(".", 1)[-1] in PROSE_KEYS or len(text) > 200:
            continue
        if looks_like_placeholder(text):
            errors.append(f"{path} is still the placeholder {text!r} - replace it with a real value.")

    return errors, warnings


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
