import json
import uuid

import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.config import Settings

workflow = click.Group(name="workflow", help="Manage workflows: create, save the DAG, test, release")


def read_json_arg(value: str):
    """Accept inline JSON or @path/to/file.json."""
    try:
        if value.startswith("@"):
            with open(value[1:]) as f:
                return json.load(f)
        return json.loads(value)
    except FileNotFoundError:
        raise click.ClickException(f"file not found: {value[1:]}")
    except json.JSONDecodeError as e:
        raise click.ClickException(f"invalid JSON in {value[:60]}: {e}")


def echo_response(response, success_message: str | None = None) -> None:
    """Print the JSON body; non-200 exits with the body as the error."""
    body = response.text.strip()
    if response.status_code != 200:
        raise click.ClickException(f"HTTP {response.status_code}: {body[:2000]}")
    if success_message:
        click.echo(success_message)
    if body:
        click.echo(json.dumps(response.json(), indent=2, ensure_ascii=False))


def pagination_options(fn):
    """--page / --page-size / --all for a list command.

    The API returns 25 items per page by default (max 500), so a bare list silently shows
    only the first page - `--all` walks every page and returns the complete set.
    """
    fn = click.option("--all", "fetch_all", is_flag=True, help="Fetch every page, not just the first 25.")(fn)
    fn = click.option(
        "--compact",
        is_flag=True,
        help="Print only id/name/status plus a total count, instead of full objects.",
    )(fn)
    fn = click.option("--page-size", type=int, default=None, help="Items per page (max 500; API default 25).")(fn)
    fn = click.option("--page", type=int, default=None, help="Page number (API default 1).")(fn)
    return fn


COMPACT_FIELDS = ("id", "name", "status", "type", "app_slug", "account_id", "external_user_id")


def compact_items(items: list) -> list:
    """Keep only the identifying fields - list payloads are mostly timestamps and avatars."""
    out = []
    for item in items:
        if not isinstance(item, dict):
            out.append(item)
            continue
        row = {k: item[k] for k in COMPACT_FIELDS if k in item}
        out.append(row or item)
    return out


def echo_list(
    api: ApiClient,
    path: str,
    params: dict | None = None,
    page: int | None = None,
    page_size: int | None = None,
    fetch_all: bool = False,
    compact: bool = False,
) -> None:
    """Print a list endpoint's items, optionally walking every page."""
    params = dict(params or {})

    if not fetch_all:
        if page:
            params["page"] = page
        if page_size:
            params["page_size"] = page_size
        response = api.get(path, params=params or None)
        if response.status_code != 200:
            raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
        body = response.json()
        pagination = body.get("pagination") or {}
        total = pagination.get("total_count")
        items = body.get("data") or []
        if total is not None and len(items) < total:
            click.echo(f"note: showing {len(items)} of {total}. Use --all to fetch every page.", err=True)
        if compact:
            click.echo(
                json.dumps(
                    {"count": len(items), "total_count": total, "items": compact_items(items)},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            click.echo(json.dumps(body, indent=2, ensure_ascii=False))
        return

    items: list = []
    current = 1
    while True:
        response = api.get(path, params={**params, "page": current, "page_size": page_size or 500})
        if response.status_code != 200:
            raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
        body = response.json()
        batch = body.get("data") or []
        items.extend(batch)
        pagination = body.get("pagination") or {}
        page_count = pagination.get("page_count")
        total = pagination.get("total_count")
        if not batch:
            break
        if page_count and current >= page_count:
            break
        if total is not None and len(items) >= total:
            break
        current += 1

    payload = {"count": len(items), "items": compact_items(items)} if compact else {"count": len(items), "data": items}
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def require_project(settings: Settings) -> str:
    """Project id from DYNAMIQ_PROJECT_ID / `dynamiq project set`; scoped endpoints need it."""
    if not settings.project_id:
        raise click.ClickException("No project set. Export DYNAMIQ_PROJECT_ID or run `dynamiq project set --id <id>`.")
    return settings.project_id


def check_connection(value, where: str) -> None:
    """A `connection` must be a real connection UUID, not a placeholder."""
    if value is None:
        return
    if not isinstance(value, str) or not _is_uuid(value):
        raise click.ClickException(
            f"connection {value!r} on {where} is not a connection UUID. "
            "Run `dynamiq connection list` and use the `id` of the connection you want."
        )


def normalize_flow(flow):
    """Make a hand-written flow acceptable to the API without changing its meaning.

    The API is strict about a handful of shapes that are easy to get wrong by hand, and
    it either rejects the body outright or (worse) drops unknown keys silently. Each fix
    below is unambiguous, and every change is reported on stderr so nothing is secret:

    * missing / non-UUID `flow.id`            -> a fresh UUID
    * `depends: "input"` or `["input"]`       -> `[{"node": "input"}]`
    * node-level `selector`                   -> `input_transformer.selector`
    * missing node `name`                     -> the node id
    * `llm` / `tools[]` without an `id`       -> a fresh UUID each
    """
    if not isinstance(flow, dict):
        return flow

    def note(message: str) -> None:
        click.echo(f"note: {message}", err=True)

    flow_id = flow.get("id")
    if not isinstance(flow_id, str) or not _is_uuid(flow_id):
        flow["id"] = str(uuid.uuid4())
        note(f"flow.id was {flow_id!r}; generated {flow['id']}")

    for index, node in enumerate(flow.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        label = node.get("id", index)

        if not node.get("name") and node.get("id"):
            node["name"] = node["id"]
            note(f"node {label!r} had no name; used its id")

        check_connection(node.get("connection"), f"node {label!r}")
        for tool in node.get("tools") or []:
            if isinstance(tool, dict):
                check_connection(tool.get("connection"), f"tool {tool.get('type', '?')} on node {label!r}")
        llm = node.get("llm") if isinstance(node.get("llm"), dict) else None
        if llm:
            check_connection(llm.get("connection"), f"llm on node {label!r}")

        depends = node.get("depends")
        if isinstance(depends, str):
            depends = [depends]
        if isinstance(depends, list):
            fixed, changed = [], False
            for dependency in depends:
                if isinstance(dependency, str):
                    fixed.append({"node": dependency})
                    changed = True
                else:
                    fixed.append(dependency)
            if changed or not isinstance(node.get("depends"), list):
                node["depends"] = fixed
                note(f'node {label!r}: rewrote depends as [{{"node": "<id>"}}]')

        if "selector" in node:
            selector = node.pop("selector")
            transformer = node.setdefault("input_transformer", {})
            transformer.setdefault("selector", selector)
            note(f"node {label!r}: moved top-level 'selector' into input_transformer (the API ignores it otherwise)")

        llm = node.get("llm")
        if isinstance(llm, dict) and not llm.get("id"):
            llm["id"] = str(uuid.uuid4())
            note(f"node {label!r}: generated an id for its llm sub-object")

        for tool in node.get("tools") or []:
            if isinstance(tool, dict) and not tool.get("id"):
                tool["id"] = str(uuid.uuid4())
                note(f"node {label!r}: generated an id for tool {tool.get('type', '?')}")

    warn_tool_chained_after_agent(flow)
    return flow


def warn_tool_chained_after_agent(flow: dict) -> None:
    """Catch the classic mistake: a tool wired as its own node after an agent.

    An agent's tools belong INSIDE the agent node's `tools[]` array - that is what lets
    the agent call them. A `dynamiq.nodes.tools.*` node that merely depends on an agent
    is a fixed pipeline step: it receives the agent's finished output as its input and
    the agent can never invoke it. Both are valid DAGs, so this is a warning, not an error.
    """
    agents = {
        node.get("id")
        for node in flow.get("nodes") or []
        if isinstance(node, dict) and node.get("type", "").startswith("dynamiq.nodes.agents.")
    }
    if not agents:
        return

    for node in flow.get("nodes") or []:
        if not isinstance(node, dict) or not node.get("type", "").startswith("dynamiq.nodes.tools."):
            continue
        depends_on_agent = [
            d.get("node") for d in node.get("depends") or [] if isinstance(d, dict) and d.get("node") in agents
        ]
        if depends_on_agent:
            click.echo(
                f"warning: node {node.get('id')!r} ({node.get('type')}) is a standalone step that runs AFTER "
                f"agent {depends_on_agent[0]!r} and receives its finished output. "
                f'If you meant to give the agent this tool, move it into that agent\'s "tools" array instead.',
                err=True,
            )


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def starter_flow() -> dict:
    """Smallest flow the API accepts: one Input node. `flow.nodes` may not be empty."""
    return {
        "id": str(uuid.uuid4()),
        "nodes": [
            {
                "id": "input",
                "name": "input",
                "type": "dynamiq.nodes.utils.Input",
                "schema": {"type": "object", "properties": {}},
            }
        ],
    }


def flow_ui_for(flow: dict) -> dict:
    """Canvas entries for a flow's nodes, laid out left to right.

    `flow_ui` is required by create/save/release, so generate a usable one whenever the
    caller does not supply their own.
    """
    ui_ids = {}
    nodes = []
    for i, node in enumerate(flow.get("nodes", [])):
        ui_id = str(uuid.uuid4())
        ui_ids[node.get("id")] = ui_id
        nodes.append(
            {
                "id": ui_id,
                "data": {
                    "metadata": {
                        "id": node.get("id"),
                        "name": node.get("name"),
                        "type": node.get("type"),
                        "depends": node.get("depends", []),
                    }
                },
                "position": {"x": i * 378, "y": 245.5},
                "width": 150,
                "height": 60,
                "node_name": node.get("id"),
            }
        )

    edges = []
    for node in flow.get("nodes", []):
        for dep in node.get("depends", []) or []:
            source = ui_ids.get(dep.get("node") if isinstance(dep, dict) else dep)
            target = ui_ids.get(node.get("id"))
            if source and target:
                edges.append(
                    {
                        "id": str(uuid.uuid4()),
                        "type": "smoothstep",
                        "source": source,
                        "target": target,
                        "source_handle": "source",
                        "target_handle": "target",
                    }
                )
    return {"nodes": nodes, "edges": edges}


@workflow.command("list")
@pagination_options
@with_api_and_settings
def list_workflows(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List workflows in the current project.

    The API returns 25 per page; pass --all to get every workflow.
    """
    echo_list(
        api,
        "/v1/workflows",
        {"project_id": require_project(settings)},
        page=page,
        page_size=page_size,
        fetch_all=fetch_all,
        compact=compact,
    )


@workflow.command("get")
@click.argument("workflow_id")
@with_api_and_settings
def get_workflow(*, api: ApiClient, settings: Settings, workflow_id: str):
    """Fetch one workflow, including its full flow and flow_ui - the shape to copy."""
    echo_response(api.get(f"/v1/workflows/{workflow_id}"))


@workflow.command("create")
@click.argument("payload")
@with_api_and_settings
def create_workflow(*, api: ApiClient, settings: Settings, payload: str):
    """Create a workflow. Required by the API: name, project_id, flow, flow_ui.

    `{"name": "my-workflow"}` is enough - project_id comes from the current project, and
    a starter flow (one Input node; `flow.nodes` may NOT be empty) plus a matching flow_ui
    are generated. Pass your own `flow` to create it fully formed. Name must be lowercase
    letters/digits/hyphens.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    body.setdefault("flow", starter_flow())
    body["flow"] = normalize_flow(body["flow"])
    body.setdefault("flow_ui", flow_ui_for(body["flow"]))
    echo_response(api.post("/v1/workflows", json=body))


@workflow.command("save")
@click.argument("workflow_id")
@click.argument("flow")
@click.option("--flow-ui", default=None, help="Canvas layout JSON (inline or @file); generated from FLOW if omitted.")
@click.option("--allow-starter", is_flag=True, help="Permit saving a flow that has only the starter Input node.")
@with_api_and_settings
def save_workflow(
    *, api: ApiClient, settings: Settings, workflow_id: str, flow: str, flow_ui: str | None, allow_starter: bool
):
    """Save the workflow DAG. This is the ONLY endpoint that persists nodes -
    PUT /v1/workflows/{id} updates name/description only and silently ignores a flow.

    The API requires BOTH flow and flow_ui; a canvas layout is generated from the flow
    when --flow-ui is omitted. Always `workflow get` afterwards to verify it persisted.
    """
    flow_body = normalize_flow(read_json_arg(flow))
    nodes = flow_body.get("nodes") or []
    if not allow_starter and len(nodes) == 1 and nodes[0].get("type") == "dynamiq.nodes.utils.Input":
        raise click.ClickException(
            "This flow contains only the starter Input node, so saving it would leave the workflow empty. "
            "Build the real DAG first (see `dynamiq workflow scaffold`), or pass --allow-starter if you "
            "really mean to save an input-only flow."
        )
    body = {
        "flow": flow_body,
        "flow_ui": read_json_arg(flow_ui) if flow_ui else flow_ui_for(flow_body),
    }
    echo_response(api.post(f"/v1/workflows/{workflow_id}/save", json=body))


@workflow.command("test")
@click.argument("flow")
@click.argument("input_data")
@click.option("--dry-run", is_flag=True, help="Validate and plan without executing nodes.")
@click.option("--last-node-output", is_flag=True, help="Return only the last node's output.")
@with_api_and_settings
def test_workflow(
    *, api: ApiClient, settings: Settings, flow: str, input_data: str, dry_run: bool, last_node_output: bool
):
    """Dry-run a flow with the given input, without saving or releasing.

    This endpoint takes a FORM (not a JSON body): `flow` and `input` are sent as
    JSON-encoded strings. FLOW/INPUT_DATA are inline JSON or @file. No project_id needed.
    """
    form = {
        "flow": json.dumps(normalize_flow(read_json_arg(flow))),
        "input": json.dumps(read_json_arg(input_data)),
        "stream": "false",
    }
    if dry_run:
        form["dry_run"] = "true"
    if last_node_output:
        form["last_node_output"] = "true"
    echo_response(api.post("/v1/workflows/test", data=form))


@workflow.command("release")
@click.argument("workflow_id")
@click.option("--name", default=None, help="New name for the released version (defaults to the current name).")
@click.option("--flow", default=None, help="Flow JSON to release (inline or @file); defaults to the saved flow.")
@click.option("--flow-ui", default=None, help="Canvas layout JSON; defaults to the saved flow_ui.")
@with_api_and_settings
def release_workflow(
    *, api: ApiClient, settings: Settings, workflow_id: str, name: str | None, flow: str | None, flow_ui: str | None
):
    """Release a new version. The API requires name, flow and flow_ui in the body, so the
    workflow's current values are fetched and re-sent unless overridden by the options.
    """
    current = api.get(f"/v1/workflows/{workflow_id}")
    if current.status_code != 200:
        raise click.ClickException(f"HTTP {current.status_code}: {current.text.strip()[:2000]}")
    data = current.json().get("data", {})

    flow_body = normalize_flow(read_json_arg(flow)) if flow else data.get("flow")
    if not flow_body:
        raise click.ClickException("Workflow has no saved flow to release. Run `workflow save` first.")
    body = {
        "name": name or data.get("name"),
        "flow": flow_body,
        "flow_ui": read_json_arg(flow_ui) if flow_ui else (data.get("flow_ui") or flow_ui_for(flow_body)),
    }
    echo_response(api.post(f"/v1/workflows/{workflow_id}/release", json=body))


@workflow.command("versions")
@click.argument("workflow_id")
@with_api_and_settings
def list_workflow_versions(*, api: ApiClient, settings: Settings, workflow_id: str):
    """List released versions of a workflow (newest first)."""
    echo_response(api.get(f"/v1/workflows/{workflow_id}/versions"))


@workflow.command("scaffold")
@click.option("--llm-connection", required=True, help="Connection id from `dynamiq connection list`.")
@click.option("--model", default="gpt-4o", show_default=True, help="LLM model name.")
@click.option("--role", default="You answer the user's question clearly and concisely.", help="Agent role/prompt.")
@click.option("--output-field", default="answer", show_default=True, help="Output node field name.")
@click.option("--agent-id", "agent_id", default="agent", show_default=True, help="Slug for the agent node.")
@click.option(
    "--tool",
    "tools",
    multiple=True,
    metavar="TYPE:CONNECTION_ID",
    help="Attach a tool, repeatable, e.g. dynamiq.nodes.tools.ExaTool:<connection_id>.",
)
@click.option("--out", "out_path", default=None, help="Write the flow JSON to this file instead of stdout.")
def scaffold_flow(
    llm_connection: str,
    model: str,
    role: str,
    output_field: str,
    agent_id: str,
    tools: tuple[str, ...],
    out_path: str | None,
):
    """Emit a valid Input -> Agent -> Output flow, ready for `workflow save`.

    Prefer `workflow create-agent`, which does create + save + verify in one step. Use
    this when you want to inspect or hand-edit the flow before saving it.
    """
    flow = build_agent_flow(
        llm_connection=llm_connection,
        model=model,
        role=role,
        output_field=output_field,
        agent_id=agent_id,
        tools=tools,
    )
    text = json.dumps(flow, indent=2, ensure_ascii=False)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
        click.echo(f"wrote {out_path} - now run: dynamiq workflow save <workflow_id> @{out_path}")
    else:
        click.echo(text)


def build_agent_flow(
    llm_connection: str,
    model: str = "gpt-4o",
    role: str = "You answer the user's question clearly and concisely.",
    output_field: str = "answer",
    agent_id: str = "agent",
    tools: tuple[str, ...] = (),
) -> dict:
    """A valid Input -> Agent -> Output flow (shared by `scaffold` and `create-agent`)."""
    tool_nodes = []
    for spec in tools:
        tool_type, _, tool_connection = spec.partition(":")
        if not tool_type or not tool_connection:
            raise click.ClickException(f"--tool must be TYPE:CONNECTION_ID, got {spec!r}")
        tool_nodes.append(
            {
                "id": str(uuid.uuid4()),
                "name": tool_type.rsplit(".", 1)[-1],
                "type": tool_type,
                "connection": tool_connection,
                "is_optimized_for_agents": True,
                "approval": {"enabled": False},
            }
        )

    agent_node = {
        "id": agent_id,
        "name": agent_id,
        "type": "dynamiq.nodes.agents.Agent",
        "depends": [{"node": "input"}],
        "input_transformer": {"selector": {"input": "$.input.output.input"}},
        "role": role,
        "max_loops": 10,
        "behaviour_on_max_loops": "return",
        "inference_mode": "XML",
        "llm": {
            "id": str(uuid.uuid4()),
            "name": "llm",
            "type": "dynamiq.nodes.llms.OpenAI",
            "model": model,
            "connection": llm_connection,
            "temperature": 0.1,
            "max_tokens": 4000,
        },
    }
    if tool_nodes:
        agent_node["tools"] = tool_nodes

    return {
        "id": str(uuid.uuid4()),
        "nodes": [
            {
                "id": "input",
                "name": "input",
                "type": "dynamiq.nodes.utils.Input",
                "schema": {
                    "type": "object",
                    "properties": {"input": {"type": "string", "required": True, "deletable": True}},
                },
            },
            agent_node,
            {
                "id": "output",
                "name": "output",
                "type": "dynamiq.nodes.utils.Output",
                "depends": [{"node": agent_id}],
                "input_transformer": {"selector": {output_field: f"$.{agent_id}.output.content"}},
                "schema": {
                    "type": "object",
                    "properties": {output_field: {"type": "Any", "required": True, "deletable": True}},
                },
            },
        ],
    }
