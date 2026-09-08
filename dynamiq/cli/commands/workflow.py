import json
import uuid

import click

from dynamiq.cli import flowcheck
from dynamiq.cli.client import ApiClient, ok
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
    if not ok(response):
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

        # A Pipedream account carries its app in a nested object, and its `name` is the
        # connected user's email - so several accounts for different apps list identically and
        # the one field that tells them apart is the one a flat field list drops. Lift it.
        app = item.get("app")
        if isinstance(app, dict):
            slug = app.get("name_slug") or app.get("name")
            if slug:
                row.setdefault("app_slug", slug)

        out.append(row or item)
    return out


def iter_pages(api: ApiClient, path: str, params: dict | None = None, page_size: int = 500):
    """Yield each page of a list endpoint until the last one.

    page_size caps at 500, so a single request is not the catalogue - it is the first 500 of
    it. Anything that reads one page and treats the result as complete reports whatever falls
    beyond it as not existing.
    """
    params = dict(params or {})
    seen, current = 0, 1
    while True:
        response = api.get(path, params={**params, "page": current, "page_size": page_size})
        if not ok(response):
            raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
        body = response.json()
        batch = body.get("data") or []
        if not batch:
            return
        yield batch
        seen += len(batch)
        pagination = body.get("pagination") or {}
        page_count = pagination.get("page_count")
        total = pagination.get("total_count")
        if page_count and current >= page_count:
            return
        if total is not None and seen >= total:
            return
        current += 1


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
        if not ok(response):
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
    for batch in iter_pages(api, path, params, page_size or 500):
        items.extend(batch)

    payload = {"count": len(items), "items": compact_items(items)} if compact else {"count": len(items), "data": items}
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def require_project(settings: Settings) -> str:
    """Project id from DYNAMIQ_PROJECT_ID / `dynamiq project set`; scoped endpoints need it."""
    if not settings.project_id:
        raise click.ClickException("No project set. Export DYNAMIQ_PROJECT_ID or run `dynamiq project set --id <id>`.")
    return settings.project_id


def check_connection(value, where: str) -> None:
    """A connection is a UUID, or a requirement reference standing in for one.

    No connection at all is normal - Input, Output and Pipedream nodes have none - so a
    missing value returns early. Folding it into the UUID branch failed every flow with an
    Input node, which is every flow.

    `requirement-add` documents {"$type": "requirement", "$id": ...} as the way a caller brings
    their own credential, and the validator accepts it - but this raised on anything that was
    not a UUID string, so a flow that passed `validate` could not be saved.
    """
    if value is None:
        return                       # Input, Output and Pipedream nodes carry no connection
    if isinstance(value, dict) and value.get("$type") == "requirement":
        if not value.get("$id"):
            raise click.ClickException(
                f"requirement reference on {where} has no `$id`. Declare it with "
                "`workflow requirement-add` and reference the id it returns."
            )
        return
    if not _is_uuid(str(value or "")):
        raise click.ClickException(
            f"connection {value!r} on {where} is not a connection UUID. Run "
            "`dynamiq connection list` and use the `id` of the one you want, or reference a "
            'requirement as {"$type": "requirement", "$id": "<id>"}.'
        )


def normalize_flow(flow, project_id: str | None = None):
    """Make a hand-written flow acceptable to the API without changing its meaning.

    The API is strict about a handful of shapes that are easy to get wrong by hand, and
    it either rejects the body outright or (worse) drops unknown keys silently. Each fix
    below is unambiguous, and every change is reported on stderr so nothing is secret:

    * missing / non-UUID `flow.id`            -> a fresh UUID
    * `depends: "input"` or `["input"]`       -> `[{"node": "input"}]`
    * node-level `selector`                   -> `input_transformer.selector`
    * missing node `name`                     -> the node id
    * `llm` / `tools[]` without an `id`       -> a fresh UUID each
    * Pipedream tool without `external_user_id` -> the current project id
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
            fill_external_user_id(tool, project_id, note, f"node {label!r}: tool")

        fill_external_user_id(node, project_id, note, "node")

    warn_tool_chained_after_agent(flow)
    return flow


def fill_external_user_id(node, project_id: str | None, note, where: str) -> None:
    """Give a Pipedream node the project id it binds accounts through.

    `external_user_id` has no default on the node model and the action-run payload sends it,
    so a tool without one cannot be constructed. Connections are bound to the project rather
    than to a person, so the value is never ambiguous - which is why the validator says the
    CLI supplies it. It has to actually do so.
    """
    if not isinstance(node, dict) or node.get("type") != PIPEDREAM_TYPE:
        return
    if node.get("external_user_id") or not project_id:
        return
    node["external_user_id"] = project_id
    note(f"{where} {node.get('name') or node.get('id') or '?'}: set external_user_id to the project id")


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


PIPEDREAM_TYPE = "dynamiq.nodes.tools.Pipedream"


def platform_node_types(api: ApiClient) -> set | None:
    """Every node type the PLATFORM accepts, from GET /v1/agent-builder/nodes.

    The authority on this is the API, not the SDK's package layout: which folder a class sits
    in is a different question from what the deployment will take, and they disagree. None on
    any failure, so the caller skips the check rather than inventing an answer.
    """
    found: set = set()

    def walk(node):
        if isinstance(node, dict):
            value = node.get("type")
            if isinstance(value, str) and value.startswith("dynamiq.nodes."):
                found.add(value)
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    try:
        for batch in iter_pages(api, "/v1/agent-builder/nodes"):
            walk(batch)
    except Exception:                                          # noqa: BLE001 - offline is fine
        return None
    return found or None


# A flow run is an agent doing LLM and tool calls; 30s is not enough.
EXECUTION_TIMEOUT = 600.0


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


def nested_custom_entries(node: dict, into: dict) -> None:
    """Record every node nested inside NODE, keyed by its own flow id.

    A tool or llm nested in an agent never appears on the canvas, so `flow_ui.nodes` has no
    entry for it and the editor addresses it by the `id` it carries in the flow.
    """
    children = []
    tools = node.get("tools")
    if isinstance(tools, list):
        children.extend(child for child in tools if isinstance(child, dict))
    for key in ("llm", "embedder", "memory"):
        child = node.get(key)
        if isinstance(child, dict):
            children.append(child)

    for child in children:
        child_id = child.get("id")
        if child_id and child_id not in into:
            into[child_id] = dict(child)
            nested_custom_entries(child, into)


def flow_ui_for(flow: dict, custom: dict | None = None) -> dict:
    """Canvas entries for a flow's nodes, in the exact shape the platform UI renders.

    `flow_ui` is required by create/save/release, so one is generated whenever the caller does
    not supply their own. The UI keys its rendering off fields a minimal payload does not have
    (`type`, `title`, `data.metadata_ui`, `data.custom_content`), and off `data.metadata.id`
    being the CANVAS node's uuid rather than the flow node's slug - a flow_ui missing those
    saves fine and then draws nothing. Shape verified against a platform-authored workflow.

    `custom_node_data` is the third key, and it is where the editor keeps everything the
    backend node model has no field for. It is keyed by the id the editor looks a node up by:
    the canvas uuid for a top-level node, and the flow `id` for a node nested inside an agent.

    What CANNOT be generated here is a Pipedream tool's `pipedreamApp` / `pipedreamComponent`:
    those come from Pipedream, not from the flow. Build such a tool with the skill's
    the skill's `pipedream_node` and pass the result via `--flow-ui`, or the tool draws
    with no logo, no account picker and no configuration form.
    """
    input_type = "dynamiq.nodes.utils.Input"
    output_type = "dynamiq.nodes.utils.Output"

    ui_ids: dict = {}
    nodes = []
    custom_node_data: dict = {}
    for i, node in enumerate(flow.get("nodes", [])):
        ui_id = str(uuid.uuid4())
        slug = node.get("id")
        ui_ids[slug] = ui_id
        node_type = node.get("type")
        position = {"x": i * 378, "y": 245.5}
        label = str(node.get("name") or slug or "").replace("-", " ").title()

        nodes.append(
            {
                "id": ui_id,
                "data": {
                    # metadata.id is the CANVAS id, not the flow node slug; the slug is `name`.
                    "metadata": {"id": ui_id, "name": slug, "type": node_type, "depends": []},
                    "metadata_ui": {"title": slug, "position": dict(position)},
                    "custom_content": {
                        "key": None,
                        "ref": None,
                        "type": "div",
                        "owner": None,
                        "props": {"children": label},
                    },
                },
                "desc": f"{slug} Node",
                "type": node_type,
                "title": slug,
                "width": 150,
                "height": 60,
                "dragging": False,
                "position": dict(position),
                "selected": False,
                # Input/Output are the fixed ends of a flow and the UI does not let you delete them.
                "deletable": node_type not in (input_type, output_type),
                "node_name": slug,
                "selectable": True,
                "position_absolute": dict(position),
            }
        )
        # A top-level node is looked up by its canvas uuid, and carries that uuid as its `id`.
        custom_node_data[ui_id] = {**node, "id": ui_id, "name": slug}
        nested_custom_entries(node, custom_node_data)

    edges = []
    for node in flow.get("nodes", []):
        target = ui_ids.get(node.get("id"))
        for dep in node.get("depends", []) or []:
            source = ui_ids.get(dep.get("node") if isinstance(dep, dict) else dep)
            if not source or not target:
                continue
            edges.append(
                {
                    "id": f"reactflow__edge-{source}source-{target}target",
                    "type": "smoothstep",
                    "label": None,
                    "style": {"stroke": "#96A1B8", "opacity": 1, "stroke_width": 2},
                    "source": source,
                    "target": target,
                    "animated": True,
                    "marker_end": {"type": "arrow", "color": "#96A1B8", "opacity": 1},
                    "source_handle": "source",
                    "target_handle": "target",
                    "is_choice_option": False,
                }
            )
    # A top-level node is keyed by the canvas uuid minted above, and a caller cannot predict
    # that - so an entry supplied under the flow node's id is resolved here. Nested nodes keep
    # their own id and fall through unchanged.
    for node_id, entry in (custom or {}).items():
        key = ui_ids.get(node_id, node_id)
        custom_node_data[key] = {**custom_node_data.get(key, {}), **entry}

    return {"nodes": nodes, "edges": edges, "custom_node_data": custom_node_data}


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
    # A bare name is the overwhelmingly common mistake here, and `read_json_arg` answers it
    # with a JSON parse error that names neither the problem nor the fix.
    if not payload.lstrip().startswith(("{", "@")):
        raise click.ClickException(
            f"`workflow create` takes a JSON payload, not a bare name. You passed {payload!r}. "
            f'Use: dynamiq workflow create \'{{"name": "{payload}"}}\''
        )
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    body.setdefault("flow", starter_flow())
    body["flow"] = normalize_flow(body["flow"], settings.project_id)
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
    flow_body = normalize_flow(read_json_arg(flow), settings.project_id)
    nodes = flow_body.get("nodes") or []
    if not allow_starter and len(nodes) == 1 and nodes[0].get("type") == "dynamiq.nodes.utils.Input":
        raise click.ClickException(
            "This flow contains only the starter Input node, so saving it would leave the workflow empty. "
            "Build the real DAG first and save that, or pass --allow-starter if you "
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
@click.option(
    "--dry-run/--no-dry-run",
    default=True,
    help="Sent as dry_run. On by default: without it the endpoint answers 400 bad_input. "
    "It does NOT stop nodes executing - tools really act.",
)
@click.option("--last-node-output", is_flag=True, help="Return only the last node's output.")
@with_api_and_settings
def test_workflow(
    *, api: ApiClient, settings: Settings, flow: str, input_data: str, dry_run: bool, last_node_output: bool
):
    """Run a flow with the given input, without saving or releasing.

    This endpoint takes a FORM (not a JSON body): `flow` and `input` are sent as
    JSON-encoded strings. FLOW/INPUT_DATA are inline JSON or @file. No project_id needed.

    `dry_run` is on by default because it is the only form the endpoint accepts - without it
    the answer is `400 bad_input` with an empty details object. Despite the name it is NOT a
    simulation: the flow executes and its tools really act, so a Notion tool creates a real
    page. Choose an obviously-test input.
    """
    form = {
        "flow": json.dumps(normalize_flow(read_json_arg(flow), settings.project_id)),
        "input": json.dumps(read_json_arg(input_data)),
        "stream": "false",
    }
    if dry_run:
        form["dry_run"] = "true"
    if last_node_output:
        form["last_node_output"] = "true"
    # This endpoint takes multipart; a urlencoded body is answered with 415.
    # Not retried, and given room to finish. A real run is an agent doing LLM and tool calls,
    # which routinely outlasts the default 30s; retrying a POST the server already accepted
    # executes the flow again, tools really acting each time.
    response = api.post("/v1/workflows/test", files={k: (None, v) for k, v in form.items()},
                        timeout=EXECUTION_TIMEOUT)
    if response.status_code == 415:
        click.echo("note: multipart rejected (415); retrying form-urlencoded.", err=True)
        response = api.post("/v1/workflows/test", data=form,
                            timeout=EXECUTION_TIMEOUT)
    echo_response(response)


@workflow.command("node-types")
@click.argument("group", required=False)
@with_api_and_settings
def list_node_types(*, api: ApiClient, settings: Settings, group: str | None):
    """List the node types the PLATFORM accepts, optionally filtered by GROUP.

    GET /v1/agent-builder/nodes. GROUP is a family such as agents, tools, llms, utils.
    These strings are the only valid values for a node's `type`; there is no local list to
    fall back on, so run this before writing a flow rather than guessing a type.
    """
    echo_response(api.get("/v1/agent-builder/nodes", params={"group": group} if group else None))


@workflow.command("requirements")
@click.argument("workflow_id")
@with_api_and_settings
def list_requirements(*, api: ApiClient, settings: Settings, workflow_id: str):
    """List a workflow's requirements - the credentials each end user supplies at run time."""
    echo_response(api.get(f"/v1/workflows/{workflow_id}/requirements"))


@workflow.command("requirement-add")
@click.argument("workflow_id")
@click.argument("payload")
@with_api_and_settings
def add_requirement(*, api: ApiClient, settings: Settings, workflow_id: str, payload: str):
    """Declare a requirement so each caller brings their OWN account instead of a pinned one.

    REQUIRED: `name`, `type`, `form` {title, description}, `spec`.
      type "pipedream_account" -> spec {"app_slug": "notion"}
      type "connection"        -> spec {"type": "dynamiq.connections.<X>"}

    `form.title` is what the end user reads on the connect screen, so write it for them.
    Reference the returned id from the flow as
    {"$type": "requirement", "$id": "<id>", "value_path": "$.account_id"}.
    """
    echo_response(api.post(f"/v1/workflows/{workflow_id}/requirements", json=read_json_arg(payload)))


@workflow.command("requirement-get")
@click.argument("workflow_id")
@click.argument("requirement_id")
@with_api_and_settings
def get_requirement(*, api: ApiClient, settings: Settings, workflow_id: str, requirement_id: str):
    """Fetch one requirement, including the spec a flow placeholder resolves against."""
    echo_response(api.get(f"/v1/workflows/{workflow_id}/requirements/{requirement_id}"))


@workflow.command("requirement-update")
@click.argument("workflow_id")
@click.argument("requirement_id")
@click.argument("payload")
@with_api_and_settings
def update_requirement(
    *, api: ApiClient, settings: Settings, workflow_id: str, requirement_id: str, payload: str
):
    """Change a requirement's `form` (its title/description). Body: {"form": {...}}."""
    echo_response(
        api.put(f"/v1/workflows/{workflow_id}/requirements/{requirement_id}", json=read_json_arg(payload))
    )


@workflow.command("requirement-delete")
@click.argument("workflow_id")
@click.argument("requirement_id")
@click.confirmation_option(prompt="Delete this requirement?")
@with_api_and_settings
def delete_requirement(*, api: ApiClient, settings: Settings, workflow_id: str, requirement_id: str):
    """Delete a requirement. Any flow placeholder still pointing at it will fail to resolve."""
    echo_response(api.delete(f"/v1/workflows/{workflow_id}/requirements/{requirement_id}"))


@workflow.command("release")
@click.argument("workflow_id")
@click.option("--name", default=None, help="New name for the released version (defaults to the current name).")
@click.option("--flow", default=None, help="Flow JSON to release (inline or @file); defaults to the saved flow.")
@click.option("--flow-ui", default=None, help="Canvas layout JSON; defaults to the saved flow_ui.")
@click.option("--allow-starter", is_flag=True, help="Permit releasing a workflow that has only the starter Input node.")
@with_api_and_settings
def release_workflow(
    *,
    api: ApiClient,
    settings: Settings,
    workflow_id: str,
    name: str | None,
    flow: str | None,
    flow_ui: str | None,
    allow_starter: bool,
):
    """Release a new version. The API requires name, flow and flow_ui in the body, so the
    workflow's current values are fetched and re-sent unless overridden by the options.
    """
    current = api.get(f"/v1/workflows/{workflow_id}")
    if not ok(current):
        raise click.ClickException(f"HTTP {current.status_code}: {current.text.strip()[:2000]}")
    data = current.json().get("data", {})

    flow_body = normalize_flow(read_json_arg(flow), settings.project_id) if flow else data.get("flow")
    if not flow_body:
        raise click.ClickException("Workflow has no saved flow to release. Run `workflow save` first.")

    nodes = flow_body.get("nodes") or []
    if not allow_starter and (
        not nodes or (len(nodes) == 1 and nodes[0].get("type") == "dynamiq.nodes.utils.Input")
    ):
        raise click.ClickException(
            f"Workflow {workflow_id} still holds only the starter Input node, so releasing it would "
            "publish an empty workflow. `workflow create` seeds that starter and only `workflow save` "
            "replaces it - save the real DAG, confirm it with `workflow get`, then release. Pass "
            "--allow-starter to override."
        )
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


@workflow.command("validate")
@click.argument("flow")
@click.option("--offline", is_flag=True, help="Skip the node-type check instead of asking the platform.")
@with_api_and_settings
def validate_flow_command(*, api: ApiClient, settings: Settings, flow: str, offline: bool):
    """Check a flow JSON locally, before it is saved. Exits non-zero on any error.

    FLOW is inline JSON or @file. Nothing is sent anywhere; this is a read of the file.

    The API accepts a flow it cannot run - unknown keys are dropped rather than rejected -
    so a misplaced selector yields empty output instead of an error, and a tool with the
    wrong schema is simply never callable. This catches those before they are persisted.
    """
    known = None if offline else platform_node_types(api)
    if known is None and not offline:
        click.echo("note: could not reach /v1/agent-builder/nodes; node types were not checked.", err=True)
    errors, warnings = flowcheck.validate(read_json_arg(flow), known_types=known)
    for warning in warnings:
        click.echo(f"warning: {warning}", err=True)
    if errors:
        click.echo("", err=True)
        for i, error in enumerate(errors, 1):
            click.echo(f"  {i}. {error}", err=True)
        raise click.ClickException(f"{len(errors)} problem(s); fix them before saving.")
    click.echo(json.dumps({"valid": True, "nodes": len(read_json_arg(flow).get("nodes") or [])}, indent=2))


@workflow.command("flow-ui")
@click.argument("flow")
@click.option("--out", "out_path", default=None, help="Write here instead of stdout.")
@click.option(
    "--custom",
    "custom_paths",
    multiple=True,
    help="JSON of extra custom_node_data entries (repeatable). Accepts the output of "
    "the skill's `pipedream_node`, or a bare {node_id: entry} mapping.",
)
@with_api_and_settings
def build_flow_ui(*, api: ApiClient, settings: Settings, flow: str, out_path, custom_paths):
    """Generate the canvas payload for a FLOW, without saving anything.

    `save` and `release` build one for you when you do not pass `--flow-ui`. Use this when
    you need to inspect it, or to merge in data the flow cannot carry - a Pipedream tool's
    app and component records live only in `custom_node_data`, and the skill's
    `pipedream_node` emits them in the shape this accepts.
    """
    flow_body = normalize_flow(read_json_arg(flow), settings.project_id)
    custom: dict = {}
    for path in custom_paths:
        payload = read_json_arg(path)
        entries = payload.get("custom_node_data", payload) if isinstance(payload, dict) else None
        if not isinstance(entries, dict):
            raise click.ClickException(f"{path}: expected an object of node id -> entry.")
        custom.update(entries)

    flow_ui = flow_ui_for(flow_body, custom)

    undepicted = [
        entry.get("name") or node_id
        for node_id, entry in flow_ui["custom_node_data"].items()
        if entry.get("type") == PIPEDREAM_TYPE and not entry.get("pipedreamComponent")
    ]
    if undepicted:
        click.echo(
            "warning: no component record for " + ", ".join(map(str, undepicted)) + " - these tools "
            "will save and then draw with no logo, no account picker and no configuration form. "
            "Build each with the skill's `pipedream_node` and pass it via --custom.",
            err=True,
        )

    text = json.dumps(flow_ui, indent=2)
    if out_path:
        with open(out_path, "w") as handle:
            handle.write(text + "\n")
        click.echo(json.dumps({"wrote": out_path, "nodes": len(flow_ui["nodes"])}, indent=2))
    else:
        click.echo(text)


@workflow.command("verify")
@click.argument("workflow_id")
@with_api_and_settings
def verify_workflow_command(*, api: ApiClient, settings: Settings, workflow_id: str):
    """Read a SAVED workflow back and check the DAG actually persisted.

    `save` answering 200 does not mean the flow was stored as written: a payload the API
    could not read leaves the workflow holding its starter Input node, and nothing says so.
    This fetches it and reports what is really there.
    """
    response = api.get(f"/v1/workflows/{workflow_id}")
    if not ok(response):
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    data = response.json().get("data", {})
    flow = data.get("flow") or {}
    nodes = flow.get("nodes") or []
    errors, _ = flowcheck.validate(flow, known_types=platform_node_types(api))

    summary = {
        "id": data.get("id"),
        "name": data.get("name"),
        "node_count": len(nodes),
        "nodes": compact_items(nodes),
        "has_flow_ui": bool(data.get("flow_ui")),
        "custom_node_data": len((data.get("flow_ui") or {}).get("custom_node_data") or {}),
        "problems": errors,
    }
    click.echo(json.dumps(summary, indent=2, ensure_ascii=False))

    if len(nodes) <= 1:
        raise click.ClickException(
            "the workflow holds only the starter node - the save did not persist. Re-check the "
            "flow with `workflow validate`, then save again."
        )
    if errors:
        raise click.ClickException(f"{len(errors)} problem(s) in the saved flow.")
