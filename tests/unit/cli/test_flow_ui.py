import json
import uuid
from types import SimpleNamespace

from click.testing import CliRunner

from dynamiq.cli import flowcheck
from dynamiq.cli.commands.context import DynamiqCtx
from dynamiq.cli.commands.workflow import canvas_mismatch, flow_ui_for, redraw_flow_ui, workflow
from dynamiq.cli.config import Settings

CHOICE = "dynamiq.nodes.operators.Choice"
PIPEDREAM = "dynamiq.nodes.tools.Pipedream"
AGENT = "dynamiq.nodes.agents.Agent"
STICKY_NOTE = "onlyUINode.StickyNote"

# The skill's `pipedream_node` gives a tool a uuid id and a slug name, and the flow refers to it by
# the id. This is the shape of the router whose canvas the editor could not open.
SITES_ID = "2e92550d-e1db-4dac-8bf0-cca222166da4"
COLLECTIONS_ID = "025912c9-3c63-4e16-b9d4-80795086f0f6"


def pipedream_tool(tool_id: str, name: str, option: str) -> dict:
    return {
        "id": tool_id,
        "name": name,
        "type": PIPEDREAM,
        "action_id": f"webflow-list-{name}",
        "depends": [{"node": "input"}, {"node": "route", "option": option}],
    }


def router_flow(*tools: dict) -> dict:
    options = [
        {"id": tool["depends"][1]["option"], "name": tool["depends"][1]["option"], "condition": {"value": "x"}}
        for tool in tools
    ]
    return {
        "id": str(uuid.uuid4()),
        "nodes": [
            {"id": "input", "name": "input", "type": flowcheck.INPUT_TYPE},
            {"id": "route", "name": "route", "type": CHOICE, "depends": [{"node": "input"}], "options": options},
            *tools,
            {
                "id": "output",
                "name": "output",
                "type": flowcheck.OUTPUT_TYPE,
                "depends": [{"node": tool["id"]} for tool in tools],
                "input_transformer": {"selector": {tool["name"]: f"$.{tool['id']}.output.content" for tool in tools}},
            },
        ],
    }


SITES = pipedream_tool(SITES_ID, "sites", "sites")
COLLECTIONS = pipedream_tool(COLLECTIONS_ID, "collections", "collections")


def canvas_node_for(ui: dict, flow_id: str) -> dict:
    return next(node for node in ui["nodes"] if node["data"]["metadata"]["name"] == flow_id)


def test_a_node_whose_id_is_not_its_name_is_drawn_under_its_id_and_labelled_by_its_name():
    ui = flow_ui_for(router_flow(SITES))
    node = canvas_node_for(ui, SITES_ID)

    # The editor saves metadata.name back as the node's id, so it must hold the id: an editor
    # save of a canvas that held the name would rename the node and break "$.<id>" selectors.
    assert node["data"]["metadata"] == {"id": node["id"], "name": SITES_ID, "type": PIPEDREAM, "depends": []}
    assert node["title"] == node["node_name"] == node["data"]["metadata_ui"]["title"] == "sites"
    assert node["desc"] == "sites Node"
    assert node["data"]["custom_content"]["props"]["children"] == "Sites"

    entry = ui["custom_node_data"][node["id"]]
    assert entry["id"] == node["id"]
    assert entry["name"] == "sites"
    assert entry["action_id"] == "webflow-list-sites"


def test_edges_and_branches_reach_a_node_through_its_id():
    ui = flow_ui_for(router_flow(SITES))
    route, sites, output = (canvas_node_for(ui, flow_id)["id"] for flow_id in ("route", SITES_ID, "output"))

    branch = next(edge for edge in ui["edges"] if edge["source"] == route and edge["target"] == sites)
    assert branch["source_handle"] == "sites"
    assert branch["is_choice_option"] is True
    assert [edge for edge in ui["edges"] if edge["source"] == sites and edge["target"] == output]


def test_a_custom_record_for_a_top_level_node_lands_on_its_canvas_node_under_the_canvas_id():
    # `pipedream_node` keys its record by the tool's flow id and carries that id inside it.
    record = {"id": SITES_ID, "name": "sites", "pipedreamApp": {"name_slug": "webflow"}}

    ui = flow_ui_for(router_flow(SITES), {SITES_ID: record})
    node = canvas_node_for(ui, SITES_ID)

    assert SITES_ID not in ui["custom_node_data"]
    entry = ui["custom_node_data"][node["id"]]
    assert entry["id"] == node["id"]
    assert entry["pipedreamApp"] == {"name_slug": "webflow"}
    assert entry["action_id"] == "webflow-list-sites"


def test_a_custom_record_for_a_nested_tool_stays_under_the_tools_own_id():
    tool_id = str(uuid.uuid4())
    agent = {
        "id": "agent",
        "name": "agent",
        "type": AGENT,
        "depends": [{"node": "input"}],
        "tools": [{"id": tool_id, "name": "sites", "type": PIPEDREAM}],
    }
    flow = {
        "id": str(uuid.uuid4()),
        "nodes": [{"id": "input", "name": "input", "type": flowcheck.INPUT_TYPE}, agent],
    }

    ui = flow_ui_for(flow, {tool_id: {"id": tool_id, "pipedreamApp": {"name_slug": "webflow"}}})

    # A nested tool has no canvas node; the editor finds its record under the tool's own id.
    assert ui["custom_node_data"][tool_id]["id"] == tool_id
    assert ui["custom_node_data"][tool_id]["pipedreamApp"] == {"name_slug": "webflow"}


def test_a_canvas_generated_for_a_flow_matches_it():
    flow = router_flow(SITES, COLLECTIONS)

    assert canvas_mismatch(flow, flow_ui_for(flow)) is None


def test_a_canvas_drawn_for_another_flow_names_what_it_misses_and_what_it_has_extra():
    saved = flow_ui_for(router_flow(SITES))
    note = {"id": "note-1", "type": STICKY_NOTE, "data": {"metadata": {"id": "note-1", "name": "note-1"}}}
    saved["nodes"].append(note)

    mismatch = canvas_mismatch(router_flow(COLLECTIONS), saved)

    # A sticky note stands for no flow node, so it is neither missing nor stale.
    assert mismatch == {"undrawn": [COLLECTIONS_ID], "stale": [SITES_ID]}


def test_an_empty_or_missing_canvas_draws_none_of_the_flow():
    flow = router_flow(SITES)
    every_node = [node["id"] for node in flow["nodes"]]

    assert canvas_mismatch(flow, {}) == {"undrawn": every_node, "stale": []}
    assert canvas_mismatch(flow, None) == {"undrawn": every_node, "stale": []}


def test_a_redrawn_canvas_keeps_the_records_of_the_nodes_still_in_the_flow():
    saved = flow_ui_for(router_flow(SITES), {SITES_ID: {"pipedreamApp": {"name_slug": "webflow"}}})

    ui = redraw_flow_ui(router_flow(SITES, COLLECTIONS), saved)

    assert canvas_mismatch(router_flow(SITES, COLLECTIONS), ui) is None
    sites = ui["custom_node_data"][canvas_node_for(ui, SITES_ID)["id"]]
    assert sites["pipedreamApp"] == {"name_slug": "webflow"}
    # The flow's fields come from the flow being released, not from the saved record.
    assert sites["id"] == canvas_node_for(ui, SITES_ID)["id"]
    assert sites["depends"] == SITES["depends"]
    assert "pipedreamApp" not in ui["custom_node_data"][canvas_node_for(ui, COLLECTIONS_ID)["id"]]


def test_a_redrawn_canvas_drops_the_records_of_nodes_that_left_the_flow():
    saved = flow_ui_for(router_flow(SITES), {SITES_ID: {"pipedreamApp": {"name_slug": "webflow"}}})

    ui = redraw_flow_ui(router_flow(COLLECTIONS), saved)

    assert not [entry for entry in ui["custom_node_data"].values() if entry.get("pipedreamApp")]
    assert not [entry for entry in ui["custom_node_data"].values() if entry.get("id") == SITES_ID]


class RecordingApi:
    """Answers `workflow get` with a saved workflow and records what `release` sends."""

    def __init__(self, saved: dict):
        self.saved = saved
        self.posts = []

    def get(self, path, **kwargs):
        body = {"data": self.saved}
        return SimpleNamespace(status_code=200, text=json.dumps(body), json=lambda: body)

    def post(self, path, **kwargs):
        self.posts.append((path, kwargs["json"]))
        return SimpleNamespace(status_code=200, text="{}", json=lambda: {})


def release(saved: dict, *args: str) -> tuple[dict, str]:
    api = RecordingApi(saved)
    dctx = DynamiqCtx()
    dctx.settings = Settings(project_id="00000000-0000-4000-8000-000000000001")
    dctx.api = api
    result = CliRunner().invoke(workflow, ["release", "wf-1", *args], obj=dctx)
    assert result.exit_code == 0, result.output
    path, body = api.posts[0]
    assert path == "/v1/workflows/wf-1/release"
    # Notes go to stderr, which `output` includes in every click version.
    return body, result.output


def saved_workflow(flow: dict) -> dict:
    return {"name": "router", "flow": flow, "flow_ui": flow_ui_for(flow)}


def test_releasing_the_saved_flow_sends_the_saved_canvas():
    saved = saved_workflow(router_flow(SITES))

    body, notes = release(saved)

    assert body["flow_ui"] == saved["flow_ui"]
    assert "saved canvas" not in notes


def test_releasing_a_flow_the_saved_canvas_still_draws_keeps_the_saved_canvas():
    saved = saved_workflow(router_flow(SITES))

    body, notes = release(saved, "--flow", json.dumps(router_flow(SITES)))

    # Positions and records the editor saved survive when the nodes are the same.
    assert body["flow_ui"] == saved["flow_ui"]
    assert "saved canvas" not in notes


def test_releasing_a_changed_flow_draws_a_canvas_for_it_instead_of_the_saved_one():
    saved = saved_workflow(router_flow(SITES))
    changed = router_flow(SITES, COLLECTIONS)

    body, notes = release(saved, "--flow", json.dumps(changed))

    assert canvas_mismatch(body["flow"], body["flow_ui"]) is None
    assert "saved canvas does not match this flow" in notes
    assert COLLECTIONS_ID in notes


def test_releasing_with_a_canvas_sends_that_canvas():
    saved = saved_workflow(router_flow(SITES))
    own = flow_ui_for(router_flow(SITES, COLLECTIONS))

    body, _ = release(saved, "--flow", json.dumps(router_flow(SITES, COLLECTIONS)), "--flow-ui", json.dumps(own))

    assert body["flow_ui"] == own


def verify(saved: dict) -> tuple[dict, str]:
    api = RecordingApi(saved)
    dctx = DynamiqCtx()
    dctx.settings = Settings(project_id="00000000-0000-4000-8000-000000000001")
    dctx.api = api
    result = CliRunner().invoke(workflow, ["verify", "wf-1"], obj=dctx)
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output[: result.output.rindex("}") + 1])
    return summary, result.output


TOTALS_ID = "5a1c1f0e-8d2b-4f5e-9a3c-7b6d4e2f1a09"


def pipeline(*steps: dict) -> dict:
    """A flow the checker passes: Input, the given steps in a row, and an Output reading the last."""
    nodes = [{"id": "input", "name": "input", "type": flowcheck.INPUT_TYPE}]
    for step in steps:
        nodes.append({**step, "depends": [{"node": nodes[-1]["id"]}]})
    nodes.append(
        {
            "id": "output",
            "name": "output",
            "type": flowcheck.OUTPUT_TYPE,
            "depends": [{"node": nodes[-1]["id"]}],
            "input_transformer": {"selector": {"result": f"$.{nodes[-1]['id']}.output"}},
        }
    )
    return {"id": str(uuid.uuid4()), "nodes": nodes}


TOTALS = {
    "id": TOTALS_ID,
    "name": "totals",
    "type": "dynamiq.nodes.operators.Expression",
    "expressions": [{"key": "x", "expression": "1"}],
}


def test_verify_reports_a_canvas_that_matches_the_flow(monkeypatch):
    monkeypatch.setattr("dynamiq.cli.commands.workflow.platform_node_types", lambda api: None)

    summary, output = verify(saved_workflow(pipeline(TOTALS)))

    assert summary["canvas"] == "matches the flow"
    assert "saved canvas does not match" not in output


def test_verify_warns_about_a_canvas_drawn_for_another_flow(monkeypatch):
    monkeypatch.setattr("dynamiq.cli.commands.workflow.platform_node_types", lambda api: None)
    saved = saved_workflow(pipeline())
    saved["flow"] = pipeline(TOTALS)

    summary, output = verify(saved)

    # A warning, not a failure: the editor draws the missing node, so the workflow still works.
    assert summary["canvas"] == {"undrawn": [TOTALS_ID], "stale": []}
    assert "saved canvas does not match the flow" in output
