import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options, read_json_arg, require_project
from dynamiq.cli.config import Settings

# A simulation runs a whole conversation; the default 30s is not enough.
EXECUTION_TIMEOUT = 600.0

voice = click.Group(name="voice", help="Voice agents: build, deploy, inspect calls")

BASE = "/v1/agents/voice"


@voice.command("catalog")
@with_api_and_settings
def voice_catalog(*, api: ApiClient, settings: Settings):
    """Providers, models and voices this platform can serve.

    Run this BEFORE writing a config: every `provider`/`model`/`voice` in a voice agent comes
    from here, and a name that is merely plausible fails at deploy time rather than at create.
    """
    echo_response(api.get(f"{BASE}/catalog"))


@voice.command("list")
@pagination_options
@with_api_and_settings
def list_voice_agents(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List voice agents in the current project."""
    echo_list(api, f"{BASE}/agents", {"project_id": require_project(settings)}, page, page_size, fetch_all, compact)


@voice.command("create")
@click.argument("payload")
@with_api_and_settings
def create_voice_agent(*, api: ApiClient, settings: Settings, payload: str):
    """Create a voice agent. REQUIRED: `name`, `config`; `project_id` is filled in automatically.

    `config` REQUIRES `instructions` and a `mode`, and the mode decides which stages it carries:

      pipeline (default) -> `stt`, `llm` and `tts`, all three, and NO `realtime`
      realtime           -> `realtime` alone, and none of stt/llm/tts

    Each stage is {"provider", "model", "connection_id", ...}; `connection_id` is a Nexus
    connection UUID and the agent cannot run without it. Optional: `welcome_message`, `tools`,
    `mcp_servers`, `transfer`, `end_call_enabled`, `recording_enabled`, `advanced`.

    Creating does not deploy. Nothing can call the agent until `voice deploy`.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post(f"{BASE}/agents", json=body))


@voice.command("get")
@click.argument("agent_id")
@with_api_and_settings
def get_voice_agent(*, api: ApiClient, settings: Settings, agent_id: str):
    """Fetch one voice agent - `status`, `deploy_error` and `phone_numbers` live here."""
    echo_response(api.get(f"{BASE}/agents/{agent_id}"))


@voice.command("update")
@click.argument("agent_id")
@click.argument("payload")
@with_api_and_settings
def update_voice_agent(*, api: ApiClient, settings: Settings, agent_id: str, payload: str):
    """Update name, description or config. PATCH, and `config` is a FULL replacement.

    A deployed agent does NOT pick this up on its own, and deploying again over a live one is
    refused - the cycle is `voice undeploy <id>` then `voice deploy <id>`, then poll `get`
    until `status` is deployed again.
    """
    echo_response(api.patch(f"{BASE}/agents/{agent_id}", json=read_json_arg(payload)))


@voice.command("deploy")
@click.argument("agent_id")
@with_api_and_settings
def deploy_voice_agent(*, api: ApiClient, settings: Settings, agent_id: str):
    """Deploy the agent so it can take calls.

    Returns before it is live. Poll `voice get <id>` until `status` settles; a failure lands in
    `deploy_error` on the agent, which is the first thing to read when a deploy does not take.
    """
    echo_response(api.post(f"{BASE}/agents/{agent_id}/deploy"))


@voice.command("undeploy")
@click.argument("agent_id")
@with_api_and_settings
def undeploy_voice_agent(*, api: ApiClient, settings: Settings, agent_id: str):
    """Take the agent offline. Calls to it stop being answered."""
    echo_response(api.post(f"{BASE}/agents/{agent_id}/undeploy"))


@voice.command("deployments")
@click.argument("agent_id")
@with_api_and_settings
def voice_deployments(*, api: ApiClient, settings: Settings, agent_id: str):
    """Deployment history for a voice agent."""
    echo_response(api.get(f"{BASE}/agents/{agent_id}/deployments"))


@voice.command("calls")
@click.argument("agent_id")
@pagination_options
@with_api_and_settings
def list_calls(*, api: ApiClient, settings: Settings, agent_id: str, page, page_size, fetch_all, compact):
    """Calls this agent has handled - the proof it is actually working."""
    echo_list(api, f"{BASE}/agents/{agent_id}/calls", None, page, page_size, fetch_all, compact)


@voice.command("call")
@click.argument("agent_id")
@click.argument("call_id")
@with_api_and_settings
def get_call(*, api: ApiClient, settings: Settings, agent_id: str, call_id: str):
    """One call, including its transcript."""
    echo_response(api.get(f"{BASE}/agents/{agent_id}/calls/{call_id}"))


@voice.command("recording")
@click.argument("agent_id")
@click.argument("call_id")
@with_api_and_settings
def get_recording(*, api: ApiClient, settings: Settings, agent_id: str, call_id: str):
    """A short-lived URL for the call recording.

    Treat it like a connect link: give it to the user, do not open it yourself.
    """
    echo_response(api.get(f"{BASE}/agents/{agent_id}/calls/{call_id}/recording-url"))


@voice.command("telephony")
@click.argument("agent_id")
@with_api_and_settings
def get_telephony(*, api: ApiClient, settings: Settings, agent_id: str):
    """Phone numbers and SIP trunks attached to this agent."""
    echo_response(api.get(f"{BASE}/agents/{agent_id}/telephony"))


@voice.command("simulate")
@click.argument("agent_id")
@click.argument("payload")
@with_api_and_settings
def simulate(*, api: ApiClient, settings: Settings, agent_id: str, payload: str):
    """Run a simulated call against the agent - a test that needs no phone.

    PAYLOAD REQUIRES `simulation_set_id`: a run replays the scenarios of a saved set, so the
    set exists first. List them with `voice simulation-sets`, or make one with
    `voice simulation-set-create`. Sets are project-scoped, not per agent, so one set tests
    many agents.

    Optional: `mode` ("text" default), `concurrency`.

    Use this as the "does it work" gate before telling anyone the agent is ready.
    """
    # Each attempt creates a simulation. A retry after a timed-out response leaves a second
    # record behind, and there is no idempotency key to tell them apart.
    echo_response(api.post(f"{BASE}/agents/{agent_id}/simulations", json=read_json_arg(payload),
                           timeout=EXECUTION_TIMEOUT))


@voice.command("simulations")
@click.argument("agent_id")
@pagination_options
@with_api_and_settings
def list_simulations(*, api: ApiClient, settings: Settings, agent_id: str, page, page_size, fetch_all, compact):
    """Simulation runs and their results."""
    echo_list(api, f"{BASE}/agents/{agent_id}/simulations", None, page, page_size, fetch_all, compact)


@voice.command("simulation-sets")
@pagination_options
@with_api_and_settings
def list_simulation_sets(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """Saved scenario sets, which `voice simulate` runs against an agent.

    GET /v1/agents/voice/simulations/sets. Project-scoped, not per agent - the `id` here is
    the `simulation_set_id` a run needs.
    """
    echo_list(api, f"{BASE}/simulations/sets", {"project_id": require_project(settings)},
              page, page_size, fetch_all, compact)


@voice.command("simulation-set")
@click.argument("set_id")
@with_api_and_settings
def get_simulation_set(*, api: ApiClient, settings: Settings, set_id: str):
    """One scenario set, including the scenarios it will replay."""
    echo_response(api.get(f"{BASE}/simulations/sets/{set_id}",
                          params={"project_id": require_project(settings)}))


@voice.command("simulation-set-create")
@click.argument("payload")
@with_api_and_settings
def create_simulation_set(*, api: ApiClient, settings: Settings, payload: str):
    """Create a scenario set. REQUIRED: `name`, `scenarios`; `project_id` is filled in.

    `scenarios` is the list a run replays - each one a conversation to put the agent through.
    Nothing can be simulated until a set exists, so this comes before `voice simulate`.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post(f"{BASE}/simulations/sets", json=body))


@voice.command("simulation-set-update")
@click.argument("set_id")
@click.argument("payload")
@with_api_and_settings
def update_simulation_set(*, api: ApiClient, settings: Settings, set_id: str, payload: str):
    """Change a scenario set. The body is a full replacement - read it with
    `simulation-set` first and edit that, or a field you omit is cleared."""
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.put(f"{BASE}/simulations/sets/{set_id}", json=body))


@voice.command("simulation-set-delete")
@click.argument("set_id")
@click.confirmation_option(prompt="Delete this simulation set?")
@with_api_and_settings
def delete_simulation_set(*, api: ApiClient, settings: Settings, set_id: str):
    """Delete a scenario set. Runs that used it keep their results."""
    echo_response(api.delete(f"{BASE}/simulations/sets/{set_id}",
                             params={"project_id": require_project(settings)}))


@voice.command("simulation-scenarios")
@click.argument("agent_id")
@click.argument("payload", required=False)
@with_api_and_settings
def generate_scenarios(*, api: ApiClient, settings: Settings, agent_id: str, payload: str | None):
    """Have the platform draft scenarios for this agent, from its own instructions.

    POST .../simulations/generate-scenarios. The quickest way to fill a set without writing
    conversations by hand - take the output, and save it with `simulation-set-create`.
    """
    echo_response(api.post(f"{BASE}/agents/{agent_id}/simulations/generate-scenarios",
                           json=read_json_arg(payload) if payload else {},
                           timeout=EXECUTION_TIMEOUT))


@voice.command("simulation")
@click.argument("agent_id")
@click.argument("run_id")
@with_api_and_settings
def get_simulation(*, api: ApiClient, settings: Settings, agent_id: str, run_id: str):
    """One simulation run - status, per-scenario pass/fail and transcripts."""
    echo_response(api.get(f"{BASE}/agents/{agent_id}/simulations/{run_id}"))


@voice.command("simulation-cancel")
@click.argument("agent_id")
@click.argument("run_id")
@with_api_and_settings
def cancel_simulation(*, api: ApiClient, settings: Settings, agent_id: str, run_id: str):
    """Stop a run that is still going."""
    echo_response(api.post(f"{BASE}/agents/{agent_id}/simulations/{run_id}/cancel"))


@voice.command("simulation-status")
@with_api_and_settings
def simulation_status(*, api: ApiClient, settings: Settings):
    """Whether the simulation service is available for this project."""
    echo_response(api.get(f"{BASE}/simulations/status",
                          params={"project_id": require_project(settings)}))


@voice.command("duplicate")
@click.argument("agent_id")
@with_api_and_settings
def duplicate_voice_agent(*, api: ApiClient, settings: Settings, agent_id: str):
    """Copy an agent, config and all - the safe way to try a change on a live one."""
    echo_response(api.post(f"{BASE}/agents/{agent_id}/duplicate"))


@voice.command("delete")
@click.argument("agent_id")
@click.confirmation_option(prompt="Delete this voice agent?")
@with_api_and_settings
def delete_voice_agent(*, api: ApiClient, settings: Settings, agent_id: str):
    """Delete the agent permanently."""
    echo_response(api.delete(f"{BASE}/agents/{agent_id}"))
