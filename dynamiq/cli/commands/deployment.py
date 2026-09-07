"""Model and database deployments: fine-tuning, inference endpoints, vector databases.

These three share a shape the workflow commands do not: they provision real infrastructure,
take minutes rather than seconds, and cost money while they exist. Every create here returns
before the thing is ready, so each group has an explicit way to watch it come up.
"""

import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options, read_json_arg, require_project
from dynamiq.cli.config import Settings

# Generation and training calls run far past the default 30s.
EXECUTION_TIMEOUT = 600.0

finetuning = click.Group(name="fine-tuning", help="Fine-tuning jobs and the adapters they produce")
inference = click.Group(name="inference", help="Deploy and call self-hosted models")
database = click.Group(name="database", help="Deploy vector databases")


# --------------------------------------------------------------------------- fine-tuning


@finetuning.command("jobs")
@pagination_options
@with_api_and_settings
def list_jobs(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List fine-tuning jobs in the current project."""
    echo_list(
        api, "/v1/fine-tuning/jobs", {"project_id": require_project(settings)}, page, page_size, fetch_all, compact
    )


@finetuning.command("start")
@click.argument("payload")
@with_api_and_settings
def start_job(*, api: ApiClient, settings: Settings, payload: str):
    """Start a fine-tuning job.

    REQUIRED: `name`, `model_id`, `training_file_id`, `resource_profile_id`, `hyperparameters`
    ({"batch_size", "number_of_epochs", "lora_rank", "learning_rate"}). `project_id` is filled
    in automatically.

    The training file must already be uploaded, and the resource profile decides the GPU you
    are paying for - list them with `dynamiq resource-profiles list` rather than guessing an id.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/fine-tuning/jobs", json=body))


@finetuning.command("job")
@click.argument("job_id")
@with_api_and_settings
def get_job(*, api: ApiClient, settings: Settings, job_id: str):
    """Fetch one job, including its status."""
    echo_response(api.get(f"/v1/fine-tuning/jobs/{job_id}"))


@finetuning.command("pods")
@click.argument("job_id")
@with_api_and_settings
def job_pods(*, api: ApiClient, settings: Settings, job_id: str):
    """The pods running a job - the pod name is what `logs` needs."""
    echo_response(api.get(f"/v1/fine-tuning/jobs/{job_id}/pods"))


@finetuning.command("logs")
@click.argument("job_id")
@click.argument("pod_name")
@with_api_and_settings
def job_logs(*, api: ApiClient, settings: Settings, job_id: str, pod_name: str):
    """Training logs for one pod. This is where a failing job explains itself."""
    echo_response(api.get(f"/v1/fine-tuning/jobs/{job_id}/pods/{pod_name}/logs"))


@finetuning.command("cancel")
@click.argument("job_id")
@click.confirmation_option(prompt="Cancel this fine-tuning job?")
@with_api_and_settings
def cancel_job(*, api: ApiClient, settings: Settings, job_id: str):
    """Cancel a running job. Compute already spent is not refunded."""
    echo_response(api.post(f"/v1/fine-tuning/jobs/{job_id}/cancel"))


@finetuning.command("adapters")
@pagination_options
@with_api_and_settings
def list_adapters(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List adapters - what a finished job produces, and what you deploy afterwards."""
    echo_list(
        api, "/v1/fine-tuning/adapters", {"project_id": require_project(settings)}, page, page_size, fetch_all, compact
    )


@finetuning.command("adapter")
@click.argument("adapter_id")
@with_api_and_settings
def get_adapter(*, api: ApiClient, settings: Settings, adapter_id: str):
    """Fetch one adapter."""
    echo_response(api.get(f"/v1/fine-tuning/adapters/{adapter_id}"))


# ---------------------------------------------------------------------------- inference


@inference.command("list")
@pagination_options
@with_api_and_settings
def list_inferences(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List model deployments in the current project."""
    echo_list(api, "/v1/inferences", {"project_id": require_project(settings)}, page, page_size, fetch_all, compact)


@inference.command("runtimes")
@pagination_options
@with_api_and_settings
def list_runtimes(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List inference runtimes. Run this FIRST - `create` needs an `inference_runtime_id`."""
    echo_list(api, "/v1/inference-runtimes", None, page, page_size, fetch_all, compact)


@inference.command("runtime-models")
@click.argument("inference_runtime_id")
@pagination_options
@with_api_and_settings
def list_runtime_models(
    *, api: ApiClient, settings: Settings, inference_runtime_id: str, page, page_size, fetch_all, compact
):
    """Models THIS runtime can actually serve - the only list `inference create` accepts.

    `GET /v1/models` is the general catalog and says nothing about runtimes; deployability is a
    (runtime, model) pair, and create rejects any pair that does not exist. This route reads
    that pairing.

    It is admin-scoped, so a project PAT may get 403. If it does, do NOT work through candidate
    models hoping one lands - every attempt is a rejected create, and a lucky one starts billing.
    Ask the user or an admin which models the runtime serves.
    """
    echo_list(
        api,
        f"/v1/admin/inferences/runtimes/{inference_runtime_id}/models",
        None,
        page,
        page_size,
        fetch_all,
        compact,
    )


@inference.command("create")
@click.argument("payload")
@with_api_and_settings
def create_inference(*, api: ApiClient, settings: Settings, payload: str):
    """Deploy a model as an inference endpoint.

    REQUIRED: `name`, `model_id`, `resource_profile_id`, `inference_runtime_id`, `engine`,
    `task`, `autoscaling`, `parameters`. `project_id` is filled in automatically.

    `task` is `text_generation`, `embedding` or `speech_to_text`. Get the runtime id from
    `inference runtimes` and the profile from `resource-profiles list`; both are UUIDs you must
    look up, not guess.

    `model_id` has to be one the RUNTIME supports, not merely one in the catalog - create
    validates the (runtime, model) pair and rejects anything else as unsupported. Check with
    `inference runtime-models <runtime_id>` first.

    This provisions GPUs and bills for them while it runs. Say so before creating one, and
    check `inference get` until it reports ready - `create` returns long before that.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/inferences", json=body))


@inference.command("get")
@click.argument("inference_id")
@with_api_and_settings
def get_inference(*, api: ApiClient, settings: Settings, inference_id: str):
    """Fetch one deployment, including status and endpoint."""
    echo_response(api.get(f"/v1/inferences/{inference_id}"))


@inference.command("update")
@click.argument("inference_id")
@click.argument("payload")
@with_api_and_settings
def update_inference(*, api: ApiClient, settings: Settings, inference_id: str, payload: str):
    """Update a deployment - typically autoscaling or parameters."""
    echo_response(api.put(f"/v1/inferences/{inference_id}", json=read_json_arg(payload)))


@inference.command("pods")
@click.argument("inference_id")
@with_api_and_settings
def inference_pods(*, api: ApiClient, settings: Settings, inference_id: str):
    """The pods serving this deployment."""
    echo_response(api.get(f"/v1/inferences/{inference_id}/pods"))


@inference.command("logs")
@click.argument("inference_id")
@click.argument("pod_name")
@with_api_and_settings
def inference_logs(*, api: ApiClient, settings: Settings, inference_id: str, pod_name: str):
    """Logs for one serving pod - where a model that will not load says why."""
    echo_response(api.get(f"/v1/inferences/{inference_id}/pods/{pod_name}/logs"))


@inference.command("test")
@click.argument("inference_id")
@click.argument("prompt")
@with_api_and_settings
def test_inference(*, api: ApiClient, settings: Settings, inference_id: str, prompt: str):
    """Send one chat completion - the proof a deployment actually serves.

    The endpoint is OpenAI-compatible, so anything that speaks that API can call it directly.
    Only meaningful for a `text_generation` deployment.
    """
    echo_response(
        api.post(
            f"/v1/inferences/{inference_id}/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}]},
            # A completion on a large model outlasts the default timeout, and a retry starts a
            # second generation on the deployment's GPU rather than waiting for the first.
            timeout=EXECUTION_TIMEOUT,
            retry=False,
        )
    )


@inference.command("restart")
@click.argument("inference_id")
@with_api_and_settings
def restart_inference(*, api: ApiClient, settings: Settings, inference_id: str):
    """Restart the deployment. Requests fail while it comes back."""
    echo_response(api.post(f"/v1/inferences/{inference_id}/restart"))


@inference.command("delete")
@click.argument("inference_id")
@click.confirmation_option(prompt="Delete this deployment? Anything calling it will break.")
@with_api_and_settings
def delete_inference(*, api: ApiClient, settings: Settings, inference_id: str):
    """Tear down the deployment and stop the billing."""
    echo_response(api.delete(f"/v1/inferences/{inference_id}"))


# ----------------------------------------------------------------------------- database


@database.command("list")
@pagination_options
@with_api_and_settings
def list_databases(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List deployed databases in the current project."""
    echo_list(api, "/v1/databases", {"project_id": require_project(settings)}, page, page_size, fetch_all, compact)


@database.command("create")
@click.argument("payload")
@with_api_and_settings
def create_database(*, api: ApiClient, settings: Settings, payload: str):
    """Deploy a vector database.

    REQUIRED: `name`, `resource_profile_id`, `engine`, `engine_version`, `parameters`.
    `project_id` is filled in automatically. `engine` is `weaviate` - it is the only one the
    platform deploys, so a request for "a vector DB" means this.

    Like an inference deployment this provisions infrastructure and bills while it runs.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/databases", json=body))


@database.command("get")
@click.argument("database_id")
@with_api_and_settings
def get_database(*, api: ApiClient, settings: Settings, database_id: str):
    """Fetch one database, including its status."""
    echo_response(api.get(f"/v1/databases/{database_id}"))


@database.command("credentials")
@click.argument("database_id")
@with_api_and_settings
def database_credentials(*, api: ApiClient, settings: Settings, database_id: str):
    """Print the connection credentials.

    These are SECRETS. Do not paste the output into a report, a file the user can see, or a
    later command line - use them to build the connection and refer to them by name.
    """
    echo_response(api.get(f"/v1/databases/{database_id}/credentials"))


@database.command("delete")
@click.argument("database_id")
@click.confirmation_option(prompt="Delete this database and its data?")
@with_api_and_settings
def delete_database(*, api: ApiClient, settings: Settings, database_id: str):
    """Tear down the database. The data goes with it."""
    echo_response(api.delete(f"/v1/databases/{database_id}"))
