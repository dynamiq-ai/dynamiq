"""The runs behind the golden fixtures in ``fixtures/``.

Each scenario runs once with an in-memory span exporter and a ``TracingCallbackHandler`` attached to the same run,
and returns both records. Model calls are scripted (only the provider call is replaced), so no scenario needs
credentials. ``generate_fixtures.py`` writes the records to disk; ``test_golden_freshness.py`` reruns them.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from litellm import ModelResponse
from litellm.types.utils import Usage
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.checkpoints.types import RunPausedException
from dynamiq.flows import Flow
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.agents.hooks import CallLimitHook
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.operators.operators import Choice, ChoiceCondition, ChoiceOption, ConditionOperator, Map, Pass
from dynamiq.nodes.operators.sub_workflow import SubWorkflow
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.nodes.utils import Input, Output
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.telemetry import ContentCaptureMode, TracingConfig
from tests.helpers.agent_hooks import ExplodingTool, SearchTool, build_agent, encode, final_step, tool_step
from tests.integration.telemetry.equivalence import ScenarioOptions

TAGS = ["golden"]
SOURCE_ID = "source-golden"
SESSION_ID = "session-golden"
USAGE = (11, 7)
# A fixed resource keeps SDK versions out of the fixtures.
RESOURCE = Resource({"service.name": "dynamiq-golden-fixtures"})


class ScenarioResult(BaseModel):
    """The finished spans of one run and the runs a handler recorded for it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spans: list[ReadableSpan]
    handler: TracingCallbackHandler


class Scenario(BaseModel):
    """A named run whose spans and handler runs become one fixture directory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    options: ScenarioOptions
    run: Callable[[RunnableConfig], None]
    tracing: dict[str, Any] = Field(default_factory=dict)

    def execute(self) -> ScenarioResult:
        exporter = InMemorySpanExporter()
        provider = TracerProvider(resource=RESOURCE)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        handler = TracingCallbackHandler(
            tags=TAGS,
            source_id=SOURCE_ID,
            session_id=SESSION_ID,
            compact_llm_prompts_enabled=self.options.prompt_compaction_enabled,
        )
        tracing = TracingConfig(
            tracer_provider=provider,
            tags=TAGS,
            source_id=SOURCE_ID,
            session_id=SESSION_ID,
            node_config_enabled=self.options.node_config_enabled,
            prompt_compaction_enabled=self.options.prompt_compaction_enabled,
            content_capture=(
                ContentCaptureMode.SPAN_ONLY if self.options.content_capture_enabled else ContentCaptureMode.NO_CONTENT
            ),
        )
        try:
            self.run(RunnableConfig(callbacks=[handler], tracing=tracing))
            return ScenarioResult(spans=list(exporter.get_finished_spans()), handler=handler)
        finally:
            provider.shutdown()


# Scripted model calls


def _response(output: dict[str, Any]) -> ModelResponse:
    message = {"content": output["content"]}
    finish = "stop"
    if output.get("tool_calls"):
        message["tool_calls"] = output["tool_calls"]
        finish = "tool_calls"
    return ModelResponse(
        choices=[{"message": message, "finish_reason": finish}],
        usage=Usage(prompt_tokens=USAGE[0], completion_tokens=USAGE[1], total_tokens=sum(USAGE)),
    )


@contextmanager
def scripted_llm(mode: InferenceMode, steps: list[tuple | dict]) -> Iterator[None]:
    """Make every LLM node answer with ``steps`` in turn: agent steps encoded for ``mode``, or a raw output dict."""
    lock = threading.Lock()
    state = {"n": 0}

    def completion(*args, **kwargs):
        with lock:
            index = state["n"]
            state["n"] += 1
        step = steps[min(index, len(steps) - 1)]
        return _response(step if isinstance(step, dict) else encode(mode, step, index))

    with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=completion):
        yield


def _llm(node_id: str = "llm", name: str = "LLM", content: str = "Say {{question}}") -> OpenAI:
    return OpenAI(
        id=node_id,
        name=name,
        connection=connections.OpenAI(id="openai-connection", api_key="test-api-key"),
        model="gpt-4o",
        max_tokens=100,
        temperature=0,
        prompt=Prompt(
            id=f"{node_id}-prompt",
            messages=[
                Message(role=MessageRole.SYSTEM, content="Answer briefly."),
                Message(role=MessageRole.USER, content=content),
            ],
        ),
    )


def _check(result: Any, status: RunnableStatus) -> None:
    if result.status != status:
        raise AssertionError(f"the scenario ended with {result.status}, not {status}: {result.error}")


# Test nodes


class Flaky(Node):
    """Fails on its first attempt and succeeds on the next."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "flaky"
    _attempts: int = PrivateAttr(default=0)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self._attempts += 1
        if self._attempts == 1:
            raise ValueError("the first attempt fails")
        return {"attempts": self._attempts}


class Failing(Node):
    group: NodeGroup = NodeGroup.UTILS
    name: str = "failing"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        raise RuntimeError("the node broke")


class Pausing(Node):
    """Pauses the run through the flow's checkpoint context."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "pausing"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        context = config.checkpoint.context if config and config.checkpoint else None
        if context is not None and context.pause_run(self.id):
            raise RunPausedException("waiting for approval")
        return {"resumed": True}


class Detached(Node):
    """Calls an API inside a span of other instrumentation, and inside that span runs a node without naming its own
    run as the parent run, which makes that node run a root."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "detached"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        tracer = config.tracing.tracer_provider.get_tracer("application")
        with tracer.start_as_current_span("http request"):
            return Input(id="inner", name="inner").run(input_data=input_data, config=config).output


# Scenarios


def llm_workflow(config: RunnableConfig) -> None:
    start = Input(id="input", name="input")
    llm = _llm()
    llm.depends = [NodeDependency(start)]
    llm.input_transformer = InputTransformer(selector={"question": "$.input.output.question"})
    end = Output(id="output", name="output", depends=[NodeDependency(llm)])
    workflow = Workflow(id="workflow", version="3", flow=Flow(id="flow", nodes=[start, llm, end]))
    with scripted_llm(InferenceMode.DEFAULT, [{"content": "Hello there."}]):
        _check(workflow.run(input_data={"question": "hello"}, config=config), RunnableStatus.SUCCESS)


def _agent_in_workflow(config: RunnableConfig, mode: InferenceMode, steps: list[tuple], **agent_kwargs) -> None:
    agent = build_agent(id="agent", name="assistant", inference_mode=mode, **agent_kwargs)
    workflow = Workflow(id="workflow", flow=Flow(id="flow", nodes=[agent]))
    with scripted_llm(mode, steps):
        _check(workflow.run(input_data={"input": "What is the answer?"}, config=config), RunnableStatus.SUCCESS)


def agent_function_calling(config: RunnableConfig) -> None:
    # The second search is over the hook's limit and blocked, which adds hook events to the agent run.
    _agent_in_workflow(
        config,
        InferenceMode.FUNCTION_CALLING,
        [tool_step("search", query="q1"), tool_step("search", query="q2"), final_step("forty-two")],
        tools=[SearchTool(id="search", result="forty-two")],
        hooks=[CallLimitHook(tools=["search"], max_per_run=1)],
        description="Answers questions.",
    )


def agent_xml(config: RunnableConfig) -> None:
    # An agent run on its own: its run is the root.
    agent = build_agent(id="agent", name="assistant", inference_mode=InferenceMode.XML, tools=[SearchTool(id="search")])
    with scripted_llm(InferenceMode.XML, [tool_step("search", query="q1"), final_step("done")]):
        _check(agent.run(input_data={"input": "What is the answer?"}, config=config), RunnableStatus.SUCCESS)


def agent_tool_failure(config: RunnableConfig) -> None:
    _agent_in_workflow(
        config,
        InferenceMode.FUNCTION_CALLING,
        [tool_step("http", query="x"), final_step("the API is down")],
        tools=[ExplodingTool(id="http")],
    )


def retry(config: RunnableConfig) -> None:
    flaky = Flaky(id="flaky", error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0.01))
    after = Pass(id="after", name="after", depends=[NodeDependency(flaky)])
    workflow = Workflow(id="workflow", flow=Flow(id="flow", nodes=[flaky, after]))
    _check(workflow.run(input_data={"value": 1}, config=config), RunnableStatus.SUCCESS)


def choice_skipped(config: RunnableConfig) -> None:
    start = Input(id="input", name="input")
    choice = Choice(
        id="choice",
        name="choice",
        options=[
            ChoiceOption(
                id="yes",
                name="yes",
                condition=ChoiceCondition(variable="$.flag", operator=ConditionOperator.BOOLEAN_EQUALS, value=True),
            ),
            ChoiceOption(id="no", name="no", condition=None),
        ],
        input_transformer=InputTransformer(selector={"flag": "$.input.output.flag"}),
        depends=[NodeDependency(start)],
    )
    taken = Pass(id="taken", name="taken", depends=[NodeDependency(choice, option="yes")])
    skipped = Pass(id="skipped", name="skipped", depends=[NodeDependency(choice, option="no")])
    workflow = Workflow(id="workflow", flow=Flow(id="flow", nodes=[start, choice, taken, skipped]))
    _check(workflow.run(input_data={"flag": True}, config=config), RunnableStatus.SUCCESS)


def map_items(config: RunnableConfig) -> None:
    mapper = Map(id="map", name="mapper", node=Pass(id="item", name="item"), max_workers=1)
    workflow = Workflow(id="workflow", flow=Flow(id="flow", nodes=[mapper]))
    _check(workflow.run(input_data={"input": [{"n": 1}, {"n": 2}, {"n": 3}]}, config=config), RunnableStatus.SUCCESS)


def sub_workflow(config: RunnableConfig) -> None:
    inner_start = Input(id="inner-input", name="inner-input")
    inner_end = Output(id="inner-output", name="inner-output", depends=[NodeDependency(inner_start)])
    node = SubWorkflow(id="sub", name="sub", flow=Flow(id="inner-flow", nodes=[inner_start, inner_end]))
    workflow = Workflow(id="workflow", flow=Flow(id="flow", nodes=[node]))
    _check(workflow.run(input_data={"x": 1}, config=config), RunnableStatus.SUCCESS)


def paused(config: RunnableConfig) -> None:
    before = Pass(id="before", name="before")
    gate = Pausing(id="gate", name="gate", depends=[NodeDependency(before)])
    flow = Flow(id="flow", nodes=[before, gate], checkpoint=CheckpointConfig(enabled=True, backend=InMemory()))
    result = Workflow(id="workflow", flow=flow).run(input_data={"value": 1}, config=config)
    _check(result, RunnableStatus.FAILURE)
    if result.error.type is not RunPausedException:
        raise AssertionError(f"the run did not pause: {result.error}")


def foreign_spans(config: RunnableConfig) -> None:
    tracer = config.tracing.tracer_provider.get_tracer("application")
    workflow = Workflow(id="workflow", flow=Flow(id="flow", nodes=[Detached(id="detached")]))
    with tracer.start_as_current_span("request"):
        _check(workflow.run(input_data={"value": 1}, config=config), RunnableStatus.SUCCESS)


def prompt_compaction(config: RunnableConfig) -> None:
    _agent_in_workflow(
        config,
        InferenceMode.FUNCTION_CALLING,
        [tool_step("search", query=f"q{index}") for index in range(3)] + [final_step("done")],
        tools=[SearchTool(id="search", result="forty-two")],
    )


def failed_workflow(config: RunnableConfig) -> None:
    broken = Failing(id="broken", name="broken")
    after = Pass(id="after", name="after", depends=[NodeDependency(broken)])
    workflow = Workflow(id="workflow", flow=Flow(id="flow", nodes=[broken, after]))
    _check(workflow.run(input_data={"value": 1}, config=config), RunnableStatus.FAILURE)


def content_capture_off(config: RunnableConfig) -> None:
    # A blocked call, a failing tool, a node that fails and returns, and a skipped branch: everything that has
    # content, recorded without it.
    agent = build_agent(
        id="agent",
        name="assistant",
        inference_mode=InferenceMode.FUNCTION_CALLING,
        tools=[SearchTool(id="search", result="forty-two"), ExplodingTool(id="http")],
        hooks=[CallLimitHook(tools=["search"], max_per_run=1)],
    )
    broken = Failing(id="broken", name="broken", error_handling=ErrorHandling(behavior=Behavior.RETURN))
    choice = Choice(
        id="choice",
        name="choice",
        options=[
            ChoiceOption(
                id="yes",
                name="yes",
                condition=ChoiceCondition(
                    variable="$.input", operator=ConditionOperator.STRING_EQUALS, value="What is the answer?"
                ),
            ),
            ChoiceOption(id="no", name="no", condition=None),
        ],
    )
    taken = Pass(id="taken", name="taken", depends=[NodeDependency(choice, option="yes")])
    skipped = Pass(id="skipped", name="skipped", depends=[NodeDependency(choice, option="no")])
    workflow = Workflow(id="workflow", flow=Flow(id="flow", nodes=[agent, broken, choice, taken, skipped]))
    steps = [
        tool_step("search", query="q1"),
        tool_step("search", query="q2"),
        tool_step("http", query="x"),
        final_step("forty-two"),
    ]
    with scripted_llm(InferenceMode.FUNCTION_CALLING, steps):
        _check(workflow.run(input_data={"input": "What is the answer?"}, config=config), RunnableStatus.SUCCESS)


SCENARIOS: list[Scenario] = [
    Scenario(
        name="llm_workflow",
        options=ScenarioOptions(
            description="Input -> LLM -> Output in a workflow, with node configuration capture on.",
            node_config_enabled=True,
        ),
        run=llm_workflow,
    ),
    Scenario(
        name="agent_function_calling",
        options=ScenarioOptions(
            description=(
                "An agent with a tool in a workflow, FUNCTION_CALLING mode; a call-limit hook blocks the second "
                "tool call."
            )
        ),
        run=agent_function_calling,
    ),
    Scenario(
        name="agent_xml",
        options=ScenarioOptions(description="An agent with a tool run on its own (a root node run), XML mode."),
        run=agent_xml,
    ),
    Scenario(
        name="agent_tool_failure",
        options=ScenarioOptions(description="An agent whose tool fails; the agent recovers and answers."),
        run=agent_tool_failure,
    ),
    Scenario(
        name="retry",
        options=ScenarioOptions(description="A node that fails its first attempt and succeeds on the retry."),
        run=retry,
    ),
    Scenario(
        name="choice_skipped",
        options=ScenarioOptions(
            description="A Choice whose second branch is not taken, with node configuration capture on.",
            node_config_enabled=True,
        ),
        run=choice_skipped,
    ),
    Scenario(
        name="map",
        options=ScenarioOptions(description="A Map over three items."),
        run=map_items,
    ),
    Scenario(
        name="sub_workflow",
        options=ScenarioOptions(description="A sub-workflow node that runs an inner flow."),
        run=sub_workflow,
    ),
    Scenario(
        name="paused",
        options=ScenarioOptions(description="A workflow that pauses on a checkpoint."),
        run=paused,
    ),
    Scenario(
        name="foreign_spans",
        options=ScenarioOptions(
            description=(
                "Spans of other instrumentation: the workflow runs inside an application span, and a node runs "
                "another node without naming a parent run inside an HTTP client span, which makes that run a root."
            )
        ),
        run=foreign_spans,
    ),
    Scenario(
        name="prompt_compaction",
        options=ScenarioOptions(
            description="An agent with four model calls, prompt compaction on for the spans and the handler.",
            prompt_compaction_enabled=True,
        ),
        run=prompt_compaction,
    ),
    Scenario(
        name="failed_workflow",
        options=ScenarioOptions(description="A workflow that fails: a node raises and the node after it is skipped."),
        run=failed_workflow,
    ),
    Scenario(
        name="content_capture_off",
        options=ScenarioOptions(
            description=(
                "Content capture off: an agent with a blocked call and a failing tool, a node that fails and "
                "returns, and a Choice branch that is not taken."
            ),
            content_capture_enabled=False,
        ),
        run=content_capture_off,
    ),
]

SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
