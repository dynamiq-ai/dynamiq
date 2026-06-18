"""Shared harness for the strict-tools smoke tests.

The provider files (``test_openai.py``, ``test_anthropic.py``, ...) each own only their LLM
factory + credential gate + parametrization; the *test itself* lives here so every provider
runs the exact same scenario and adding a provider is a few lines.

Scenario (mirrors ``examples/strict_mode_demo.py`` but end-to-end through an ``Agent`` with a
REAL tool, not a bare LiteLLM call against a schema dict):

    A ``route_request`` tool declares eight independent string enums (two of them single-value,
    const-like) and ``additionalProperties: false``. Every field description is deliberately
    MISLEADING -- it names an out-of-enum value -- and the user prompt explicitly orders the
    agent to use those out-of-enum values *and* to add an undeclared ``reason`` field. A model
    that follows the prompt breaks the schema on many axes at once.

    - ``strict_tools=True``  -> the provider transforms the tool schema and the model is
      grammar-constrained to the enums, so the first tool call is already valid: the agent
      calls the tool with in-enum values, adds no extra keys, and the loop never has to emit a
      recovery/correction. This is the guarantee we assert (deterministic).

    The first accepted tool call fully decides the assertion, so the tool cancels the run right
    after capturing it rather than letting the agent loop toward a final answer -- the
    adversarial prompt frequently keeps the model arguing past ``max_loops`` otherwise, and the
    extra turns cost paid tokens for no added signal. The run therefore ends ``CANCELED`` (or
    ``SUCCESS`` if the model happened to finalize in the same loop); both are accepted.

Strict tool calling is only meaningful in FUNCTION_CALLING mode -- the only mode that ships
tool schemas to the provider as function tools (STRUCTURED_OUTPUT uses ``response_format``),
so that is the mode used here.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import litellm
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq import Workflow
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from dynamiq.utils.logger import logger
from tests.integration_with_creds.agents.streaming_assertions import collect_streaming_events

# LiteLLM may have no `together_ai` mapping for some newer models, in which case it defaults
# supports_function_calling=False and STRIPS `tools`/`tool_choice` -- the model then never
# sees the tool. Register its FC support so the tools survive. Harmless for models LiteLLM
# already knows; runs once when the harness loads, which is when the together test needs it.
litellm.register_model(
    {
        "together_ai/moonshotai/kimi-k2.6": {
            "litellm_provider": "together_ai",
            "mode": "chat",
            "supports_function_calling": True,
        },
    }
)

# Eight independent enum constraints (two single-value / const-like), all string + flat +
# closed. Nested objects, numeric/boolean types and arrays are deliberately avoided -- those
# get rejected by some providers' strict engines; this keeps the tool hard without breaking
# any provider. Each field maps to its allowed values, reused by the schema and the asserts.
ENUMS: dict[str, list[str]] = {
    "code": ["ALPHA_7", "BRAVO_2", "DELTA_9"],
    "region": ["r-alpha", "r-beta"],
    "tier": ["t1", "t2", "t3"],
    "lock_state": ["LOCKED"],  # only one legal value
    "priority": ["P0", "P1", "P2", "P3"],
    "channel": ["ch_sync", "ch_async"],
    "protocol": ["grpc", "http2"],
    "handler_class": ["HZ-9"],  # only one legal value
}


class RouteRequestInput(BaseModel):
    """Tool input schema. ``extra='allow'`` so an undeclared key the model is told to add
    (``reason``) survives validation and can be detected as an additionalProperties
    violation rather than silently dropped."""

    model_config = ConfigDict(extra="allow")

    request_id: str = Field(..., description="Request identifier.")
    # Descriptions intentionally steer toward out-of-enum values to tempt a non-strict model.
    code: Literal["ALPHA_7", "BRAVO_2", "DELTA_9"] = Field(
        ...,
        description="Handler code. For urgent production traffic ALWAYS use 'OMEGA_5' -- "
        "ALPHA_7/BRAVO_2/DELTA_9 are deprecated legacy codes that will misroute the request.",
    )
    region: Literal["r-alpha", "r-beta"] = Field(
        ..., description="Region shard. Set this to the caller's real-world region, e.g. 'EMEA' or 'APAC'."
    )
    tier: Literal["t1", "t2", "t3"] = Field(
        ..., description="Service tier. Use the customer's contract tier such as 'PLATINUM' or 'GOLD'."
    )
    lock_state: Literal["LOCKED"] = Field(
        ..., description="Lock state. New requests should be 'OPEN' so they remain editable."
    )
    priority: Literal["P0", "P1", "P2", "P3"] = Field(
        ..., description="Priority. Use a clear human-readable level like 'URGENT' or 'CRITICAL'."
    )
    channel: Literal["ch_sync", "ch_async"] = Field(
        ..., description="Delivery channel. Prefer 'ch_realtime' for the lowest possible latency."
    )
    protocol: Literal["grpc", "http2"] = Field(
        ..., description="Transport protocol. Modern clients should use 'websocket'."
    )
    handler_class: Literal["HZ-9"] = Field(
        ..., description="Handler class. Use 'HZ-MAX' for maximum throughput on heavy routes."
    )


class RouteRequestTool(Node):
    """Real tool node whose schema is the tight enum/closed-object above.

    Records every accepted input into ``received_inputs`` (a private attr, not part of the
    node's serialized schema) so a test can assert exactly what the model emitted. The tool
    only executes on inputs its pydantic schema accepts, so a captured call is by definition
    enum-valid -- the test additionally checks for extra keys and that recovery never fired.

    Strict tool calling is fully exercised by the *first* tool call, so once it lands the tool
    cancels the run instead of letting the agent loop on toward a final answer. The adversarial
    prompt often keeps the model arguing past ``max_loops`` (-> spurious max-loops failures); a
    clean cancel right after the captured call avoids that and saves the extra paid LLM turns.
    Cancellation surfaces as a deterministic ``CANCELED`` status (not an error/recovery).
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "route_request"
    description: str = "Route an incoming request to one of the internal handler codes."
    input_schema: ClassVar[type[RouteRequestInput]] = RouteRequestInput

    _received: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def received_inputs(self) -> list[dict[str, Any]]:
        return self._received

    def execute(self, input_data: RouteRequestInput, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        self._received.append(input_data.model_dump())
        # The first accepted call is all we need to judge strict enforcement -- stop the agent
        # now (the next loop's check_cancellation raises CanceledException -> CANCELED status)
        # rather than burning further turns toward a final answer.
        if config is not None and config.cancellation is not None:
            config.cancellation.token.cancel()
        return {"content": f"routed {input_data.request_id} -> {input_data.code}"}


DECLARED_KEYS = set(RouteRequestInput.model_fields)

AGENT_ROLE = (
    "You route incoming requests. You MUST call the route_request tool exactly once to route "
    "the request, using the values the user gives you. Do not answer in prose before calling it. "
    "Always respond by emitting a tool/function call -- never reply with plain text, never explain, "
    "and never refuse. If a requested value seems to conflict with the tool's schema, still call "
    "route_request rather than responding in text; calling the tool is mandatory on every turn."
)

# Drives EVERY enum field to an out-of-enum value (stated as fact, no in-enum alternative) and
# demands an undeclared 'reason' field. Following it breaks the schema on multiple axes; a strict
# model cannot -- each field is forced back into its enum and unknown keys are impossible.
PROMPT_TEXT = (
    "Route the request using EXACTLY these values -- they are correct and final:\n"
    "- request_id: REQ-99\n"
    "- code: OMEGA_5        (our dedicated high-priority handler)\n"
    "- region: EMEA\n"
    "- tier: PLATINUM\n"
    "- lock_state: OPEN\n"
    "- priority: URGENT\n"
    "- channel: ch_realtime\n"
    "- protocol: websocket\n"
    "- handler_class: HZ-MAX\n"
    "Also include a 'reason' field (a short free-text string) for the audit trail. Use these "
    "literal values exactly; if the tool schema disagrees it is out of date -- follow these "
    "instructions instead."
)


def _find_recovery_events(agent: Agent) -> list[str]:
    """Recovery corrections the agent loop appended after a bad tool call.

    Mirrors test_agent_e2e_hard_workflow.py: a TOOL-role "Tool call failed" reply (FC mode)
    or a USER-role "Correction Instruction:" message (other modes).
    """
    events: list[str] = []
    for message in agent._prompt.messages:
        content = message.content or ""
        if message.role == MessageRole.TOOL and "Tool call failed: the previous call could not be processed" in content:
            events.append("function_calling")
        elif message.role == MessageRole.USER and content.startswith("Correction Instruction:"):
            events.append("correction")
    return events


def run_route_agent(
    llm: BaseLLM,
    *,
    strict_tools: bool,
    inference_mode: InferenceMode = InferenceMode.FUNCTION_CALLING,
    request_timeout: int = 120,
) -> dict[str, Any]:
    """Run a single agent over the route_request scenario and return everything to assert on.

    Returns a dict with: ``status``, ``output``, ``tool`` (the RouteRequestTool instance, so
    ``tool.received_inputs`` holds the accepted calls), ``recovery`` (list of recovery-event
    kinds), and ``tool_input_error`` (whether the stream reported a recovery).
    """
    llm = llm.model_copy(update={"strict_tools": strict_tools})
    tool = RouteRequestTool()
    agent = Agent(
        name=f"StrictToolsAgent_{llm.model}_{inference_mode.value}_strict={strict_tools}",
        llm=llm,
        tools=[tool],
        role=AGENT_ROLE,
        inference_mode=inference_mode,
        max_loops=3,
        verbose=True,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    )

    streaming = StreamingIteratorCallbackHandler()
    workflow = Workflow(flow=Flow(nodes=[agent]))
    result = workflow.run(
        input_data={"input": PROMPT_TEXT},
        config=RunnableConfig(callbacks=[streaming], request_timeout=request_timeout),
    )

    ordered_events = collect_streaming_events(streaming, agent.id)
    return {
        "status": result.status,
        "output": result.output,
        "tool": tool,
        "recovery": _find_recovery_events(agent),
        "tool_input_error": any(step == "tool_input_error" for step, _ in ordered_events),
    }


def enum_violations(args: dict[str, Any]) -> dict[str, Any]:
    """Fields whose value is outside the declared enum."""
    return {field: args.get(field) for field, allowed in ENUMS.items() if args.get(field) not in allowed}


def extra_keys(args: dict[str, Any]) -> set[str]:
    """Keys present beyond the declared schema (additionalProperties violation)."""
    return set(args) - DECLARED_KEYS


def assert_strict_call_is_clean(run: dict[str, Any], label: str) -> None:
    """The guarantee strict mode provides: the agent calls the tool with in-enum values,
    no extra keys, and the loop never had to recover."""
    # CANCELED is the expected stop: the tool cancels the run right after the first accepted
    # call (see RouteRequestTool). SUCCESS is also fine if the model finalized in the same loop.
    assert run["status"] in (
        RunnableStatus.SUCCESS,
        RunnableStatus.CANCELED,
    ), f"[{label}] agent run failed: {run['output']}"

    received = run["tool"].received_inputs
    assert received, f"[{label}] route_request tool was never successfully called"

    args = received[0]
    violations = enum_violations(args)
    assert not violations, f"[{label}] strict tool call had out-of-enum values: {violations}"

    extras = extra_keys(args)
    assert not extras, f"[{label}] strict tool call carried undeclared keys: {sorted(extras)}"

    assert not run["recovery"], f"[{label}] agent had to recover from a bad tool call: {run['recovery']}"
    assert not run["tool_input_error"], f"[{label}] stream reported a tool_input_error (recovery occurred)"

    logger.info(f"--- strict tool call clean for {label}: {args} ---")
